from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedpost.analysis.finding2.hooks import CaptureHiddenStates
from fedpost.analysis.finding2.synthesizers import (
    AnchorCorrectionSynthesizer,
    CachedMeanSynthesizer,
    DistilledRandomFeatureSynthesizer,
    ExactSynthesizer,
    LowRankSynthesizer,
    NeighborLinearSynthesizer,
    Synthesizer,
)
from fedpost.analysis.finding2.static_reconstruction import HiddenTraceLoader
from fedpost.data.collators.sft_collator import SFTCollator
from fedpost.data.processors import SFTSample


@dataclass
class Finding2TrainConfig:
    model_name_or_path: str
    output_dir: str
    train_dataset_name: str = "tatsu-lab/alpaca"
    train_dataset_config: str | None = None
    train_split: str = "train"
    eval_dataset_name: str | None = None
    eval_dataset_config: str | None = None
    eval_split: str = "train"
    train_samples: int = 128
    eval_samples: int = 64
    max_length: int = 512
    batch_size: int = 1
    train_steps: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 42
    dtype: str = "bfloat16"
    device: str | None = None
    device_map: str | None = None
    trust_remote_code: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: list[str] | None = None
    adapter_name: str = "default"


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_lora_model_and_tokenizer(cfg: Finding2TrainConfig):
    from fedpost.models.peft_utils import apply_lora, validate_lora_targets

    dtype = parse_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.device_map:
        kwargs["device_map"] = cfg.device_map
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **kwargs)
    if hasattr(model, "config"):
        model.config.use_cache = False
    if cfg.device_map is None:
        model.to(resolve_device(cfg.device))

    peft_cfg = SimpleNamespace(
        r=cfg.lora_r,
        alpha=cfg.lora_alpha,
        dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules or infer_lora_targets(cfg.model_name_or_path),
        adapter_name=cfg.adapter_name,
    )
    validate_lora_targets(model, peft_cfg.target_modules)
    model = apply_lora(model, peft_cfg)
    model.train()
    return model, tokenizer


def load_sft_samples(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    max_samples: int,
    seed: int,
) -> list[SFTSample]:
    if dataset_config:
        dataset = load_dataset(dataset_name, dataset_config, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    samples = []
    for rec in dataset:
        sample = record_to_sft_sample(dataset_name, rec)
        if sample is not None:
            samples.append(sample)
    rng = random.Random(seed)
    rng.shuffle(samples)
    return samples[:max_samples]


def record_to_sft_sample(dataset_name: str, record: dict[str, Any]) -> SFTSample | None:
    if dataset_name in {"tatsu-lab/alpaca", "yahma/alpaca-cleaned"}:
        instruction = _clean(record.get("instruction"))
        response = _clean(record.get("output") or record.get("response"))
        input_text = _clean(record.get("input"))
        if not instruction or not response:
            return None
        prompt = instruction if not input_text else f"{instruction}\n\nInput:\n{input_text}"
        return SFTSample(prompt=prompt, response=response, metadata={"source": dataset_name})

    if dataset_name == "databricks/databricks-dolly-15k":
        instruction = _clean(record.get("instruction"))
        response = _clean(record.get("response"))
        context = _clean(record.get("context"))
        if not instruction or not response:
            return None
        prompt = instruction if not context else f"{instruction}\n\nContext:\n{context}"
        return SFTSample(prompt=prompt, response=response, metadata={"source": dataset_name})

    if dataset_name == "openai/gsm8k":
        question = _clean(record.get("question"))
        answer = _clean(record.get("answer"))
        if not question or not answer:
            return None
        return SFTSample(prompt=question, response=answer, metadata={"source": dataset_name})

    raise ValueError(f"Unsupported SFT dataset for Finding2 downstream runner: {dataset_name}")


def make_dataloader(
    samples: list[SFTSample],
    tokenizer,
    max_length: int,
    batch_size: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return DataLoader(
        samples,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=SFTCollator(tokenizer, max_length=max_length),
    )


def fit_synthesizer_from_hidden_trace(
    hidden_trace_path: str,
    synthesizer: Synthesizer,
    layer_idx: int,
    calibration_vectors: int,
    seed: int,
    allow_small_calibration: bool = False,
) -> tuple[Synthesizer, dict[str, Any]]:
    if calibration_vectors <= 0:
        raise ValueError(f"calibration_vectors must be positive, got {calibration_vectors}.")

    payload = torch.load(hidden_trace_path, map_location="cpu")
    hidden = payload["hidden_states"]
    if hidden.ndim != 4:
        raise ValueError(
            f"Expected hidden_states with shape [samples, probes, layers, hidden_dim], "
            f"got {tuple(hidden.shape)} from {hidden_trace_path}."
        )
    hidden_flat = hidden.reshape(-1, hidden.shape[2], hidden.shape[3]).contiguous()
    num_vectors = hidden_flat.shape[0]
    used_vectors = calibration_vectors
    warning = None
    if calibration_vectors > num_vectors:
        if not allow_small_calibration:
            raise ValueError(
                f"Requested {calibration_vectors} calibration vectors from {hidden_trace_path}, "
                f"but only {num_vectors} are available. Re-run with more hidden-state samples, "
                f"lower --calibration_vectors, or pass --allow_small_calibration for an explicit "
                f"small-data run."
            )
        used_vectors = num_vectors
        warning = (
            f"Requested {calibration_vectors} calibration vectors from {hidden_trace_path}, "
            f"but only {num_vectors} are available; using all available vectors because "
            f"small calibration is explicitly allowed."
        )
        print(f"WARNING: {warning}")

    indices = list(range(num_vectors))
    random.Random(seed).shuffle(indices)
    loader = HiddenTraceLoader(hidden_flat, indices[:used_vectors])
    synthesizer.fit(loader, layer_idx=layer_idx, model=None)
    fit_info = {
        "hidden_trace_path": hidden_trace_path,
        "hidden_shape": list(hidden.shape),
        "flattened_vectors": num_vectors,
        "requested_calibration_vectors": calibration_vectors,
        "used_calibration_vectors": used_vectors,
        "allow_small_calibration": allow_small_calibration,
        "warning": warning,
    }
    return synthesizer, fit_info


def build_synthesizer(
    name: str,
    seed: int,
    low_rank_rank: int = 64,
    s5_rank: int = 64,
    s5_top_k: int = 64,
    distilled_feature_dim: int = 128,
    neighbor_k: int = 1,
) -> Synthesizer:
    if name == "S0_exact":
        return ExactSynthesizer(seed=seed)
    if name == "S1_cached":
        return CachedMeanSynthesizer(seed=seed)
    if name == "S2_low_rank":
        return LowRankSynthesizer(rank=low_rank_rank, seed=seed)
    if name == "S3_distilled":
        return DistilledRandomFeatureSynthesizer(feature_dim=distilled_feature_dim, seed=seed)
    if name == "S4_neighbor":
        return NeighborLinearSynthesizer(k=neighbor_k, seed=seed)
    if name == "S5_anchor_correction":
        return AnchorCorrectionSynthesizer(rank=s5_rank, top_k=s5_top_k, seed=seed)
    raise ValueError(f"Unsupported synthesizer: {name}")


def capture_trace_for_batch(model, batch: dict[str, torch.Tensor], layer_indices: list[int]) -> dict[int, torch.Tensor]:
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode(), CaptureHiddenStates(model, layer_indices=layer_indices, to_cpu=True) as capture:
            model(**batch, output_hidden_states=False, return_dict=True, use_cache=False)
            return capture.snapshot()
    finally:
        if was_training:
            model.train()


def evaluate_loss(model, dataloader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict[str, float]:
    was_training = model.training
    model.eval()
    losses = []
    try:
        with torch.inference_mode():
            for batch_idx, batch in enumerate(dataloader):
                batch = move_batch_to_device(batch, device)
                outputs = model(**batch, return_dict=True, use_cache=False)
                losses.append(float(outputs.loss.detach().cpu()))
                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break
    finally:
        if was_training:
            model.train()
    if not losses:
        raise RuntimeError("Evaluation produced no batches.")
    tensor = torch.tensor(losses, dtype=torch.float64)
    return {
        "loss_mean": float(tensor.mean().item()),
        "loss_std": float(tensor.std(unbiased=False).item()),
        "num_batches": len(losses),
    }


def save_lora_artifacts(model, tokenizer, output_dir: str, adapter_name: str = "default") -> dict[str, str]:
    from fedpost.models.peft_utils import save_adapter_pretrained, save_adapter_state

    os.makedirs(output_dir, exist_ok=True)
    adapter_state_path = os.path.join(output_dir, "adapter_state.pt")
    adapter_dir = os.path.join(output_dir, "adapter_model")
    save_adapter_state(model, adapter_state_path, adapter_name=adapter_name)
    save_adapter_pretrained(model, tokenizer, adapter_dir)
    return {
        "adapter_state_path": adapter_state_path,
        "adapter_dir": adapter_dir,
    }


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def infer_input_device(model, requested_device: str | None = None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    hf_device_map = getattr(model, "hf_device_map", {})
    for device in hf_device_map.values():
        if isinstance(device, int):
            return torch.device(f"cuda:{device}")
        if isinstance(device, str) and device not in {"cpu", "disk"}:
            return torch.device(device)
    return next(model.parameters()).device


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_dtype(dtype_name: str):
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def infer_lora_targets(model_name_or_path: str) -> list[str]:
    lowered = model_name_or_path.lower()
    if "gpt2" in lowered:
        return ["c_attn", "c_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
