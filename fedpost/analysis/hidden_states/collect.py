from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedpost.analysis.hidden_states.probe_data import (
    ProbeDatasetConfig,
    load_probe_texts,
)


@dataclass
class HiddenStateCollectionConfig:
    model_name_or_path: str
    dataset_kind: str
    dataset_name: str
    output_path: str
    dataset_split: str = "train"
    dataset_config: str | None = None
    text_field: str | None = None
    num_samples: int = 512
    max_length: int = 2048
    seed: int = 42
    dtype: str = "float16"
    trust_remote_code: bool = False
    use_flash_attn: bool = False
    device: str | None = None
    device_map: str | None = None
    capture_mode: str = "auto"
    downsample_hidden: int | None = None
    projection_seed: int = 1234
    dry_run: bool = False


def collect_hidden_states(cfg: HiddenStateCollectionConfig) -> dict[str, Any]:
    sample_limit = 16 if cfg.dry_run else cfg.num_samples
    probe_cfg = ProbeDatasetConfig(
        kind=cfg.dataset_kind,
        dataset_name=cfg.dataset_name,
        split=cfg.dataset_split,
        dataset_config=cfg.dataset_config,
        text_field=cfg.text_field,
        max_samples=sample_limit,
        seed=cfg.seed,
    )
    records = load_probe_texts(probe_cfg)
    if cfg.dry_run:
        records = records[:16]
    if not records:
        raise ValueError("No probe records were loaded.")

    storage_dtype = _parse_dtype(cfg.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(cfg)
    model.eval()

    capture_mode = cfg.capture_mode
    if capture_mode == "auto":
        capture_mode = "hidden_states"

    rng = random.Random(cfg.seed)
    hidden_rows = []
    sample_metadata = []
    projection = None

    hook_handles = None
    hook_context = None
    if capture_mode == "hooks":
        hook_handles, hook_context = _register_hidden_hooks(model)

    try:
        for sample_idx, record in enumerate(records):
            encoded = tokenizer(
                record["text"],
                return_tensors="pt",
                truncation=True,
                max_length=cfg.max_length,
                return_special_tokens_mask=True,
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            special_mask = encoded["special_tokens_mask"][0].tolist()
            positions = _select_probe_positions(
                input_ids=input_ids[0],
                special_mask=special_mask,
                rng=rng,
            )
            input_device = _infer_input_device(model, cfg.device, cfg.device_map)
            inputs = {
                "input_ids": input_ids.to(input_device),
                "attention_mask": attention_mask.to(input_device),
            }

            with torch.inference_mode():
                try:
                    if capture_mode == "hidden_states":
                        states = _forward_with_hidden_states(model, inputs, positions)
                    elif capture_mode == "hooks":
                        states = _forward_with_hooks(model, inputs, positions, hook_context)
                    else:
                        raise ValueError(f"Unsupported capture_mode: {cfg.capture_mode}")
                except RuntimeError as exc:
                    if cfg.capture_mode == "auto" and _is_oom(exc):
                        torch.cuda.empty_cache()
                        hook_handles, hook_context = _register_hidden_hooks(model)
                        capture_mode = "hooks"
                        states = _forward_with_hooks(model, inputs, positions, hook_context)
                    else:
                        raise

            states = states.cpu().to(torch.float32)
            if cfg.downsample_hidden is not None:
                if projection is None:
                    projection = _build_projection(
                        hidden_dim=states.shape[-1],
                        projected_dim=cfg.downsample_hidden,
                        seed=cfg.projection_seed,
                    )
                states = torch.matmul(states, projection)
            hidden_rows.append(states.to(storage_dtype))
            sample_metadata.append({
                "sample_idx": sample_idx,
                "record_id": record["id"],
                "source": record["source"],
                "kind": record["kind"],
                "text_preview": record["text"][:200],
                "seq_len": int(attention_mask.sum().item()),
                "probe_positions": positions,
            })

            if cfg.dry_run:
                print(
                    f"[dry_run] sample={sample_idx} seq_len={sample_metadata[-1]['seq_len']} "
                    f"positions={positions} states={tuple(states.shape)}"
                )

    finally:
        if hook_handles is not None:
            for handle in hook_handles:
                handle.remove()

    hidden_states = torch.stack(hidden_rows, dim=0)
    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    payload = {
        "hidden_states": hidden_states,
        "sample_metadata": sample_metadata,
        "config": asdict(cfg),
        "metadata": {
            "shape": tuple(hidden_states.shape),
            "dtype": str(hidden_states.dtype),
            "num_samples": hidden_states.shape[0],
            "num_probe_positions": hidden_states.shape[1],
            "num_layers": hidden_states.shape[2],
            "hidden_dim": hidden_states.shape[3],
            "capture_mode": capture_mode,
        },
    }
    if projection is not None:
        payload["projection"] = {
            "matrix": projection,
            "seed": cfg.projection_seed,
            "input_hidden_dim": projection.shape[0],
            "projected_hidden_dim": projection.shape[1],
        }
    torch.save(payload, cfg.output_path)

    summary_path = f"{os.path.splitext(cfg.output_path)[0]}.summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable_summary(payload), f, ensure_ascii=False, indent=2)

    print(f"saved hidden states: {cfg.output_path}")
    print(f"shape: {tuple(hidden_states.shape)} dtype: {hidden_states.dtype}")
    return payload


def _load_model(cfg: HiddenStateCollectionConfig):
    dtype = _parse_dtype(cfg.dtype)
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": cfg.trust_remote_code,
        "output_hidden_states": True,
    }
    if cfg.use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"
    if cfg.device_map:
        kwargs["device_map"] = cfg.device_map

    model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **kwargs)
    if cfg.device_map is None:
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
    if hasattr(model, "config"):
        model.config.output_hidden_states = True
        model.config.use_cache = False
    return model


def _forward_with_hidden_states(model, inputs: dict[str, torch.Tensor], positions: list[int]) -> torch.Tensor:
    outputs = model(
        **inputs,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states. Try --capture_mode hooks.")

    expected_layers = getattr(getattr(model, "config", None), "num_hidden_layers", None)
    if expected_layers is not None and len(hidden_states) != expected_layers + 1:
        raise RuntimeError(
            f"Incomplete hidden_states: got {len(hidden_states)}, expected {expected_layers + 1}."
        )

    rows = []
    for state in hidden_states:
        if state.shape[1] <= max(positions):
            raise RuntimeError(
                f"Hidden state sequence length {state.shape[1]} is shorter than probe position {max(positions)}."
            )
        rows.append(state[0, positions, :].detach().cpu())
    return torch.stack(rows, dim=1)


def _register_hidden_hooks(model):
    context = {"positions": None, "embedding": None, "layers": []}
    handles = []

    embedding = model.get_input_embeddings()

    def embedding_hook(_module, _inputs, output):
        context["embedding"] = _select_output(output, context["positions"])

    handles.append(embedding.register_forward_hook(embedding_hook))

    for _name, block in _resolve_transformer_blocks(model):
        def block_hook(_module, _inputs, output, ctx=context):
            ctx["layers"].append(_select_output(output, ctx["positions"]))

        handles.append(block.register_forward_hook(block_hook))

    return handles, context


def _forward_with_hooks(
    model,
    inputs: dict[str, torch.Tensor],
    positions: list[int],
    context: dict[str, Any],
) -> torch.Tensor:
    context["positions"] = positions
    context["embedding"] = None
    context["layers"] = []
    _ = model(**inputs, output_hidden_states=False, return_dict=True, use_cache=False)

    if context["embedding"] is None:
        raise RuntimeError("Embedding hook did not capture hidden states.")
    if not context["layers"]:
        raise RuntimeError("Layer hooks did not capture hidden states.")
    return torch.stack([context["embedding"], *context["layers"]], dim=1)


def _select_output(output, positions: list[int]) -> torch.Tensor:
    hidden = output[0] if isinstance(output, tuple) else output
    return hidden[0, positions, :].detach().cpu()


def _resolve_transformer_blocks(model) -> list[tuple[str, Any]]:
    for path in (
        "model.layers",
        "transformer.h",
        "gpt_neox.layers",
        "transformer.blocks",
        "decoder.layers",
    ):
        module = _get_submodule(model, path)
        if isinstance(module, (torch.nn.ModuleList, list, tuple)):
            return [(f"{path}.{idx}", block) for idx, block in enumerate(module)]
    raise RuntimeError("Could not resolve transformer blocks for hook capture.")


def _get_submodule(model, path: str):
    current = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _select_probe_positions(
    input_ids: torch.Tensor,
    special_mask: list[int],
    rng: random.Random,
) -> list[int]:
    non_special = [
        idx for idx in range(int(input_ids.numel()))
        if idx >= len(special_mask) or special_mask[idx] == 0
    ]
    if not non_special:
        non_special = list(range(int(input_ids.numel())))

    start = non_special[min(4, len(non_special) - 1)]
    middle = non_special[len(non_special) // 2]
    tail_start = max(0, math.floor(len(non_special) * 0.9) - 1)
    end = rng.choice(non_special[tail_start:])
    return [int(start), int(middle), int(end)]


def _build_projection(hidden_dim: int, projected_dim: int, seed: int) -> torch.Tensor:
    if projected_dim <= 0:
        raise ValueError("downsample_hidden must be positive.")
    if projected_dim >= hidden_dim:
        raise ValueError("downsample_hidden must be smaller than the original hidden dimension.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    projection = torch.randn(hidden_dim, projected_dim, generator=generator, dtype=torch.float32)
    projection /= math.sqrt(projected_dim)
    return projection


def _infer_input_device(model, requested_device: str | None, device_map: str | None) -> torch.device:
    if requested_device:
        return torch.device(requested_device)
    if device_map:
        hf_device_map = getattr(model, "hf_device_map", {})
        for device in hf_device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
    return next(model.parameters()).device


def _parse_dtype(dtype_name: str) -> torch.dtype:
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


def _is_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _jsonable_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "config": payload["config"],
        "metadata": payload["metadata"],
        "projection": (
            {
                "seed": payload["projection"]["seed"],
                "input_hidden_dim": payload["projection"]["input_hidden_dim"],
                "projected_hidden_dim": payload["projection"]["projected_hidden_dim"],
            }
            if "projection" in payload
            else None
        ),
        "num_sample_metadata": len(payload["sample_metadata"]),
        "first_sample": payload["sample_metadata"][0] if payload["sample_metadata"] else None,
    }


def parse_args() -> HiddenStateCollectionConfig:
    parser = argparse.ArgumentParser(description="Collect residual-stream hidden states.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset_kind", required=True, choices=["alpaca", "alpaca_eval", "gsm8k", "mmlu", "text"])
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--dataset_split", default="train")
    parser.add_argument("--dataset_config", default=None)
    parser.add_argument("--text_field", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="float16", choices=["float16", "fp16", "bfloat16", "bf16", "float32", "fp32"])
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--capture_mode", default="auto", choices=["auto", "hidden_states", "hooks"])
    parser.add_argument("--downsample_hidden", type=int, default=None)
    parser.add_argument("--projection_seed", type=int, default=1234)
    parser.add_argument("--dry_run", action="store_true")
    return HiddenStateCollectionConfig(**vars(parser.parse_args()))


def main() -> None:
    collect_hidden_states(parse_args())


if __name__ == "__main__":
    main()
