from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from fedpost.analysis.finding2.synthesizers import (
    AnchorCorrectionSynthesizer,
    CachedMeanSynthesizer,
    DistilledRandomFeatureSynthesizer,
    ExactSynthesizer,
    LowRankSynthesizer,
    NeighborLinearSynthesizer,
    Synthesizer,
)


@dataclass
class StaticReconstructionConfig:
    input_path: str
    output_dir: str
    layer_indices: list[int]
    calibration_vectors: int = 2000
    eval_vectors: int | None = None
    seed: int = 42
    quick: bool = False
    allow_small_data: bool = False
    fail_on_threshold: bool = False
    low_rank_ranks: list[int] | None = None
    s5_rank: int = 64
    s5_top_k: int = 64
    distilled_feature_dim: int = 128
    neighbor_k: int = 1


class HiddenTraceLoader:
    def __init__(self, hidden_flat: torch.Tensor, indices: list[int]):
        self.hidden_flat = hidden_flat
        self.indices = indices

    def __iter__(self):
        for idx in self.indices:
            sample = self.hidden_flat[idx]
            yield {layer_idx: sample[layer_idx] for layer_idx in range(sample.shape[0])}

    def __len__(self):
        return len(self.indices)


def run_static_reconstruction(cfg: StaticReconstructionConfig) -> dict[str, Any]:
    _set_determinism(cfg.seed)
    payload = torch.load(cfg.input_path, map_location="cpu")
    if "hidden_states" not in payload:
        raise KeyError(f"{cfg.input_path} does not contain 'hidden_states'.")

    hidden = payload["hidden_states"]
    if hidden.ndim != 4:
        raise ValueError(f"Expected hidden_states shape [N, P, L, D], got {tuple(hidden.shape)}")

    hidden_flat = hidden.reshape(-1, hidden.shape[2], hidden.shape[3]).contiguous()
    num_vectors, num_layers, hidden_dim = hidden_flat.shape
    for layer_idx in cfg.layer_indices:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} is invalid for {num_layers} layers.")

    calibration_vectors, eval_vectors = _resolve_split_sizes(cfg, num_vectors)
    all_indices = list(range(num_vectors))
    rng = random.Random(cfg.seed)
    rng.shuffle(all_indices)
    calibration_indices = all_indices[:calibration_vectors]
    eval_indices = all_indices[calibration_vectors:calibration_vectors + eval_vectors]

    if not eval_indices:
        raise ValueError("Evaluation split is empty.")

    os.makedirs(cfg.output_dir, exist_ok=True)
    result = {
        "config": asdict(cfg),
        "input_metadata": payload.get("metadata", {}),
        "split": {
            "num_vectors": num_vectors,
            "calibration_vectors": len(calibration_indices),
            "eval_vectors": len(eval_indices),
            "seed": cfg.seed,
        },
        "layers": {},
        "warnings": [],
    }

    for layer_idx in cfg.layer_indices:
        layer_result = _evaluate_layer(
            cfg=cfg,
            hidden_flat=hidden_flat,
            calibration_indices=calibration_indices,
            eval_indices=eval_indices,
            layer_idx=layer_idx,
        )
        result["layers"][str(layer_idx)] = layer_result
        result["warnings"].extend(layer_result["warnings"])

    output_json = os.path.join(cfg.output_dir, "static_reconstruction.json")
    output_pt = os.path.join(cfg.output_dir, "static_reconstruction.pt")
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, ensure_ascii=False, indent=2)
    torch.save(result, output_pt)

    if result["warnings"]:
        print("Finding2 static reconstruction warnings:")
        for warning in result["warnings"]:
            print(f"- {warning}")

    print(f"saved static reconstruction: {output_json}")
    return result


def _evaluate_layer(
    cfg: StaticReconstructionConfig,
    hidden_flat: torch.Tensor,
    calibration_indices: list[int],
    eval_indices: list[int],
    layer_idx: int,
) -> dict[str, Any]:
    layer_result = {
        "layer_idx": layer_idx,
        "synthesizers": {},
        "warnings": [],
    }

    for synthesizer in _build_synthesizers(cfg):
        calibration_loader = HiddenTraceLoader(hidden_flat, calibration_indices)
        synthesizer.fit(calibration_loader, layer_idx=layer_idx, model=None)
        metrics = _evaluate_synthesizer(synthesizer, hidden_flat, eval_indices, layer_idx)
        threshold_warnings = _threshold_warnings(layer_idx, synthesizer, metrics)
        if cfg.fail_on_threshold and threshold_warnings:
            raise RuntimeError("; ".join(threshold_warnings))
        layer_result["warnings"].extend(threshold_warnings)
        layer_result["synthesizers"][synthesizer.name] = metrics

    return layer_result


def _build_synthesizers(cfg: StaticReconstructionConfig) -> list[Synthesizer]:
    ranks = cfg.low_rank_ranks or [16, 32, 64, 128]
    synthesizers: list[Synthesizer] = [
        ExactSynthesizer(seed=cfg.seed),
        CachedMeanSynthesizer(seed=cfg.seed),
        *[LowRankSynthesizer(rank=rank, seed=cfg.seed) for rank in ranks],
        DistilledRandomFeatureSynthesizer(
            feature_dim=cfg.distilled_feature_dim,
            seed=cfg.seed,
        ),
        NeighborLinearSynthesizer(k=cfg.neighbor_k, seed=cfg.seed),
        AnchorCorrectionSynthesizer(
            rank=cfg.s5_rank,
            top_k=cfg.s5_top_k,
            seed=cfg.seed,
        ),
    ]
    return synthesizers


def _evaluate_synthesizer(
    synthesizer: Synthesizer,
    hidden_flat: torch.Tensor,
    eval_indices: list[int],
    layer_idx: int,
) -> dict[str, Any]:
    mu_values = []
    cosine_values = []
    for idx in eval_indices:
        sample = hidden_flat[idx]
        trace = {layer: sample[layer] for layer in range(sample.shape[0])}
        target = trace[layer_idx].to(torch.float32)
        anchor = synthesizer.build_anchor(trace)
        prediction = synthesizer.synthesize(anchor).to(torch.float32)
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"{synthesizer.name} produced shape {tuple(prediction.shape)} "
                f"for target shape {tuple(target.shape)}"
            )
        mu = torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1e-12)
        cosine = F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1), dim=-1)[0]
        mu_values.append(mu)
        cosine_values.append(cosine)

    mu_tensor = torch.stack(mu_values).to(torch.float64)
    cosine_tensor = torch.stack(cosine_values).to(torch.float64)
    return {
        "mu": _scalar_stats(mu_tensor),
        "cosine": _scalar_stats(cosine_tensor),
        "costs": {
            "flops": synthesizer.cost_flops(),
            "params": synthesizer.cost_params(),
            "comm_bytes": synthesizer.cost_comm_bytes(),
        },
    }


def _threshold_warnings(
    layer_idx: int,
    synthesizer: Synthesizer,
    metrics: dict[str, Any],
) -> list[str]:
    warnings = []
    mu_mean = metrics["mu"]["mean"]
    if synthesizer.name == "S0_exact" and mu_mean >= 1e-5:
        warnings.append(f"Layer {layer_idx} S0 Exact mu={mu_mean:.6g} violates mu < 1e-5")
    if synthesizer.name.startswith("S5_anchor_correction") and mu_mean > 0.06:
        warnings.append(f"Layer {layer_idx} S5 Anchor+Correction mu={mu_mean:.6g} exceeds 0.06")
    return warnings


def _resolve_split_sizes(cfg: StaticReconstructionConfig, num_vectors: int) -> tuple[int, int]:
    if cfg.quick:
        requested_calibration = min(32, max(1, num_vectors // 2))
        requested_eval = min(32, max(1, num_vectors - requested_calibration))
    else:
        requested_calibration = cfg.calibration_vectors
        requested_eval = cfg.eval_vectors
        if requested_eval is None:
            requested_eval = num_vectors - requested_calibration

    if requested_calibration <= 0 or requested_eval <= 0:
        raise ValueError("calibration_vectors and eval_vectors must be positive.")

    if requested_calibration + requested_eval > num_vectors:
        if not cfg.allow_small_data:
            raise ValueError(
                f"Need {requested_calibration + requested_eval} vectors but only {num_vectors} are available. "
                "Use --allow_small_data for smoke tests."
            )
        requested_calibration = min(requested_calibration, max(1, num_vectors // 2))
        requested_eval = min(requested_eval, num_vectors - requested_calibration)
        if requested_eval <= 0:
            raise ValueError("Not enough vectors to build a non-empty eval split.")

    return int(requested_calibration), int(requested_eval)


def _scalar_stats(values: torch.Tensor) -> dict[str, float | int]:
    mean = values.mean()
    var = values.var(unbiased=False)
    return {
        "mean": float(mean.item()),
        "std": float(var.sqrt().item()),
        "var": float(var.item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "count": int(values.numel()),
    }


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _parse_layers(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_ranks(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> StaticReconstructionConfig:
    parser = argparse.ArgumentParser(description="Finding2 Part A static reconstruction.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_indices", default="8,16,24")
    parser.add_argument("--calibration_vectors", type=int, default=2000)
    parser.add_argument("--eval_vectors", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow_small_data", action="store_true")
    parser.add_argument("--fail_on_threshold", action="store_true")
    parser.add_argument("--low_rank_ranks", default=None)
    parser.add_argument("--s5_rank", type=int, default=64)
    parser.add_argument("--s5_top_k", type=int, default=64)
    parser.add_argument("--distilled_feature_dim", type=int, default=128)
    parser.add_argument("--neighbor_k", type=int, default=1)
    args = parser.parse_args()
    return StaticReconstructionConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        layer_indices=_parse_layers(args.layer_indices),
        calibration_vectors=args.calibration_vectors,
        eval_vectors=args.eval_vectors,
        seed=args.seed,
        quick=args.quick,
        allow_small_data=args.allow_small_data,
        fail_on_threshold=args.fail_on_threshold,
        low_rank_ranks=_parse_ranks(args.low_rank_ranks),
        s5_rank=args.s5_rank,
        s5_top_k=args.s5_top_k,
        distilled_feature_dim=args.distilled_feature_dim,
        neighbor_k=args.neighbor_k,
    )


def main() -> None:
    run_static_reconstruction(parse_args())


if __name__ == "__main__":
    main()
