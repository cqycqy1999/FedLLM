from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class ResidualStreamMetricsConfig:
    input_path: str
    output_path: str
    random_pairs: int = 2000
    seed: int = 42
    batch_size: int = 8
    random_pair_batch: int = 64
    eps: float = 1e-12


def compute_residual_stream_metrics(cfg: ResidualStreamMetricsConfig) -> dict[str, Any]:
    payload = torch.load(cfg.input_path, map_location="cpu")
    hidden = payload["hidden_states"]
    if hidden.ndim != 4:
        raise ValueError(f"Expected hidden_states with shape [N, P, L, D], got {tuple(hidden.shape)}")

    num_samples, num_probes, num_layers, hidden_dim = hidden.shape
    num_segments = num_layers - 1
    if num_segments <= 0:
        raise ValueError("Need at least two hidden-state layers to compute residual-stream metrics.")

    random_pairs = _sample_random_layer_pairs(
        num_layers=num_layers,
        num_pairs=cfg.random_pairs,
        seed=cfg.seed,
    )

    delta_sum = torch.zeros(num_segments, dtype=torch.float64)
    delta_sumsq = torch.zeros(num_segments, dtype=torch.float64)
    delta_count = 0

    cosine_sum = torch.zeros(num_layers, num_layers, dtype=torch.float64)
    cosine_sumsq = torch.zeros(num_layers, num_layers, dtype=torch.float64)
    cosine_count = 0

    rho_sum = 0.0
    rho_sumsq = 0.0
    rho_count = 0

    random_sum = 0.0
    random_sumsq = 0.0
    random_count = 0

    for start in range(0, num_samples, cfg.batch_size):
        chunk = hidden[start:start + cfg.batch_size].to(torch.float32)
        batch_size = chunk.shape[0]
        flat_count = batch_size * num_probes

        deltas = _local_delta(chunk, cfg.eps)
        delta_sum += deltas.sum(dim=(0, 1)).to(torch.float64)
        delta_sumsq += (deltas ** 2).sum(dim=(0, 1)).to(torch.float64)
        delta_count += flat_count

        cosine = _cosine_matrices(chunk, cfg.eps)
        cosine_sum += cosine.sum(dim=(0, 1)).to(torch.float64)
        cosine_sumsq += (cosine ** 2).sum(dim=(0, 1)).to(torch.float64)
        cosine_count += flat_count

        rho = _trajectory_rho(chunk, cfg.eps)
        rho_sum += float(rho.sum().item())
        rho_sumsq += float((rho ** 2).sum().item())
        rho_count += int(rho.numel())

        for pair_start in range(0, len(random_pairs), cfg.random_pair_batch):
            pair_batch = random_pairs[pair_start:pair_start + cfg.random_pair_batch]
            random_values = _random_delta(chunk, pair_batch, cfg.eps)
            random_sum += float(random_values.sum().item())
            random_sumsq += float((random_values ** 2).sum().item())
            random_count += int(random_values.numel())

    result = {
        "metadata": {
            "input_path": cfg.input_path,
            "shape": tuple(hidden.shape),
            "num_samples": num_samples,
            "num_probe_positions": num_probes,
            "num_layers": num_layers,
            "num_segments": num_segments,
            "hidden_dim": hidden_dim,
            "random_pairs": cfg.random_pairs,
            "seed": cfg.seed,
        },
        "delta": _mean_var(delta_sum, delta_sumsq, delta_count),
        "cosine": _mean_var(cosine_sum, cosine_sumsq, cosine_count),
        "rho": _scalar_mean_var(rho_sum, rho_sumsq, rho_count),
        "delta_random": _scalar_mean_var(random_sum, random_sumsq, random_count),
        "random_layer_pairs": random_pairs,
    }

    os.makedirs(os.path.dirname(cfg.output_path), exist_ok=True)
    torch.save(result, cfg.output_path)
    json_path = f"{os.path.splitext(cfg.output_path)[0]}.summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable_summary(result), f, ensure_ascii=False, indent=2)

    print(f"saved metrics: {cfg.output_path}")
    print(
        "delta mean range: "
        f"{float(result['delta']['mean'].min()):.6f} - {float(result['delta']['mean'].max()):.6f}"
    )
    print(f"rho mean: {result['rho']['mean']:.6f}")
    print(f"delta_random mean: {result['delta_random']['mean']:.6f}")
    return result


def _local_delta(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    prev = hidden[:, :, :-1, :]
    nxt = hidden[:, :, 1:, :]
    numerator = torch.linalg.vector_norm(nxt - prev, dim=-1)
    denominator = torch.linalg.vector_norm(prev, dim=-1).clamp_min(eps)
    return numerator / denominator


def _cosine_matrices(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = hidden / torch.linalg.vector_norm(hidden, dim=-1, keepdim=True).clamp_min(eps)
    return torch.einsum("bpld,bpmd->bplm", normalized, normalized)


def _trajectory_rho(hidden: torch.Tensor, eps: float) -> torch.Tensor:
    segment_lengths = torch.linalg.vector_norm(hidden[:, :, 1:, :] - hidden[:, :, :-1, :], dim=-1)
    path_length = segment_lengths.sum(dim=-1)
    endpoint = torch.linalg.vector_norm(hidden[:, :, -1, :] - hidden[:, :, 0, :], dim=-1).clamp_min(eps)
    return path_length / endpoint


def _random_delta(
    hidden: torch.Tensor,
    pairs: list[tuple[int, int]],
    eps: float,
) -> torch.Tensor:
    values = []
    for left, right in pairs:
        numerator = torch.linalg.vector_norm(hidden[:, :, right, :] - hidden[:, :, left, :], dim=-1)
        denominator = torch.linalg.vector_norm(hidden[:, :, left, :], dim=-1).clamp_min(eps)
        values.append(numerator / denominator)
    return torch.stack(values, dim=-1)


def _sample_random_layer_pairs(
    num_layers: int,
    num_pairs: int,
    seed: int,
) -> list[tuple[int, int]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    min_gap = math.ceil((num_layers - 1) / 2)
    candidates = [
        (i, j)
        for i in range(num_layers)
        for j in range(num_layers)
        if i != j and abs(i - j) >= min_gap
    ]
    if not candidates:
        raise ValueError("No valid random layer pairs can satisfy |i-j| >= L/2.")

    indices = torch.randint(len(candidates), (num_pairs,), generator=generator).tolist()
    return [candidates[idx] for idx in indices]


def _mean_var(total: torch.Tensor, total_sq: torch.Tensor, count: int) -> dict[str, torch.Tensor]:
    mean = total / max(1, count)
    var = total_sq / max(1, count) - mean ** 2
    return {
        "mean": mean.to(torch.float32),
        "var": var.clamp_min(0.0).to(torch.float32),
        "std": var.clamp_min(0.0).sqrt().to(torch.float32),
        "count": count,
    }


def _scalar_mean_var(total: float, total_sq: float, count: int) -> dict[str, float | int]:
    mean = total / max(1, count)
    var = total_sq / max(1, count) - mean ** 2
    var = max(0.0, var)
    return {
        "mean": mean,
        "var": var,
        "std": math.sqrt(var),
        "count": count,
    }


def _jsonable_summary(result: dict[str, Any]) -> dict[str, Any]:
    delta_mean = result["delta"]["mean"]
    cosine_mean = result["cosine"]["mean"]
    return {
        "metadata": result["metadata"],
        "delta": {
            "mean": delta_mean.tolist(),
            "std": result["delta"]["std"].tolist(),
            "count": result["delta"]["count"],
        },
        "rho": result["rho"],
        "delta_random": result["delta_random"],
        "cosine": {
            "mean_shape": list(cosine_mean.shape),
            "diag_mean": torch.diagonal(cosine_mean).tolist(),
            "count": result["cosine"]["count"],
        },
    }


def parse_args() -> ResidualStreamMetricsConfig:
    parser = argparse.ArgumentParser(description="Compute residual-stream metrics from collected hidden states.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--random_pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--random_pair_batch", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-12)
    return ResidualStreamMetricsConfig(**vars(parser.parse_args()))


def main() -> None:
    compute_residual_stream_metrics(parse_args())


if __name__ == "__main__":
    main()
