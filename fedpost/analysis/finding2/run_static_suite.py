from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import torch
import yaml

from fedpost.analysis.finding2.static_reconstruction import (
    StaticReconstructionConfig,
    run_static_reconstruction,
)


def run_from_config(
    config_path: str,
    experiment_name: str | None = None,
    quick: bool = False,
    allow_small_data: bool = False,
    fail_on_threshold: bool = False,
) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    experiments = raw.get("experiments") or []
    if experiment_name is not None:
        experiments = [exp for exp in experiments if exp.get("name") == experiment_name]
        if not experiments:
            raise ValueError(f"No experiment named {experiment_name} found in {config_path}")
    if not experiments:
        raise ValueError(f"No experiments found in {config_path}")

    output_root = raw.get("output_root", "outputs/finding2/static_reconstruction")
    required_samples = int(raw.get("required_samples", 1000))
    results = []
    summary_rows = []

    for exp in experiments:
        name = exp["name"]
        input_path = exp["input_path"]
        shape = _load_hidden_shape(input_path)
        num_samples, num_probes, num_layers, hidden_dim = shape

        effective_allow_small = bool(allow_small_data or raw.get("allow_small_data", False))
        if quick:
            effective_allow_small = True
        if num_samples < required_samples and not effective_allow_small:
            raise ValueError(
                f"{name} has {num_samples} hidden-state samples, but Part A requires "
                f"{required_samples}. Re-run residual-stream collection with more samples "
                f"or pass --allow_small_data for an explicitly marked small-data run."
            )
        if num_samples < required_samples:
            print(
                f"WARNING: {name} has {num_samples} samples < required_samples={required_samples}; "
                f"running as small-data Part A."
            )

        layer_indices = _resolve_layer_indices(
            exp.get("layer_indices", raw.get("layer_indices", "auto_quarters")),
            num_layers=num_layers,
        )
        exp_output_dir = exp.get("output_dir", os.path.join(output_root, name))
        cfg = StaticReconstructionConfig(
            input_path=input_path,
            output_dir=exp_output_dir,
            layer_indices=layer_indices,
            calibration_vectors=int(exp.get("calibration_vectors", raw.get("calibration_vectors", 2000))),
            eval_vectors=exp.get("eval_vectors", raw.get("eval_vectors")),
            seed=int(exp.get("seed", raw.get("seed", 42))),
            quick=bool(quick or exp.get("quick", raw.get("quick", False))),
            allow_small_data=effective_allow_small,
            fail_on_threshold=bool(fail_on_threshold or exp.get("fail_on_threshold", raw.get("fail_on_threshold", False))),
            low_rank_ranks=exp.get("low_rank_ranks", raw.get("low_rank_ranks")),
            s5_rank=int(exp.get("s5_rank", raw.get("s5_rank", 64))),
            s5_top_k=int(exp.get("s5_top_k", raw.get("s5_top_k", 64))),
            distilled_feature_dim=int(exp.get("distilled_feature_dim", raw.get("distilled_feature_dim", 128))),
            neighbor_k=int(exp.get("neighbor_k", raw.get("neighbor_k", 1))),
        )

        print(f"running Finding2 Part A static reconstruction: {name}, layers={layer_indices}")
        result = run_static_reconstruction(cfg)
        exp_summary = {
            "name": name,
            "input_path": input_path,
            "output_dir": exp_output_dir,
            "hidden_shape": list(shape),
            "required_samples": required_samples,
            "small_data": num_samples < required_samples,
            "layer_indices": layer_indices,
            "warnings": result.get("warnings", []),
        }
        results.append(exp_summary)
        summary_rows.extend(_summary_rows(name, exp, exp_summary, result, num_layers))

    suite_summary = {
        "protocol": "finding2_part_a_static_reconstruction_suite",
        "config_path": config_path,
        "required_samples": required_samples,
        "quick": quick,
        "allow_small_data": allow_small_data,
        "experiments": results,
    }
    os.makedirs(output_root, exist_ok=True)
    summary_json = os.path.join(output_root, "suite_summary.json")
    summary_csv = os.path.join(output_root, "suite_summary.csv")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)
    _write_summary_csv(summary_csv, summary_rows)
    print(f"saved Part A suite summary: {summary_json}")
    print(f"saved Part A suite table: {summary_csv}")
    return suite_summary


def _load_hidden_shape(input_path: str) -> tuple[int, int, int, int]:
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)
    payload = torch.load(input_path, map_location="cpu")
    if "hidden_states" not in payload:
        raise KeyError(f"{input_path} does not contain 'hidden_states'.")
    hidden = payload["hidden_states"]
    if hidden.ndim != 4:
        raise ValueError(f"Expected hidden_states shape [N, P, L, D], got {tuple(hidden.shape)}")
    return tuple(int(dim) for dim in hidden.shape)


def _resolve_layer_indices(layer_spec: Any, num_layers: int) -> list[int]:
    if layer_spec is None or layer_spec == "auto_quarters":
        num_blocks = num_layers - 1
        layers = [num_blocks // 4, num_blocks // 2, (3 * num_blocks) // 4]
    elif isinstance(layer_spec, str):
        layers = [int(item.strip()) for item in layer_spec.split(",") if item.strip()]
    else:
        layers = [int(item) for item in layer_spec]

    resolved = []
    for layer_idx in layers:
        if layer_idx <= 0 or layer_idx >= num_layers:
            raise ValueError(f"Layer index {layer_idx} is invalid for {num_layers} hidden-state layers.")
        if layer_idx not in resolved:
            resolved.append(layer_idx)
    return resolved


def _summary_rows(
    name: str,
    exp: dict[str, Any],
    exp_summary: dict[str, Any],
    result: dict[str, Any],
    num_layers: int,
) -> list[dict[str, Any]]:
    rows = []
    num_blocks = num_layers - 1
    model_alias, dataset_alias = _split_experiment_name(name)
    for layer_key, layer_result in result.get("layers", {}).items():
        layer_idx = int(layer_key)
        for synthesizer_name, metrics in layer_result.get("synthesizers", {}).items():
            costs = metrics.get("costs", {})
            rows.append(
                {
                    "experiment": name,
                    "model": exp.get("model_alias", model_alias),
                    "dataset": exp.get("dataset_alias", dataset_alias),
                    "input_path": exp_summary["input_path"],
                    "hidden_shape": "x".join(str(dim) for dim in exp_summary["hidden_shape"]),
                    "small_data": exp_summary["small_data"],
                    "layer_idx": layer_idx,
                    "normalized_depth": layer_idx / max(1, num_blocks),
                    "synthesizer": synthesizer_name,
                    "mu_mean": metrics["mu"]["mean"],
                    "mu_std": metrics["mu"]["std"],
                    "mu_count": metrics["mu"]["count"],
                    "cosine_mean": metrics["cosine"]["mean"],
                    "cosine_std": metrics["cosine"]["std"],
                    "flops": costs.get("flops"),
                    "params": costs.get("params"),
                    "comm_bytes": costs.get("comm_bytes"),
                }
            )
    return rows


def _split_experiment_name(name: str) -> tuple[str, str]:
    if "__" not in name:
        return name, ""
    model, dataset = name.split("__", 1)
    return model, dataset


def _write_summary_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("No Part A summary rows were produced.")
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Finding2 Part A static reconstruction suite from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow_small_data", action="store_true")
    parser.add_argument("--fail_on_threshold", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_from_config(
        config_path=args.config,
        experiment_name=args.experiment_name,
        quick=args.quick,
        allow_small_data=args.allow_small_data,
        fail_on_threshold=args.fail_on_threshold,
    )


if __name__ == "__main__":
    main()
