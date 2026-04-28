from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any

import yaml

from fedpost.analysis.hidden_states.collect import (
    HiddenStateCollectionConfig,
    collect_hidden_states,
)
from fedpost.analysis.hidden_states.metrics import (
    ResidualStreamMetricsConfig,
    compute_residual_stream_metrics,
)


def run_from_config(
    config_path: str,
    dry_run: bool = False,
    collect_only: bool = False,
    metrics_only: bool = False,
    experiment_name: str | None = None,
) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    experiments = _expand_experiments(raw)
    if experiment_name is not None:
        experiments = [exp for exp in experiments if exp["name"] == experiment_name]
        if not experiments:
            raise ValueError(f"No experiment named {experiment_name} found in {config_path}")

    for exp in experiments:
        merged = _deep_merge(defaults, exp)
        name = merged["name"]
        output_root = merged.get("output_root", "outputs/residual_stream")
        exp_dir = os.path.join(output_root, name)
        hidden_path = merged.get("hidden_output_path", os.path.join(exp_dir, "hidden_states.pt"))
        metrics_path = merged.get("metrics_output_path", os.path.join(exp_dir, "metrics.pt"))

        collect_cfg = HiddenStateCollectionConfig(
            model_name_or_path=merged["model_name_or_path"],
            dataset_kind=merged["dataset_kind"],
            dataset_name=merged["dataset_name"],
            dataset_split=merged.get("dataset_split", "train"),
            dataset_config=merged.get("dataset_config"),
            text_field=merged.get("text_field"),
            output_path=hidden_path,
            num_samples=int(merged.get("num_samples", 512)),
            max_length=int(merged.get("max_length", 2048)),
            seed=int(merged.get("seed", 42)),
            dtype=merged.get("dtype", "float16"),
            trust_remote_code=bool(merged.get("trust_remote_code", False)),
            use_flash_attn=bool(merged.get("use_flash_attn", False)),
            device=merged.get("device"),
            device_map=merged.get("device_map"),
            capture_mode=merged.get("capture_mode", "auto"),
            downsample_hidden=merged.get("downsample_hidden"),
            projection_seed=int(merged.get("projection_seed", 1234)),
            dry_run=bool(dry_run or merged.get("dry_run", False)),
        )

        metrics_cfg = ResidualStreamMetricsConfig(
            input_path=hidden_path,
            output_path=metrics_path,
            random_pairs=int(merged.get("random_pairs", 2000)),
            seed=int(merged.get("seed", 42)),
            batch_size=int(merged.get("metrics_batch_size", 8)),
            random_pair_batch=int(merged.get("random_pair_batch", 64)),
            eps=float(merged.get("eps", 1e-12)),
        )

        _write_effective_config(exp_dir, collect_cfg, metrics_cfg)
        print(f"running residual-stream experiment: {name}")
        if not metrics_only:
            collect_hidden_states(collect_cfg)
        if not collect_only:
            compute_residual_stream_metrics(metrics_cfg)


def _expand_experiments(raw: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = raw.get("experiments")
    if explicit:
        return explicit

    models = raw.get("models", [])
    datasets = raw.get("datasets", [])
    experiments = []
    for model in models:
        for dataset in datasets:
            name = f"{model['name']}__{dataset['name']}"
            experiment = {**model, **dataset}
            experiment["name"] = name
            experiment["model_alias"] = model["name"]
            experiment["probe_dataset_alias"] = dataset["name"]
            experiments.append(experiment)
    return experiments


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _write_effective_config(
    exp_dir: str,
    collect_cfg: HiddenStateCollectionConfig,
    metrics_cfg: ResidualStreamMetricsConfig,
) -> None:
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "effective_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "collect": asdict(collect_cfg),
                "metrics": asdict(metrics_cfg),
            },
            f,
            sort_keys=False,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run residual-stream hidden-state experiments from YAML.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment_name", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--metrics_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_from_config(
        config_path=args.config,
        dry_run=args.dry_run,
        collect_only=args.collect_only,
        metrics_only=args.metrics_only,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
