from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Asset:
    name: str
    repo_id: str
    kind: str
    split: str | None = None
    config_name: str | None = None
    gated: bool = False
    large: bool = False
    note: str = ""


PAPER_MODELS = [
    Asset("qwen3_0_6b", "Qwen/Qwen3-0.6B", "model", large=False),
    Asset("qwen3_1_7b", "Qwen/Qwen3-1.7B", "model", large=True, note="Memory-wall study"),
    Asset("qwen3_4b", "Qwen/Qwen3-4B", "model", large=True, note="Memory-wall study"),
    Asset("qwen3_8b", "Qwen/Qwen3-8B", "model", large=True, note="Memory-wall study"),
    Asset("qwen3_14b", "Qwen/Qwen3-14B", "model", large=True, note="Memory-wall study"),
    Asset("phi2_2_7b", "microsoft/phi-2", "model", large=True),
    Asset("llama3_8b", "meta-llama/Meta-Llama-3-8B", "model", gated=True, large=True),
    Asset("qwen2_5_32b", "Qwen/Qwen2.5-32B", "model", large=True),
    Asset("mixtral_8x7b", "mistralai/Mixtral-8x7B-v0.1", "model", large=True),
]


PAPER_DATASETS = [
    Asset("codealpaca_20k", "shi0222/CodeAlpaca-20k", "dataset", split="train"),
    Asset("finance_en", "ssbuild/alpaca_finance_en", "dataset", split="train"),
    Asset("medical_instruction", "ssbuild/alpaca_medical", "dataset", split="train"),
    Asset("qqp_memory_wall", "SetFit/qqp", "dataset", split="train"),
    Asset("humaneval_eval", "openai/openai_humaneval", "dataset", split="test"),
    Asset("finqa_eval", "ChanceFocus/flare-finqa", "dataset", split="test"),
    Asset("medqa_usmle_eval", "bigbio/med_qa", "dataset", split="test", config_name="med_qa_en_4options_source"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models and datasets for the benchmark experiments.")
    parser.add_argument("--output-dir", default="assets", help="Root directory for downloaded snapshots.")
    parser.add_argument("--asset", choices=["all", "models", "datasets"], default="datasets")
    parser.add_argument("--only", nargs="*", default=None, help="Asset names to download.")
    parser.add_argument("--include-large", action="store_true", help="Allow multi-GB model downloads.")
    parser.add_argument("--include-gated", action="store_true", help="Attempt gated model downloads with HF_TOKEN.")
    parser.add_argument("--save-datasets-to-disk", action="store_true", help="Materialize datasets with datasets.save_to_disk().")
    parser.add_argument("--export-jsonl", action="store_true", help="Write datasets to assets/jsonl/{name}.jsonl.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per dataset before saving/exporting.")
    parser.add_argument("--manifest-out", default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    assets = select_assets(args.asset, args.only)
    manifest = []
    for asset in assets:
        if asset.large and not args.include_large:
            manifest.append(record(asset, "skipped_large", None))
            continue
        if asset.gated and not args.include_gated:
            manifest.append(record(asset, "skipped_gated", None))
            continue

        try:
            path = download_asset(
                asset,
                args.output_dir,
                args.save_datasets_to_disk,
                export_jsonl=args.export_jsonl,
                max_samples=args.max_samples,
            )
            manifest.append(record(asset, "downloaded", path))
            print(f"[downloaded] {asset.name}: {path}")
        except Exception as exc:
            manifest.append(record(asset, "failed", None, str(exc)))
            print(f"[failed] {asset.name}: {exc}")

    manifest_out = args.manifest_out or os.path.join(args.output_dir, "paper_assets_manifest.json")
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"manifest: {manifest_out}")


def select_assets(asset_kind: str, only: list[str] | None) -> list[Asset]:
    if asset_kind == "models":
        assets = PAPER_MODELS
    elif asset_kind == "datasets":
        assets = PAPER_DATASETS
    else:
        assets = [*PAPER_MODELS, *PAPER_DATASETS]

    if only:
        wanted = set(only)
        assets = [asset for asset in assets if asset.name in wanted]
        missing = wanted - {asset.name for asset in assets}
        if missing:
            raise ValueError(f"Unknown asset names: {sorted(missing)}")
    return assets


def download_asset(
    asset: Asset,
    output_dir: str,
    save_datasets_to_disk: bool,
    export_jsonl: bool = False,
    max_samples: int | None = None,
) -> str:
    if asset.kind == "model":
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=asset.repo_id,
            local_dir=os.path.join(output_dir, "models", asset.name),
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )

    from datasets import load_dataset

    dataset_kwargs = {}
    if asset.config_name:
        dataset_kwargs["name"] = asset.config_name
    dataset = load_dataset(asset.repo_id, split=asset.split, **dataset_kwargs)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    jsonl_path = None
    if export_jsonl:
        jsonl_path = os.path.join(output_dir, "jsonl", f"{asset.name}.jsonl")
        export_dataset_jsonl(dataset, jsonl_path)

    if save_datasets_to_disk:
        path = os.path.join(output_dir, "datasets", asset.name)
        dataset.save_to_disk(path)
        return jsonl_path or path
    if jsonl_path:
        return jsonl_path
    return f"{asset.repo_id}:{asset.split}"


def export_dataset_jsonl(dataset, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in dataset:
            record = normalize_instruction_record(dict(row))
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def normalize_instruction_record(record: dict) -> dict:
    if "prompt" not in record and "instruction" in record:
        instruction = str(record.get("instruction", "")).strip()
        extra_input = str(record.get("input", "")).strip()
        record["prompt"] = f"{instruction}\n{extra_input}".strip()
    if "response" not in record and "output" in record:
        record["response"] = record.get("output")
    return record


def record(asset: Asset, status: str, path: str | None, error: str | None = None) -> dict:
    payload = asdict(asset)
    payload.update({"status": status, "path": path})
    if error:
        payload["error"] = error
    return payload


if __name__ == "__main__":
    main()
