from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from datasets import load_dataset


@dataclass
class ProbeDatasetConfig:
    kind: str
    dataset_name: str
    split: str = "train"
    dataset_config: str | None = None
    text_field: str | None = None
    max_samples: int | None = None
    seed: int = 42


def load_probe_texts(cfg: ProbeDatasetConfig) -> list[dict[str, Any]]:
    if cfg.kind in {"alpaca_eval", "alpaca-eval"}:
        records = _load_alpaca_eval(cfg)
    else:
        records = _load_hf_records(cfg)

    rng = random.Random(cfg.seed)
    rng.shuffle(records)
    if cfg.max_samples is not None:
        records = records[: cfg.max_samples]
    return records


def _load_hf_records(cfg: ProbeDatasetConfig) -> list[dict[str, Any]]:
    if cfg.dataset_config:
        dataset = load_dataset(cfg.dataset_name, cfg.dataset_config, split=cfg.split)
    else:
        dataset = load_dataset(cfg.dataset_name, split=cfg.split)

    records = []
    for idx, rec in enumerate(dataset):
        text = _record_to_text(cfg.kind, rec, cfg.text_field)
        if text:
            records.append({
                "id": str(rec.get("id", idx)) if isinstance(rec, dict) else str(idx),
                "text": text,
                "source": cfg.dataset_name,
                "kind": cfg.kind,
            })
    return records


def _load_alpaca_eval(cfg: ProbeDatasetConfig) -> list[dict[str, Any]]:
    import json
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=cfg.dataset_name or "tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    records = []
    for idx, rec in enumerate(dataset):
        instruction = str(rec.get("instruction", "")).strip()
        if instruction:
            records.append({
                "id": str(rec.get("id", idx)),
                "text": instruction,
                "source": cfg.dataset_name or "tatsu-lab/alpaca_eval",
                "kind": cfg.kind,
            })
    return records


def _record_to_text(kind: str, rec: dict, text_field: str | None) -> str | None:
    if text_field:
        return _clean(rec.get(text_field))

    if kind == "alpaca":
        instruction = _clean(rec.get("instruction"))
        input_text = _clean(rec.get("input") or rec.get("context"))
        if not instruction:
            return None
        if input_text:
            return f"{instruction}\n\nInput:\n{input_text}"
        return instruction

    if kind == "gsm8k":
        return _clean(rec.get("question"))

    if kind == "mmlu":
        question = _clean(rec.get("question"))
        choices = rec.get("choices")
        if not question:
            return None
        if isinstance(choices, list) and choices:
            labels = ["A", "B", "C", "D", "E", "F"]
            rendered = "\n".join(
                f"{labels[idx]}. {choice}"
                for idx, choice in enumerate(choices)
                if idx < len(labels)
            )
            return f"{question}\n{rendered}"
        return question

    return _clean(rec.get("text") or rec.get("prompt") or rec.get("instruction"))


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
