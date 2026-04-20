from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTSample:
    prompt: str
    response: str
    metadata: dict | None = None


@dataclass
class DPOSample:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict | None = None


def _first_nonempty(record: dict, candidate_keys: list[str]) -> str | None:
    for key in candidate_keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_sft_sample(record: dict, cfg) -> SFTSample | None:
    prompt = _first_nonempty(record, [cfg.data.prompt_field, "prompt", "instruction", "input"])
    response = _first_nonempty(record, [cfg.data.response_field, "response", "output", "answer"])

    if prompt is None or response is None:
        return None

    metadata = {k: v for k, v in record.items() if k not in {
        cfg.data.prompt_field, cfg.data.response_field,
        "prompt", "instruction", "input",
        "response", "output", "answer",
    }}
    return SFTSample(prompt=prompt, response=response, metadata=metadata)


def build_dpo_sample(record: dict, cfg) -> DPOSample | None:
    prompt = _first_nonempty(record, [cfg.data.prompt_field, "prompt", "instruction", "input"])
    chosen = _first_nonempty(record, [cfg.data.chosen_field, "chosen"])
    rejected = _first_nonempty(record, [cfg.data.rejected_field, "rejected"])

    if prompt is None or chosen is None or rejected is None:
        return None

    metadata = {k: v for k, v in record.items() if k not in {
        cfg.data.prompt_field, cfg.data.chosen_field, cfg.data.rejected_field,
        "prompt", "instruction", "input",
        "chosen", "rejected",
    }}
    return DPOSample(prompt=prompt, chosen=chosen, rejected=rejected, metadata=metadata)