from __future__ import annotations

import re
from typing import Any

from fedpost.bpfedpeft.planner import BlockSpec


_LAYER_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)h\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)blocks\.(\d+)(?:\.|$)"),
    re.compile(r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)"),
)


def extract_layer_index(name: str) -> int | None:
    for pattern in _LAYER_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def is_lora_parameter(name: str) -> bool:
    return "lora_" in name or ".modules_to_save." in name


def state_for_block(
    state: dict[str, Any],
    block: BlockSpec,
    include_unindexed: bool = False,
) -> dict[str, Any]:
    return {
        key: value
        for key, value in state.items()
        if key_belongs_to_block(key, block, include_unindexed=include_unindexed)
    }


def key_belongs_to_block(
    key: str,
    block: BlockSpec,
    include_unindexed: bool = False,
) -> bool:
    layer_idx = extract_layer_index(key)
    if layer_idx is None:
        return include_unindexed
    return block.start <= layer_idx <= block.end


def set_active_peft_block(
    model,
    block: BlockSpec,
    include_unindexed: bool = False,
) -> dict[str, int]:
    """Enable gradients only for PEFT parameters in the active block."""

    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        if not is_lora_parameter(name):
            param.requires_grad = False
            frozen += 1
            continue

        active = key_belongs_to_block(name, block, include_unindexed=include_unindexed)
        param.requires_grad = active
        if active:
            trainable += 1
        else:
            frozen += 1

    return {"trainable_params": trainable, "frozen_params": frozen}


def state_delta_norm(before: dict[str, Any], after: dict[str, Any]) -> tuple[float, float]:
    """Return (absolute_delta_norm, relative_delta_norm)."""

    squared_delta = 0.0
    squared_ref = 0.0
    for key, after_value in after.items():
        before_value = before.get(key)
        if before_value is None or not hasattr(after_value, "detach"):
            continue
        delta = after_value.detach().float().cpu() - before_value.detach().float().cpu()
        squared_delta += float((delta * delta).sum().item())
        squared_ref += float((before_value.detach().float().cpu() ** 2).sum().item())

    abs_norm = squared_delta ** 0.5
    rel_norm = abs_norm / ((squared_ref ** 0.5) + 1e-12)
    return abs_norm, rel_norm
