from __future__ import annotations

from typing import Any

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


def apply_lora(model: Any, peft_cfg) -> Any:
    lora_config = LoraConfig(
        r=peft_cfg.r,
        lora_alpha=peft_cfg.alpha,
        lora_dropout=peft_cfg.dropout,
        target_modules=peft_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config, adapter_name=peft_cfg.adapter_name)
    return model


def export_peft_state(model: Any, adapter_name: str = "default") -> dict:
    return get_peft_model_state_dict(
        model,
        adapter_name=adapter_name,
    )


def load_peft_state(
    model: Any,
    peft_state_dict: dict,
    adapter_name: str = "default",
) -> None:
    set_peft_model_state_dict(
        model,
        peft_state_dict,
        adapter_name=adapter_name,
    )


def count_parameters(model) -> dict[str, int]:
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return {
        "total": total,
        "trainable": trainable,
    }


def get_trainable_keys(model) -> list[str]:
    return [name for name, p in model.named_parameters() if p.requires_grad]


def validate_lora_targets(model, target_modules: list[str]) -> None:
    module_names = [name for name, _ in model.named_modules()]
    for target in target_modules:
        found = any(target in name for name in module_names)
        if not found:
            raise ValueError(f"LoRA target module '{target}' not found in model.")