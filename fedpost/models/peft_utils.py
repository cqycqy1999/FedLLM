from __future__ import annotations

from typing import Any

import os
import torch
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM


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
    state_dict = get_peft_model_state_dict(
        model,
        adapter_name=adapter_name,
    )
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in state_dict.items()
    }


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


def save_adapter_state(model, path: str, adapter_name: str = "default") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = export_peft_state(model, adapter_name=adapter_name)
    torch.save(state, path)
    return path


def save_adapter_pretrained(model, tokenizer, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)
    return output_dir


def save_merged_pretrained(
    base_model_name_or_path: str,
    adapter_dir: str,
    output_dir: str,
    torch_dtype,
    trust_remote_code: bool = False,
    tokenizer=None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(output_dir)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    return output_dir

    
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
