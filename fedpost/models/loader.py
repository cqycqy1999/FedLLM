from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedpost.models.peft_utils import (
    apply_lora,
    count_parameters,
    export_peft_state,
    get_trainable_keys,
    load_peft_state,
    validate_lora_targets,
)
from fedpost.models.reference_model import ReferenceModelManager
from fedpost.models.state_spec import ModelStateSpec


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    model_state_spec: ModelStateSpec
    reference_model: Any = None


class HFModelManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self) -> ModelBundle:
        tokenizer = self._build_tokenizer()
        model = self._build_model()
        model = self._apply_peft_if_needed(model)

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
        else:
            stats = count_parameters(model)
            print(f"Trainable params: {stats['trainable']} / {stats['total']}")

        reference_model = None
        if self.cfg.task == "dpo":
            reference_model = ReferenceModelManager(self.cfg).build(model)

        state_spec = self._build_state_spec(model)

        return ModelBundle(
            model=model,
            tokenizer=tokenizer,
            model_state_spec=state_spec,
            reference_model=reference_model,
        )

    def _build_tokenizer(self):
        name = self.cfg.model.tokenizer_name_or_path or self.cfg.model.model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _build_model(self):
        dtype = self._parse_dtype(self.cfg.model.torch_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.model_name_or_path,
            trust_remote_code=self.cfg.model.trust_remote_code,
            torch_dtype=dtype,
        )
        return model

    def _apply_peft_if_needed(self, model):
        if self.cfg.peft.method == "none":
            return model
        if self.cfg.peft.method == "lora":
            validate_lora_targets(model, self.cfg.peft.target_modules)
            return apply_lora(model, self.cfg.peft)
        raise ValueError(f"Unsupported peft method: {self.cfg.peft.method}")

    def _build_state_spec(self, model) -> ModelStateSpec:
        trainable_keys = get_trainable_keys(model)
        frozen_keys = [name for name, p in model.named_parameters() if not p.requires_grad]
        state_type = "adapter_only" if self.cfg.peft.method == "lora" else "full"

        return ModelStateSpec(
            state_type="full",
            trainable_keys=trainable_keys,
            aggregatable_keys=trainable_keys,
            frozen_keys=[],
        )

    def get_trainable_state(self, model) -> dict[str, Any]:
        if self.cfg.peft.method == "lora":
            return export_peft_state(
                model,
                adapter_name=self.cfg.peft.adapter_name,
            )

        state = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                state[name] = p.detach().cpu().clone()
        return state

    def load_trainable_state(self, model, state: dict[str, Any]) -> None:
        if self.cfg.peft.method == "lora":
            load_peft_state(
                model,
                state,
                adapter_name=self.cfg.peft.adapter_name,
            )
            return

        named_params = dict(model.named_parameters())
        for key, value in state.items():
            if key not in named_params:
                continue
            param = named_params[key]
            param.data.copy_(value.to(param.device, dtype=param.dtype))

    @staticmethod
    def _parse_dtype(dtype_name: str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if dtype_name not in mapping:
            raise ValueError(f"Unsupported torch dtype: {dtype_name}")
        return mapping[dtype_name]