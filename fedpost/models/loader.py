from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedpost.models.peft_utils import (
    apply_lora,
    count_parameters,
    export_peft_state,
    get_trainable_keys,
    load_peft_state,
    save_adapter_pretrained,
    save_adapter_state,
    save_merged_pretrained,
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
        kwargs = {
            "trust_remote_code": self.cfg.model.trust_remote_code,
            "torch_dtype": dtype,
        }
        if self.cfg.model.use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.model_name_or_path,
            **kwargs,
        )
        if self.cfg.model.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "config"):
                model.config.use_cache = False
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
            state_type=state_type,
            trainable_keys=trainable_keys,
            aggregatable_keys=trainable_keys,
            frozen_keys=frozen_keys,
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

    def export_round_artifacts(
        self,
        model_bundle: ModelBundle,
        round_dir: str,
        save_adapter: bool = True,
        merge_model: bool = False,
    ) -> dict[str, str]:
        os.makedirs(round_dir, exist_ok=True)
        model = model_bundle.model
        tokenizer = model_bundle.tokenizer

        artifacts = {}

        if self.cfg.peft.method == "lora":
            adapter_state_path = os.path.join(round_dir, "adapter_state.pt")
            adapter_dir = os.path.join(round_dir, "adapter_model")
            merged_dir = os.path.join(round_dir, "merged_model")

            if save_adapter or merge_model:
                save_adapter_state(
                    model,
                    path=adapter_state_path,
                    adapter_name=self.cfg.peft.adapter_name,
                )
                save_adapter_pretrained(
                    model,
                    tokenizer=tokenizer,
                    output_dir=adapter_dir,
                )
                artifacts["adapter_state_path"] = adapter_state_path
                artifacts["adapter_dir"] = adapter_dir

            if merge_model:
                save_merged_pretrained(
                    base_model_name_or_path=self.cfg.model.model_name_or_path,
                    adapter_dir=adapter_dir,
                    output_dir=merged_dir,
                    torch_dtype=self._parse_dtype(self.cfg.model.torch_dtype),
                    trust_remote_code=self.cfg.model.trust_remote_code,
                    tokenizer=tokenizer,
                )
                artifacts["merged_model_dir"] = merged_dir
            return artifacts

        # full fine-tuning fallback
        if not merge_model:
            return artifacts
        merged_dir = os.path.join(round_dir, "merged_model")
        model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        artifacts["merged_model_dir"] = merged_dir
        return artifacts

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
