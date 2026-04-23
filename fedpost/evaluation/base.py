from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class Evaluator:
    def __init__(self, cfg, tokenizer, logger=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.logger = logger

    def _parse_dtype(self, dtype_name: str):
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        return mapping.get(dtype_name, torch.float32)

    def load_round_model(self, model_artifacts: dict):
        """
        优先从当前 round 导出的 merged_model_dir 加载。
        """
        merged_dir = model_artifacts.get("merged_model_dir")
        if not merged_dir:
            raise ValueError("merged_model_dir is required for round-based evaluation.")

        tokenizer = AutoTokenizer.from_pretrained(merged_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            merged_dir,
            torch_dtype=self._parse_dtype(self.cfg.model.torch_dtype),
            trust_remote_code=self.cfg.model.trust_remote_code,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, tokenizer, merged_dir

    def evaluate(self, model, round_idx: int, model_artifacts: dict | None = None):
        raise NotImplementedError