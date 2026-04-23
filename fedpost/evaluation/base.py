from __future__ import annotations

import os
import gc

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
        device_name = self.cfg.eval.eval_device
        if device_name is None:
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model.to(device)
        model.eval()
        return model, tokenizer, merged_dir

    def require_merged_model_dir(self, model_artifacts: dict) -> str:
        merged_dir = model_artifacts.get("merged_model_dir")
        if not merged_dir:
            raise ValueError("merged_model_dir is required for this evaluation path.")
        return merged_dir

    def release_round_model(self, model) -> None:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def record_runner_result(
        self,
        name: str,
        result: dict,
        metrics: dict,
        artifacts: dict,
        output_dir: str,
        returncode_key: str = "returncode",
    ) -> bool:
        returncode = int(result.get(returncode_key, 1))
        metrics[f"{name}_returncode"] = float(returncode)
        failed = returncode != 0
        metrics[f"{name}/failed"] = float(failed)

        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")
        if failed:
            os.makedirs(output_dir, exist_ok=True)
            stdout_path = os.path.join(output_dir, f"{name}_stdout.txt")
            stderr_path = os.path.join(output_dir, f"{name}_stderr.txt")
            with open(stdout_path, "w", encoding="utf-8") as f:
                f.write(stdout)
            with open(stderr_path, "w", encoding="utf-8") as f:
                f.write(stderr)
            artifacts[f"{name}_stdout_path"] = stdout_path
            artifacts[f"{name}_stderr_path"] = stderr_path
            if self.cfg.eval.fail_on_eval_error:
                raise RuntimeError(f"{name} failed with returncode={returncode}. stderr saved to {stderr_path}")

        return not failed

    def evaluate(self, model, round_idx: int, model_artifacts: dict | None = None):
        raise NotImplementedError
