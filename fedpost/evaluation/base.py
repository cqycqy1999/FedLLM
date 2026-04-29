from __future__ import annotations

import os
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from fedpost.evaluation.runners.lm_eval_runner import LMEvalRunner


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

    def resolve_lm_eval_model(self, model_artifacts: dict) -> tuple[str, str | None]:
        """
        Return (model_path, adapter_path). Prefer a merged model when available.
        If only a LoRA adapter was exported, use lm-eval's HF PEFT path:
        pretrained=<base_model>, peft=<adapter_dir>.
        """
        merged_dir = model_artifacts.get("merged_model_dir")
        if merged_dir:
            return merged_dir, None

        adapter_dir = model_artifacts.get("adapter_dir")
        model_backend = getattr(self.cfg.eval, "lm_eval_model_backend", "hf")
        allow_adapter = getattr(self.cfg.eval, "lm_eval_allow_adapter", True)
        if adapter_dir and allow_adapter and self.cfg.peft.method == "lora":
            if model_backend != "hf":
                raise ValueError("Adapter-only lm-eval currently requires lm_eval_model_backend='hf'.")
            return self.cfg.model.model_name_or_path, adapter_dir

        return self.require_merged_model_dir(model_artifacts), None

    def run_lm_eval(
        self,
        round_idx: int,
        model_artifacts: dict,
        metrics: dict,
        artifacts: dict,
    ) -> None:
        if not self.cfg.eval.run_lm_eval or not self.cfg.eval.lm_eval_tasks:
            return

        output_dir = os.path.join(self.eval_dir, f"round_{round_idx+1}", "lm_eval")
        model_path, adapter_path = self.resolve_lm_eval_model(model_artifacts)
        model_args = dict(getattr(self.cfg.eval, "lm_eval_model_args", {}) or {})
        if getattr(self.cfg.eval, "lm_eval_parallelize", False):
            model_args.setdefault("parallelize", True)

        runner = LMEvalRunner(output_dir)
        result = runner.run(
            model_path=model_path,
            adapter_path=adapter_path,
            tasks=self.cfg.eval.lm_eval_tasks,
            batch_size=self.cfg.eval.lm_eval_batch_size,
            device=self.cfg.eval.lm_eval_device or self.cfg.eval.eval_device,
            model_backend=getattr(self.cfg.eval, "lm_eval_model_backend", "hf"),
            model_args=model_args,
            dtype=getattr(self.cfg.eval, "lm_eval_dtype", None) or self.cfg.model.torch_dtype,
            trust_remote_code=self.cfg.model.trust_remote_code,
            num_fewshot=getattr(self.cfg.eval, "lm_eval_num_fewshot", None),
            limit=getattr(self.cfg.eval, "lm_eval_limit", None),
            log_samples=getattr(self.cfg.eval, "lm_eval_log_samples", False),
            include_path=getattr(self.cfg.eval, "lm_eval_include_path", None),
            use_cache=getattr(self.cfg.eval, "lm_eval_use_cache", None),
            apply_chat_template=getattr(self.cfg.eval, "lm_eval_apply_chat_template", False),
            fewshot_as_multiturn=getattr(self.cfg.eval, "lm_eval_fewshot_as_multiturn", None),
            gen_kwargs=getattr(self.cfg.eval, "lm_eval_gen_kwargs", None),
            seed=getattr(self.cfg.eval, "lm_eval_seed", None),
            predict_only=getattr(self.cfg.eval, "lm_eval_predict_only", False),
            timeout=getattr(self.cfg.eval, "lm_eval_timeout", None),
        )
        succeeded = self.record_runner_result("lm_eval", result, metrics, artifacts, output_dir)
        if succeeded:
            self._record_lm_eval_metrics(result, metrics)
        artifacts["lm_eval_result_path"] = result["result_path"]
        artifacts["lm_eval_model_path"] = model_path
        if adapter_path:
            artifacts["lm_eval_adapter_path"] = adapter_path

    @staticmethod
    def _record_lm_eval_metrics(result: dict, metrics: dict) -> None:
        parsed = result.get("parsed")
        if not parsed or "results" not in parsed:
            return
        for task_name, task_metrics in parsed["results"].items():
            for metric_name, metric_value in task_metrics.items():
                if isinstance(metric_value, (float, int)):
                    metrics[f"lm_eval/{task_name}/{metric_name}"] = float(metric_value)

    def evaluate(self, model, round_idx: int, model_artifacts: dict | None = None):
        raise NotImplementedError
