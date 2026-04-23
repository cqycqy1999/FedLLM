from __future__ import annotations

import json
import os

import torch
from huggingface_hub import hf_hub_download

from fedpost.evaluation.base import Evaluator
from fedpost.evaluation.runners.alpaca_eval_runner import AlpacaEvalRunner
from fedpost.evaluation.runners.lm_eval_runner import LMEvalRunner
from fedpost.federation.message import EvalResult
from fedpost.utils.registry import Registry


@Registry.register("evaluator", "sft")
class SFTGenerationEvaluator(Evaluator):
    def __init__(self, cfg, tokenizer, logger=None):
        super().__init__(cfg, tokenizer, logger)
        self.eval_dir = os.path.join(cfg.output_dir, "eval")

    def _load_alpaca_eval_set(self) -> list[dict]:
        eval_path = hf_hub_download(
            repo_id="tatsu-lab/alpaca_eval",
            filename="alpaca_eval.json",
            repo_type="dataset",
        )
        with open(eval_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _generate_alpaca_outputs(self, model, tokenizer, max_new_tokens: int):
        eval_set = self._load_alpaca_eval_set()
        device = next(model.parameters()).device
        # model.eval()

        outputs = []
        for ex in eval_set:
            instruction = ex["instruction"]
            prompt = f"User: {instruction}\nAssistant:"
            batch = self.tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
            answer = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
            outputs.append({
                "instruction": instruction,
                "output": answer,
                "generator": self.cfg.eval.alpaca_eval_model_name,
            })
        return outputs

    def evaluate(self, model, round_idx: int, model_artifacts: dict | None = None):
        model_artifacts = model_artifacts or {}
        metrics = {}
        artifacts = dict(model_artifacts)

        round_model, round_tokenizer, merged_dir = self.load_round_model(model_artifacts)

        if self.cfg.eval.run_alpaca_eval:
            outputs = self._generate_alpaca_outputs(
                round_model,
                round_tokenizer,
                max_new_tokens=self.cfg.eval.eval_generation_max_new_tokens,
            )
            runner = AlpacaEvalRunner(os.path.join(self.eval_dir, f"round_{round_idx+1}", "alpaca_eval"))
            result = runner.run(
                outputs,
                annotators_config=self.cfg.eval.alpaca_eval_annotators_config,
            )
            metrics["alpaca_eval_returncode"] = float(result["returncode"])
            artifacts["alpaca_eval_result_dir"] = result["result_dir"]
            artifacts["alpaca_eval_outputs_path"] = result["outputs_path"]

        if self.cfg.eval.run_lm_eval and self.cfg.eval.lm_eval_tasks:
            runner = LMEvalRunner(os.path.join(self.eval_dir, f"round_{round_idx+1}", "lm_eval"))
            result = runner.run(
                model_path=merged_dir,
                tasks=self.cfg.eval.lm_eval_tasks,
                batch_size=self.cfg.eval.lm_eval_batch_size,
                device=self.cfg.eval.lm_eval_device,
            )
            metrics["lm_eval_returncode"] = float(result["returncode"])
            if result["parsed"] and "results" in result["parsed"]:
                for task_name, task_metrics in result["parsed"]["results"].items():
                    for metric_name, metric_value in task_metrics.items():
                        if isinstance(metric_value, (float, int)):
                            metrics[f"lm_eval/{task_name}/{metric_name}"] = float(metric_value)
            artifacts["lm_eval_result_path"] = result["result_path"]

        return EvalResult(
            round_idx=round_idx,
            split="val",
            metrics=metrics,
            artifacts=artifacts,
        )
