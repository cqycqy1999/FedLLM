from __future__ import annotations

import os
import datasets
import torch

from fedpost.evaluation.base import Evaluator
from fedpost.evaluation.runners.alpaca_eval_runner import AlpacaEvalRunner
from fedpost.evaluation.runners.lm_eval_runner import LMEvalRunner
from fedpost.evaluation.runners.mt_bench_runner import MTBenchRunner
from fedpost.federation.message import EvalResult
from fedpost.utils.registry import Registry


@Registry.register("evaluator", "dpo")
class DPOComboEvaluator(Evaluator):
    def __init__(self, cfg, tokenizer, logger=None):
        super().__init__(cfg, tokenizer, logger)
        self.eval_dir = os.path.join(cfg.output_dir, "eval")

    def _generate_alpaca_outputs(self, model, max_new_tokens: int = 128):
        eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
        device = next(model.parameters()).device
        model.eval()

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

        if self.cfg.eval.run_alpaca_eval:
            alpaca_outputs = self._generate_alpaca_outputs(model)
            runner = AlpacaEvalRunner(os.path.join(self.eval_dir, f"round_{round_idx}", "alpaca_eval"))
            result = runner.run(alpaca_outputs)
            metrics["alpaca_eval_returncode"] = float(result["returncode"])
            artifacts["alpaca_eval_result_dir"] = result["result_dir"]

        if self.cfg.eval.run_mt_bench:
            model_path = model_artifacts.get("merged_model_dir", self.cfg.model.model_name_or_path)
            runner = MTBenchRunner(os.path.join(self.eval_dir, f"round_{round_idx}", "mt_bench"))
            result = runner.run(
                model_path=model_path,
                model_id=self.cfg.eval.mt_bench_model_id,
            )
            metrics["mt_bench_gen_returncode"] = float(result["gen_returncode"])
            metrics["mt_bench_judge_returncode"] = float(result["judge_returncode"])
            artifacts["mt_bench_answer_file"] = result["answer_file"]

        if self.cfg.eval.run_lm_eval and self.cfg.eval.lm_eval_tasks:
            model_path = model_artifacts.get("merged_model_dir", self.cfg.model.model_name_or_path)
            runner = LMEvalRunner(os.path.join(self.eval_dir, f"round_{round_idx}", "lm_eval"))
            result = runner.run(
                model_path=model_path,
                tasks=self.cfg.eval.lm_eval_tasks,
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