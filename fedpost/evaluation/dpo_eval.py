from __future__ import annotations

import json
import os
import torch
from huggingface_hub import hf_hub_download

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

        outputs = []
        for ex in eval_set:
            instruction = ex["instruction"]
            prompt = f"User: {instruction}\nAssistant:"
            batch = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                gen = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            text = tokenizer.decode(gen[0], skip_special_tokens=True)
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

        if not any([
            self.cfg.eval.run_alpaca_eval,
            self.cfg.eval.run_mt_bench,
            self.cfg.eval.run_lm_eval and self.cfg.eval.lm_eval_tasks,
        ]):
            return EvalResult(
                round_idx=round_idx,
                split="val",
                metrics=metrics,
                artifacts=artifacts,
            )

        merged_dir = self.require_merged_model_dir(model_artifacts)

        if self.cfg.eval.run_alpaca_eval:
            round_model, round_tokenizer, _ = self.load_round_model(model_artifacts)
            try:
                outputs = self._generate_alpaca_outputs(
                    round_model,
                    round_tokenizer,
                    max_new_tokens=self.cfg.eval.eval_generation_max_new_tokens,
                )
                output_dir = os.path.join(self.eval_dir, f"round_{round_idx+1}", "alpaca_eval")
                runner = AlpacaEvalRunner(output_dir)
                result = runner.run(
                    outputs,
                    annotators_config=self.cfg.eval.alpaca_eval_annotators_config,
                )
                self.record_runner_result("alpaca_eval", result, metrics, artifacts, output_dir)
                artifacts["alpaca_eval_result_dir"] = result["result_dir"]
                artifacts["alpaca_eval_outputs_path"] = result["outputs_path"]
            finally:
                self.release_round_model(round_model)

        if self.cfg.eval.run_mt_bench:
            output_dir = os.path.join(self.eval_dir, f"round_{round_idx+1}", "mt_bench")
            runner = MTBenchRunner(output_dir)
            result = runner.run(
                model_path=merged_dir,
                model_id=f"{self.cfg.eval.mt_bench_model_id}_r{round_idx+1}",
                judge_model=self.cfg.eval.mt_bench_judge_model,
                parallel=self.cfg.eval.mt_bench_parallel,
            )
            self._record_mt_bench_result(result, metrics, artifacts, output_dir)

        if self.cfg.eval.run_lm_eval and self.cfg.eval.lm_eval_tasks:
            output_dir = os.path.join(self.eval_dir, f"round_{round_idx+1}", "lm_eval")
            runner = LMEvalRunner(output_dir)
            result = runner.run(
                model_path=merged_dir,
                tasks=self.cfg.eval.lm_eval_tasks,
                batch_size=self.cfg.eval.lm_eval_batch_size,
                device=self.cfg.eval.lm_eval_device,
            )
            self.record_runner_result("lm_eval", result, metrics, artifacts, output_dir)
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

    def _record_mt_bench_result(self, result: dict, metrics: dict, artifacts: dict, output_dir: str) -> None:
        metrics["mt_bench_gen_returncode"] = float(result["gen_returncode"])
        metrics["mt_bench_judge_returncode"] = float(result["judge_returncode"])
        metrics["mt_bench_show_returncode"] = float(result["show_returncode"])
        failed = any(result[key] != 0 for key in ("gen_returncode", "judge_returncode", "show_returncode"))
        metrics["mt_bench/failed"] = float(failed)
        if result["mt_bench_score"] is not None:
            metrics["mt_bench/score"] = float(result["mt_bench_score"])
        artifacts["mt_bench_answer_file"] = result["answer_file"]
        artifacts["mt_bench_judgment_file"] = result["judgment_file"]

        if failed:
            os.makedirs(output_dir, exist_ok=True)
            for name in ("gen", "judge", "show"):
                stdout_path = os.path.join(output_dir, f"mt_bench_{name}_stdout.txt")
                stderr_path = os.path.join(output_dir, f"mt_bench_{name}_stderr.txt")
                with open(stdout_path, "w", encoding="utf-8") as f:
                    f.write(result.get(f"{name}_stdout", ""))
                with open(stderr_path, "w", encoding="utf-8") as f:
                    f.write(result.get(f"{name}_stderr", ""))
                artifacts[f"mt_bench_{name}_stdout_path"] = stdout_path
                artifacts[f"mt_bench_{name}_stderr_path"] = stderr_path
            if self.cfg.eval.fail_on_eval_error:
                raise RuntimeError(f"mt_bench failed. Logs saved under {output_dir}")
