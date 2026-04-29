from __future__ import annotations

import json
import os

import torch
from huggingface_hub import hf_hub_download

from fedpost.evaluation.base import Evaluator
from fedpost.evaluation.runners.alpaca_eval_runner import AlpacaEvalRunner
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

        if not self.cfg.eval.run_alpaca_eval and not (self.cfg.eval.run_lm_eval and self.cfg.eval.lm_eval_tasks):
            return EvalResult(
                round_idx=round_idx,
                split="val",
                metrics=metrics,
                artifacts=artifacts,
            )

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

        self.run_lm_eval(round_idx, model_artifacts, metrics, artifacts)

        return EvalResult(
            round_idx=round_idx,
            split="val",
            metrics=metrics,
            artifacts=artifacts,
        )
