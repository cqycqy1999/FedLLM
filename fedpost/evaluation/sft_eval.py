from __future__ import annotations

from fedpost.evaluation.base import Evaluator
from fedpost.federation.message import EvalResult
from fedpost.utils.registry import Registry


@Registry.register("evaluator", "sft")
class SFTGenerationEvaluator(Evaluator):
    def evaluate(self, model, round_idx: int):
        metrics = {
            "dummy_sft_metric": 0.0,
        }
        return EvalResult(
            round_idx=round_idx,
            split="val",
            metrics=metrics,
        )