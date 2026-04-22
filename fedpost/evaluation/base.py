from __future__ import annotations


class Evaluator:
    def __init__(self, cfg, tokenizer, logger=None):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.logger = logger

    def evaluate(self, model, round_idx: int, model_artifacts: dict | None = None):
        raise NotImplementedError