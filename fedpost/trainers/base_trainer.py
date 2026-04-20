from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseTrainer:
    def __init__(self, cfg, model_bundle, model_manager, collator, logger=None):
        self.cfg = cfg
        self.model = model_bundle.model
        self.tokenizer = model_bundle.tokenizer
        self.reference_model = model_bundle.reference_model
        self.model_state_spec = model_bundle.model_state_spec
        self.model_manager = model_manager
        self.collator = collator
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.reference_model is not None:
            self.reference_model.to(self.device)

        self.optimizer = self.build_optimizer()

    def build_optimizer(self):
        raise NotImplementedError

    def build_dataloader(self, dataset):
        batch_size = self.cfg.sft.batch_size if self.cfg.task == "sft" else self.cfg.dpo.batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

    def set_trainable_state(self, state: dict[str, Any]) -> None:
        self.model_manager.load_trainable_state(self.model, state)

    def get_trainable_state(self) -> dict[str, Any]:
        return self.model_manager.get_trainable_state(self.model)

    def train_one_round(self, dataset, round_idx: int):
        raise NotImplementedError

    def compute_loss(self, batch: dict):
        raise NotImplementedError

    def _move_batch_to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _aggregate_metrics(self, metrics_list: list[dict]) -> dict:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

    def _reach_local_budget(self, step_idx: int) -> bool:
        local_steps = self.cfg.federated.local_steps
        if local_steps is None:
            return False
        return (step_idx + 1) >= local_steps