from __future__ import annotations

import torch

from fedpost.trainers.base_trainer import BaseTrainer
from fedpost.utils.registry import Registry


@Registry.register("trainer", "sft")
class SFTTrainer(BaseTrainer):
    def build_optimizer(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.cfg.sft.lr)

    def compute_loss(self, batch: dict):
        batch = self._move_batch_to_device(batch)
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss, {"loss": float(loss.detach().cpu())}

    def train_one_round(self, dataset, round_idx: int):
        self.model.train()
        dataloader = self.build_dataloader(dataset)

        metrics_list = []
        for step_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            loss, metrics = self.compute_loss(batch)
            loss.backward()
            self.optimizer.step()
            metrics_list.append(metrics)

            if self._reach_local_budget(step_idx):
                break

        update = self.get_trainable_state()
        metrics = self._aggregate_metrics(metrics_list)
        return update, metrics