from __future__ import annotations

from fedpost.trainers.base_trainer import BaseTrainer
from fedpost.utils.registry import Registry


@Registry.register("trainer", "sft")
class SFTTrainer(BaseTrainer):
    def compute_loss(self, batch: dict):
        batch = self._move_batch_to_device(batch)
        outputs = self.model(**batch)
        loss = outputs.loss
        return loss, {"loss": float(loss.detach().cpu())}
