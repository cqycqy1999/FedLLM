from __future__ import annotations

import torch

from fedpost.trainers.base_trainer import BaseTrainer
from fedpost.trainers.loss.dpo_loss import compute_dpo_loss
from fedpost.utils.registry import Registry


@Registry.register("trainer", "dpo")
class DPOTrainer(BaseTrainer):
    def _sequence_logp(self, model, input_ids, attention_mask, response_mask):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_logps = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        valid_mask = response_mask[:, 1:].float()
        seq_logps = (token_logps * valid_mask).sum(dim=-1)
        return seq_logps

    def compute_logps(self, batch: dict):
        batch = self._move_batch_to_device(batch)

        policy_chosen = self._sequence_logp(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_response_mask"],
        )
        policy_rejected = self._sequence_logp(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_response_mask"],
        )

        with torch.no_grad():
            ref_chosen = self._sequence_logp(
                self.reference_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_response_mask"],
            )
            ref_rejected = self._sequence_logp(
                self.reference_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_response_mask"],
            )

        return {
            "policy_chosen_logps": policy_chosen,
            "policy_rejected_logps": policy_rejected,
            "ref_chosen_logps": ref_chosen,
            "ref_rejected_logps": ref_rejected,
        }

    def compute_loss(self, batch: dict):
        logps = self.compute_logps(batch)
        loss, aux = compute_dpo_loss(
            policy_chosen_logps=logps["policy_chosen_logps"],
            policy_rejected_logps=logps["policy_rejected_logps"],
            ref_chosen_logps=logps["ref_chosen_logps"],
            ref_rejected_logps=logps["ref_rejected_logps"],
            beta=self.cfg.dpo.beta,
        )
        metrics = {"loss": float(loss.detach().cpu()), **aux}
        return loss, metrics
