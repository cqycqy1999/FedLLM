from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta: float,
):
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios

    loss = -F.logsigmoid(beta * logits).mean()

    aux = {
        "preference_accuracy": float((logits > 0).float().mean().detach().cpu()),
        "margin": float(logits.mean().detach().cpu()),
    }
    return loss, aux