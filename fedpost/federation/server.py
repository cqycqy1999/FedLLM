from __future__ import annotations

import os
import torch

from fedpost.federation.message import BroadcastPayload


class Server:
    def __init__(self, cfg, global_model_manager, model_bundle, aggregator, logger=None):
        self.cfg = cfg
        self.model_manager = global_model_manager
        self.model_bundle = model_bundle
        self.aggregator = aggregator
        self.logger = logger

        self.round_idx = 0
        self.global_step = 0

    def get_broadcast_payload(self) -> BroadcastPayload:
        model_state = self.model_manager.get_trainable_state(self.model_bundle.model)
        return BroadcastPayload(
            round_idx=self.round_idx,
            global_step=self.global_step,
            model_state=model_state,
            algo_state={},
            metadata={},
        )

    def apply_updates(self, client_results):
        aggregated_state, agg_metrics = self.aggregator.aggregate(client_results)
        self.model_manager.load_trainable_state(self.model_bundle.model, aggregated_state)
        self.global_step += 1
        return agg_metrics

    def save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trainable_state = self.model_manager.get_trainable_state(self.model_bundle.model)
        torch.save(
            {
                "round_idx": self.round_idx,
                "global_step": self.global_step,
                "trainable_state": trainable_state,
                "state_type": self.model_bundle.model_state_spec.state_type,
            },
            path,
        )

    def evaluate_model(self):
        return self.model_bundle.model