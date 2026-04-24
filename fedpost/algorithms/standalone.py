from __future__ import annotations

import torch

from fedpost.algorithms.base import FederatedAlgorithm
from fedpost.utils.registry import Registry


class StandaloneAggregator:
    def __init__(self, cfg):
        self.cfg = cfg

    def aggregate(self, client_results):
        valid_results = [result for result in client_results if result.success]
        if not valid_results:
            raise RuntimeError("Standalone training produced no valid client result.")
        if len(valid_results) != 1:
            valid_client_ids = [result.client_id for result in valid_results]
            raise RuntimeError(
                "standalone expects exactly one successful client result, "
                f"but received {len(valid_results)}: {valid_client_ids}"
            )

        result = valid_results[0]
        aggregated_state = {}
        for key, value in result.update.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Client {result.client_id} produced non-tensor update for key {key}: "
                    f"{type(value).__name__}"
                )
            aggregated_state[key] = value.detach().cpu().clone()

        metrics = {
            "num_success_clients": 1,
            "aggregated_client_ids": [result.client_id],
            "aggregation_reference_client_id": result.client_id,
            "aggregation_mode": "standalone_single_client",
            "aggregation_update_keys": list(aggregated_state.keys()),
            "aggregation_update_key_count": len(aggregated_state),
        }
        return aggregated_state, metrics


@Registry.register("algorithm", "standalone")
class StandaloneAlgorithm(FederatedAlgorithm):
    aggregator_cls = StandaloneAggregator
