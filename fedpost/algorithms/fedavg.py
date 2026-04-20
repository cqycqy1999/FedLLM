from __future__ import annotations

from fedpost.algorithms.base import FederatedAlgorithm
from fedpost.utils.registry import Registry


class FedAvgAggregator:
    def __init__(self, cfg):
        self.cfg = cfg

    def aggregate(self, client_results):
        valid_results = [r for r in client_results if r.success]
        if not valid_results:
            raise RuntimeError("No valid client results to aggregate.")

        total = sum(r.num_train_samples for r in valid_results)
        keys = valid_results[0].update.keys()

        aggregated = {}
        for key in keys:
            agg_value = None
            for r in valid_results:
                weight = r.num_train_samples / total
                value = r.update[key] * weight
                agg_value = value if agg_value is None else (agg_value + value)
            aggregated[key] = agg_value

        metrics = {
            "num_success_clients": len(valid_results),
        }
        return aggregated, metrics


@Registry.register("algorithm", "fedavg")
class FedAvgAlgorithm(FederatedAlgorithm):
    pass