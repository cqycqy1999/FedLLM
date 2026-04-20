from __future__ import annotations

import random

from fedpost.utils.registry import Registry


class BasePartitioner:
    def __init__(self, cfg):
        self.cfg = cfg

    def partition(self, dataset: list) -> dict[str, list[int]]:
        raise NotImplementedError


@Registry.register("partitioner", "iid")
class IIDPartitioner(BasePartitioner):
    def partition(self, dataset: list) -> dict[str, list[int]]:
        indices = list(range(len(dataset)))
        rnd = random.Random(self.cfg.data.partition_seed)
        rnd.shuffle(indices)

        n_clients = self.cfg.federated.num_clients
        out = {f"client_{i}": [] for i in range(n_clients)}
        for i, idx in enumerate(indices):
            out[f"client_{i % n_clients}"].append(idx)
        return out