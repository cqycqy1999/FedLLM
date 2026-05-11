from __future__ import annotations

from collections import defaultdict
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


@Registry.register("partitioner", "semantic_dirichlet")
class SemanticDirichletPartitioner(BasePartitioner):
    """Dirichlet partitioner over semantic pseudo-labels."""

    def partition(self, dataset: list) -> dict[str, list[int]]:
        n_clients = self.cfg.federated.num_clients
        alpha = self.cfg.data.dirichlet_alpha
        if alpha <= 0:
            raise ValueError("data.dirichlet_alpha must be positive")

        label_field = self.cfg.data.semantic_label_field
        by_label = defaultdict(list)
        for idx, sample in enumerate(dataset):
            metadata = getattr(sample, "metadata", None) or {}
            label = metadata.get(label_field, "unlabeled")
            by_label[str(label)].append(idx)

        rnd = random.Random(self.cfg.data.partition_seed)
        out = {f"client_{i}": [] for i in range(n_clients)}
        for indices in by_label.values():
            rnd.shuffle(indices)
            weights = [rnd.gammavariate(alpha, 1.0) for _ in range(n_clients)]
            total = sum(weights)
            if total == 0:
                weights = [1.0 for _ in range(n_clients)]
                total = float(n_clients)

            counts = [int(len(indices) * weight / total) for weight in weights]
            while sum(counts) < len(indices):
                counts[min(range(n_clients), key=lambda i: counts[i])] += 1

            cursor = 0
            for client_idx, count in enumerate(counts):
                out[f"client_{client_idx}"].extend(indices[cursor:cursor + count])
                cursor += count

        self._rebalance_empty_clients(out, rnd)
        for values in out.values():
            rnd.shuffle(values)
        return out

    @staticmethod
    def _rebalance_empty_clients(out: dict[str, list[int]], rnd: random.Random) -> None:
        empty = [client_id for client_id, values in out.items() if not values]
        for empty_client in empty:
            donor = max(out, key=lambda client_id: len(out[client_id]))
            if len(out[donor]) <= 1:
                break
            moved_idx = rnd.randrange(len(out[donor]))
            out[empty_client].append(out[donor].pop(moved_idx))
