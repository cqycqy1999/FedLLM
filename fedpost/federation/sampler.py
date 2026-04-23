from __future__ import annotations

import math
import random


class ClientSampler:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, clients, round_idx: int):
        raise NotImplementedError


class UniformClientSampler(ClientSampler):
    def sample(self, clients, round_idx: int):
        proportion = self.cfg.federated.proportion
        if proportion is None:
            k = self.cfg.federated.clients_per_round
        else:
            k = math.ceil(len(clients) * proportion)
        k = max(1, min(k, len(clients)))
        rnd = random.Random(self.cfg.seed + round_idx)
        return rnd.sample(clients, k=k)
