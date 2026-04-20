from __future__ import annotations

import random


class ClientSampler:
    def __init__(self, cfg):
        self.cfg = cfg

    def sample(self, clients, round_idx: int):
        raise NotImplementedError


class UniformClientSampler(ClientSampler):
    def sample(self, clients, round_idx: int):
        k = self.cfg.federated.clients_per_round
        rnd = random.Random(self.cfg.seed + round_idx)
        return rnd.sample(clients, k=k)