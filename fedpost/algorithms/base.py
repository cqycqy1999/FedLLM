from __future__ import annotations


class FederatedAlgorithm:
    aggregator_cls = None

    def __init__(self, cfg, aggregator):
        self.cfg = cfg
        self.aggregator = aggregator

    def before_broadcast(self, server, clients, round_idx: int):
        return None

    def make_broadcast_payload(self, server, round_idx: int):
        return server.get_broadcast_payload()

    def after_local_train(self, results, round_idx: int):
        return results

    def server_update(self, server, results):
        return server.apply_updates(results)
