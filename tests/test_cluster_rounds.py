from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.run_cluster_rounds import reconcile_client_count, sample_clients, select_manifest_clients


class ClusterRoundsTest(unittest.TestCase):
    def test_auto_adapts_client_count(self):
        cfg = SimpleNamespace(
            federated=SimpleNamespace(num_clients=14, clients_per_round=14, proportion=None),
            seed=42,
        )
        clients = [{"client_id": f"client_{idx}"} for idx in range(5)]
        reconcile_client_count(cfg, clients)
        self.assertEqual(cfg.federated.num_clients, 5)
        self.assertEqual(cfg.federated.clients_per_round, 5)
        self.assertEqual(len(sample_clients(clients, cfg, 0)), 5)

    def test_clients_per_round_override(self):
        cfg = SimpleNamespace(
            federated=SimpleNamespace(num_clients=14, clients_per_round=14, proportion=1.0),
            seed=42,
        )
        clients = [{"client_id": f"client_{idx}"} for idx in range(8)]
        reconcile_client_count(cfg, clients, clients_per_round=3)
        self.assertEqual(cfg.federated.num_clients, 8)
        self.assertEqual(cfg.federated.clients_per_round, 3)
        self.assertIsNone(cfg.federated.proportion)
        self.assertEqual(len(sample_clients(clients, cfg, 0)), 3)

    def test_strict_count_rejects_mismatch(self):
        cfg = SimpleNamespace(
            federated=SimpleNamespace(num_clients=14, clients_per_round=14, proportion=None),
            seed=42,
        )
        clients = [{"client_id": f"client_{idx}"} for idx in range(5)]
        with self.assertRaises(ValueError):
            reconcile_client_count(cfg, clients, strict_client_count=True)

    def test_selected_client_ids(self):
        clients = [{"client_id": f"client_{idx}"} for idx in range(5)]
        selected = select_manifest_clients(clients, ["client_1", "client_3"])
        self.assertEqual([client["client_id"] for client in selected], ["client_1", "client_3"])


if __name__ == "__main__":
    unittest.main()
