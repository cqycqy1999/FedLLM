from __future__ import annotations

import unittest

from scripts.prepare_semantic_dirichlet import dirichlet_partition, keyword_label


class SemanticDirichletTest(unittest.TestCase):
    def test_keyword_label(self):
        self.assertEqual(keyword_label("Write a Python function", ["code", "finance", "general"]), "code")
        self.assertEqual(keyword_label("Estimate market revenue", ["code", "finance", "general"]), "finance")
        self.assertEqual(keyword_label("Tell me a story", ["code", "finance", "general"]), "general")

    def test_dirichlet_partition_preserves_records(self):
        records = [{"id": idx, "semantic_label": "code" if idx < 6 else "finance"} for idx in range(12)]
        shards = dirichlet_partition(records, "semantic_label", num_clients=4, alpha=0.3, seed=7)
        self.assertEqual(sum(len(shard) for shard in shards), 12)
        self.assertTrue(all(len(shard) > 0 for shard in shards))
        ids = sorted(row["id"] for shard in shards for row in shard)
        self.assertEqual(ids, list(range(12)))


if __name__ == "__main__":
    unittest.main()
