from __future__ import annotations

import unittest

from fedpost.bpfedpeft.planner import (
    blocks_from_end_layers,
    linear_cka,
    plan_equal_blocks,
    solve_cka_partition,
)


class PlannerTest(unittest.TestCase):
    def test_blocks_from_paper_end_layers(self):
        blocks = blocks_from_end_layers([2, 4, 6], overlap_layers=1)
        self.assertEqual([(b.start, b.end) for b in blocks], [(0, 1), (1, 3), (3, 5)])

    def test_equal_blocks_cover_model(self):
        blocks = plan_equal_blocks(num_layers=8, num_blocks=4, overlap_layers=1)
        self.assertEqual(blocks[0].start, 0)
        self.assertEqual(blocks[-1].end, 7)
        self.assertEqual([(b.start, b.end) for b in blocks], [(0, 1), (1, 3), (3, 5), (5, 7)])

    def test_cka_partition_prefers_low_similarity_boundary(self):
        similarity = [
            [1.0, 0.9, 0.9, 0.9],
            [0.9, 1.0, 0.1, 0.9],
            [0.9, 0.1, 1.0, 0.8],
            [0.9, 0.9, 0.8, 1.0],
        ]
        blocks = solve_cka_partition(similarity, num_blocks=2, max_block_layers=3)
        self.assertEqual([(b.start, b.end) for b in blocks], [(0, 1), (1, 3)])

    def test_linear_cka_identity(self):
        x = [[1, 0], [0, 1], [1, 1]]
        self.assertAlmostEqual(linear_cka(x, x), 1.0)


if __name__ == "__main__":
    unittest.main()
