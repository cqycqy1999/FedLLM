from __future__ import annotations

import unittest

from fedpost.bpfedpeft.planner import BlockSpec
from fedpost.bpfedpeft.state import extract_layer_index, state_for_block


class StateTest(unittest.TestCase):
    def test_extract_common_layer_paths(self):
        self.assertEqual(extract_layer_index("base_model.model.model.layers.12.self_attn.q_proj.lora_A.weight"), 12)
        self.assertEqual(extract_layer_index("base_model.model.transformer.h.3.attn.c_attn.lora_B.weight"), 3)
        self.assertEqual(extract_layer_index("decoder.layers.7.mlp.down_proj.lora_A.weight"), 7)

    def test_state_for_block_filters_layer_keys(self):
        state = {
            "model.layers.0.q_proj.lora_A.weight": "a",
            "model.layers.1.q_proj.lora_A.weight": "b",
            "model.layers.3.q_proj.lora_A.weight": "c",
            "lm_head.modules_to_save.default.weight": "head",
        }
        filtered = state_for_block(state, BlockSpec(1, 2), include_unindexed=False)
        self.assertEqual(filtered, {"model.layers.1.q_proj.lora_A.weight": "b"})


if __name__ == "__main__":
    unittest.main()
