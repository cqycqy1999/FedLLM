from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from fedpost.bpfedpeft.planner import blocks_from_end_layers, plan_equal_blocks
from fedpost.data.collators.sft_collator import SFTCollator
from fedpost.data.dataset_builder import DatasetBuilder
from fedpost.models.block_runtime import infer_num_decoder_layers
from fedpost.models.block_vectors import compute_block_depth_vectors, save_block_depth_vectors
from fedpost.models.loader import HFModelManager
from fedpost.utils.config import ConfigLoader
from fedpost.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute BP-FedPEFT depth vectors.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()

    cfg = ConfigLoader.from_yaml(args.config)
    ConfigLoader.validate(cfg)
    set_seed(cfg.seed)

    model_manager = HFModelManager(cfg)
    bundle = model_manager.build()
    dataset = DatasetBuilder(cfg).build_task_dataset()
    collator = SFTCollator(bundle.tokenizer, max_length=cfg.sft.max_length)
    dataloader = DataLoader(dataset, batch_size=cfg.sft.batch_size, collate_fn=collator)

    if cfg.bpfedpeft.block_end_layers:
        blocks = blocks_from_end_layers(
            cfg.bpfedpeft.block_end_layers,
            overlap_layers=cfg.bpfedpeft.overlap_layers,
        )
    else:
        num_layers = infer_num_decoder_layers(bundle.model)
        blocks = plan_equal_blocks(
            num_layers=num_layers,
            num_blocks=cfg.bpfedpeft.num_blocks,
            overlap_layers=cfg.bpfedpeft.overlap_layers,
        )

    vectors = compute_block_depth_vectors(
        bundle.model,
        dataloader,
        blocks,
        max_batches=args.max_batches,
    )
    output = args.output or cfg.bpfedpeft.vector_path
    if output is None:
        output = os.path.join(cfg.output_dir, "bpfedpeft_depth_vectors.pt")
    print(save_block_depth_vectors(vectors, output))


if __name__ == "__main__":
    main()
