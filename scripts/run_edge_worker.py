from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import fedpost.algorithms  # noqa: F401
import fedpost.trainers  # noqa: F401
from fedpost.data.collators.sft_collator import SFTCollator
from fedpost.data.dataset_builder import DatasetBuilder
from fedpost.data.federated_dataset import ClientContext
from fedpost.federation.client import Client
from fedpost.federation.message import BroadcastPayload
from fedpost.models.loader import HFModelManager
from fedpost.utils.config import ConfigLoader
from fedpost.utils.registry import Registry
from fedpost.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one BP-FedPEFT client update on an edge node.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-data", required=True)
    parser.add_argument("--broadcast", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = ConfigLoader.from_yaml(args.config)
    cfg.data.source = "local"
    cfg.data.data_path = args.client_data
    cfg.federated.num_clients = 1
    cfg.federated.clients_per_round = 1
    ConfigLoader.validate(cfg)
    set_seed(cfg.seed)

    payload = load_payload(args.broadcast)
    model_manager = HFModelManager(cfg)
    model_bundle = model_manager.build()
    trainer_name = "bpfedpeft_sft" if cfg.federated.algorithm == "bpfedpeft" else cfg.task
    trainer_cls = Registry.get("trainer", trainer_name)
    collator = SFTCollator(model_bundle.tokenizer, max_length=cfg.sft.max_length)
    trainer = trainer_cls(
        cfg=cfg,
        model_bundle=model_bundle,
        model_manager=model_manager,
        collator=collator,
    )

    dataset = DatasetBuilder(cfg).build_task_dataset()
    client = Client(
        context=ClientContext(
            client_id=args.client_id,
            num_samples=len(dataset),
            metadata={"client_data": args.client_data, "runner": "edge_worker"},
        ),
        trainer=trainer,
        dataset=dataset,
    )
    result = client.run_round(payload, device=args.device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(result, args.output)
    if not result.success:
        raise RuntimeError(result.error_msg or "edge client failed")


def load_payload(path: str) -> BroadcastPayload:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, BroadcastPayload):
        return payload
    return BroadcastPayload(**payload)


if __name__ == "__main__":
    main()
