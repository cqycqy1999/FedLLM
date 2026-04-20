from __future__ import annotations

from dataclasses import dataclass

from fedpost.data.federated_dataset import ClientContext, FederatedDataset
from fedpost.data.io import load_records
from fedpost.data.processors import (
    SFTSample,
    DPOSample,
    build_sft_sample,
    build_dpo_sample,
)
from fedpost.utils.registry import Registry


class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_task_dataset(self) -> list:
        raw_records = load_records(
            path=self.cfg.data.data_path,
            file_type=self.cfg.data.file_type,
        )

        if self.cfg.task == "sft":
            samples = []
            for rec in raw_records:
                sample = build_sft_sample(rec, self.cfg)
                if sample is not None:
                    samples.append(sample)
            if not samples:
                raise ValueError("No valid SFT samples found in data file.")
            return samples

        if self.cfg.task == "dpo":
            samples = []
            for rec in raw_records:
                sample = build_dpo_sample(rec, self.cfg)
                if sample is not None:
                    samples.append(sample)
            if not samples:
                raise ValueError("No valid DPO samples found in data file.")
            return samples

        raise ValueError(f"Unsupported task: {self.cfg.task}")

    def build_federated_dataset(self) -> FederatedDataset:
        task_dataset = self.build_task_dataset()

        partitioner_cls = Registry.get("partitioner", self.cfg.data.partitioner)
        partitioner = partitioner_cls(self.cfg)
        client_to_indices = partitioner.partition(task_dataset)

        client_to_data = {
            client_id: [task_dataset[i] for i in indices]
            for client_id, indices in client_to_indices.items()
        }
        client_contexts = {
            client_id: ClientContext(
                client_id=client_id,
                num_samples=len(data),
                metadata={"task": self.cfg.task},
            )
            for client_id, data in client_to_data.items()
        }
        return FederatedDataset(client_to_data, client_contexts)