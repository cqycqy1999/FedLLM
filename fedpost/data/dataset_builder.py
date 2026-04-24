from __future__ import annotations

import fedpost.data.adapters

from fedpost.data.federated_dataset import ClientContext, FederatedDataset
from fedpost.data.hf_dataset_builder import HFDatasetLoader
from fedpost.utils.registry import Registry


class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_task_dataset(self) -> list:
        if self.cfg.data.source != "hf":
            raise ValueError("This template currently supports HF datasets only.")

        hf_ds = HFDatasetLoader(self.cfg).load()
        adapter_cls = Registry.get("dataset_adapter", self.cfg.data.dataset_name)
        adapter = adapter_cls(self.cfg)

        samples = []
        if self.cfg.task == "sft":
            for rec in hf_ds:
                sample = adapter.to_sft_sample(rec)
                if sample is not None:
                    samples.append(sample)
        elif self.cfg.task == "dpo":
            for rec in hf_ds:
                sample = adapter.to_dpo_sample(rec)
                if sample is not None:
                    samples.append(sample)
        else:
            raise ValueError(f"Unsupported task: {self.cfg.task}")

        if not samples:
            raise ValueError("No valid samples were parsed from the dataset.")
        return samples

    def build_federated_dataset(self) -> FederatedDataset:
        task_dataset = self.build_task_dataset()

        if self.cfg.federated.algorithm == "standalone":
            client_id = "local_client"
            client_to_data = {client_id: task_dataset}
            client_contexts = {
                client_id: ClientContext(
                    client_id=client_id,
                    num_samples=len(task_dataset),
                    metadata={
                        "task": self.cfg.task,
                        "algorithm": self.cfg.federated.algorithm,
                        "mode": "single_machine_standalone",
                    },
                )
            }
            return FederatedDataset(client_to_data, client_contexts)

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
