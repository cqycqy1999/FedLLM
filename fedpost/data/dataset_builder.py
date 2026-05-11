from __future__ import annotations

import os

import fedpost.data.adapters

from fedpost.data.federated_dataset import ClientContext, FederatedDataset
from fedpost.data.hf_dataset_builder import HFDatasetLoader
from fedpost.data.io import load_records
from fedpost.data.processors import SFTSample
from fedpost.utils.registry import Registry


class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build_task_dataset(self) -> list:
        if self.cfg.data.source == "local":
            return self._build_local_dataset()

        if self.cfg.data.source != "hf":
            raise ValueError(f"Unsupported data source: {self.cfg.data.source}")

        hf_ds = HFDatasetLoader(self.cfg).load()
        adapter_cls = Registry.get("dataset_adapter", self.cfg.data.dataset_name)
        adapter = adapter_cls(self.cfg)

        samples = []
        for rec in hf_ds:
            sample = adapter.to_sft_sample(rec)
            if sample is not None:
                samples.append(sample)

        if not samples:
            raise ValueError("No valid samples were parsed from the dataset.")
        return samples

    def _build_local_dataset(self) -> list:
        if not self.cfg.data.data_path:
            raise ValueError("data.data_path is required when data.source='local'.")
        if os.path.isdir(self.cfg.data.data_path):
            raise ValueError(
                "data.data_path points to a directory of client shards. "
                "Call build_federated_dataset() so the pre-partitioned layout can be used."
            )

        records = load_records(self.cfg.data.data_path, self.cfg.data.file_type)
        if self.cfg.data.max_samples is not None:
            records = records[:self.cfg.data.max_samples]

        return self._records_to_samples(records)

    def _records_to_samples(self, records: list[dict]) -> list[SFTSample]:
        samples = []
        for rec in records:
            prompt = _clean(rec.get(self.cfg.data.prompt_field))
            response = _clean(rec.get(self.cfg.data.response_field))
            if prompt and response:
                metadata = dict(rec)
                metadata["source"] = metadata.get("source", "local")
                samples.append(SFTSample(prompt=prompt, response=response, metadata=metadata))

        if not samples:
            raise ValueError("No valid local samples were parsed from the dataset.")
        return samples

    def build_federated_dataset(self) -> FederatedDataset:
        if self.cfg.data.source == "local" and self.cfg.data.data_path:
            if os.path.isdir(self.cfg.data.data_path):
                return self._build_prepartitioned_federated_dataset(self.cfg.data.data_path)

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

    def _build_prepartitioned_federated_dataset(self, shard_dir: str) -> FederatedDataset:
        client_to_data = {}
        client_contexts = {}
        shard_paths = [
            os.path.join(shard_dir, name)
            for name in sorted(os.listdir(shard_dir))
            if _is_client_shard_file(name)
        ]
        if not shard_paths:
            raise ValueError(f"No .jsonl or .json client shards found under {shard_dir}")

        for path in shard_paths:
            client_id = os.path.splitext(os.path.basename(path))[0]
            file_type = "jsonl" if path.endswith(".jsonl") else "json"
            records = load_records(path, file_type)
            samples = self._records_to_samples(records)
            client_to_data[client_id] = samples
            labels = sorted({
                str(sample.metadata.get(self.cfg.data.semantic_label_field))
                for sample in samples
                if sample.metadata and sample.metadata.get(self.cfg.data.semantic_label_field) is not None
            })
            client_contexts[client_id] = ClientContext(
                client_id=client_id,
                num_samples=len(samples),
                metadata={
                    "task": self.cfg.task,
                    "prepartitioned": True,
                    "shard_path": path,
                    "semantic_labels": labels,
                },
            )

        return FederatedDataset(client_to_data, client_contexts)


def _clean(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _is_client_shard_file(name: str) -> bool:
    if name.startswith("."):
        return False
    if name in {"manifest.json", "metadata.json"}:
        return False
    return name.endswith((".jsonl", ".json"))
