from __future__ import annotations

from fedpost.data.adapters.base import BaseDatasetAdapter
from fedpost.data.processors import SFTSample
from fedpost.utils.registry import Registry


@Registry.register("dataset_adapter", "databricks/databricks-dolly-15k")
class DollySFTAdapter(BaseDatasetAdapter):
    def to_sft_sample(self, record: dict) -> SFTSample | None:
        instruction = str(record.get("instruction", "")).strip()
        context = str(record.get("context", "")).strip()
        response = str(record.get("response", "")).strip()

        if not instruction or not response:
            return None

        prompt = instruction
        if context:
            prompt = f"{instruction}\n\nContext:\n{context}"

        return SFTSample(
            prompt=prompt,
            response=response,
            metadata={
                "category": record.get("category"),
                "source_dataset": "databricks/databricks-dolly-15k",
            },
        )

    def to_dpo_sample(self, record: dict):
        return None