from __future__ import annotations

from datasets import load_dataset


class HFDatasetLoader:
    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        ds = load_dataset(
            self.cfg.data.dataset_name,
            split=self.cfg.data.dataset_split,
        )
        if self.cfg.data.max_samples is not None:
            max_n = min(self.cfg.data.max_samples, len(ds))
            ds = ds.select(range(max_n))
        return ds