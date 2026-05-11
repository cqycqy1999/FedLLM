from __future__ import annotations

from abc import ABC


class BaseDatasetAdapter(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    def to_sft_sample(self, record: dict):
        raise NotImplementedError
