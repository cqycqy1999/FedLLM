from dataclasses import dataclass


@dataclass
class ModelStateSpec:
    state_type: str
    trainable_keys: list[str]
    aggregatable_keys: list[str]
    frozen_keys: list[str]