from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrainResult:
    client_id: str
    round_idx: int
    num_train_samples: int
    update: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error_msg: Optional[str] = None
    error_traceback: Optional[str] = None


@dataclass
class BroadcastPayload:
    round_idx: int
    global_step: int
    model_state: dict[str, Any]
    algo_state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    round_idx: int
    split: str
    metrics: dict[str, float]
    artifacts: dict[str, str] = field(default_factory=dict)
