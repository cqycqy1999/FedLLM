from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch


class Synthesizer(ABC):
    name: str = "base"

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.layer_idx: int | None = None
        self.hidden_dim: int | None = None
        self.dtype: torch.dtype | None = None
        self.device = torch.device("cpu")
        self._anchor_numel = 0
        self._anchor_bytes = 0

    @abstractmethod
    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        raise NotImplementedError

    @abstractmethod
    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def cost_flops(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def cost_params(self) -> int:
        raise NotImplementedError

    def cost_comm_bytes(self) -> int:
        return int(self._anchor_bytes)

    def _set_target_meta(self, tensor: torch.Tensor, layer_idx: int) -> None:
        self.layer_idx = layer_idx
        self.hidden_dim = int(tensor.shape[-1])
        self.dtype = tensor.dtype
        self.device = tensor.device

    def _require_fitted(self) -> None:
        if self.layer_idx is None or self.hidden_dim is None or self.dtype is None:
            raise RuntimeError(f"{self.name} has not been fitted.")


def flatten_trace_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor.reshape(-1, tensor.shape[-1])


def collect_layer_matrix(
    calibration_loader: Iterable[dict[int, torch.Tensor]],
    layer_idx: int,
) -> torch.Tensor:
    rows = []
    for trace in calibration_loader:
        if layer_idx not in trace:
            raise KeyError(f"Layer {layer_idx} missing from calibration trace.")
        rows.append(flatten_trace_tensor(trace[layer_idx]).detach().cpu().to(torch.float32))
    if not rows:
        raise ValueError("Calibration loader is empty.")
    return torch.cat(rows, dim=0)


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())
