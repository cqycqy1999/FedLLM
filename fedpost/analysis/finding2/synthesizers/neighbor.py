from __future__ import annotations

from typing import Any, Iterable

import torch

from fedpost.analysis.finding2.synthesizers.base import (
    Synthesizer,
    collect_layer_matrix,
    tensor_bytes,
)


class NeighborLinearSynthesizer(Synthesizer):
    """S4 Neighbor.

    Uses h^(i-k) and fits per-dimension affine regression:
      h_hat[d] = a[d] * h_neighbor[d] + b[d].
    Params: 2D. FLOPs: 2D. Communication: D * bytes(dtype).
    """

    def __init__(self, k: int = 1, seed: int = 42):
        super().__init__(seed=seed)
        self.k = k
        self.name = f"S4_neighbor_k{k}"

    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        source_idx = layer_idx - self.k
        if source_idx < 0:
            raise ValueError(f"Neighbor layer {source_idx} is invalid for layer {layer_idx}.")
        x = collect_layer_matrix(calibration_loader, source_idx)
        y = collect_layer_matrix(calibration_loader, layer_idx)
        self._set_target_meta(y[0], layer_idx)
        self.source_idx = source_idx

        x_mean = x.mean(dim=0)
        y_mean = y.mean(dim=0)
        cov = ((x - x_mean) * (y - y_mean)).mean(dim=0)
        var = ((x - x_mean) ** 2).mean(dim=0).clamp_min(1e-8)
        self.scale = cov / var
        self.bias = y_mean - self.scale * x_mean

    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        h = h_full_trace[self.source_idx]
        self._anchor_bytes = tensor_bytes(h)
        return {"h_neighbor": h.detach().clone()}

    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        h = anchor_payload["h_neighbor"].to(torch.float32)
        out = h * self.scale.reshape(*([1] * (h.ndim - 1)), -1) + self.bias.reshape(*([1] * (h.ndim - 1)), -1)
        return out.to(device=self.device, dtype=self.dtype)

    def cost_flops(self) -> int:
        self._require_fitted()
        return int(2 * self.hidden_dim)

    def cost_params(self) -> int:
        self._require_fitted()
        return int(2 * self.hidden_dim)
