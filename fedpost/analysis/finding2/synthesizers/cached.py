from __future__ import annotations

from typing import Any, Iterable

import torch

from fedpost.analysis.finding2.synthesizers.base import (
    Synthesizer,
    collect_layer_matrix,
)


class CachedMeanSynthesizer(Synthesizer):
    """S1 Cached.

    Uses the calibration mean of h^(i) as a previous-round cache.
    Params: D cached scalars. FLOPs: 0. Communication: 0 bytes.
    """

    name = "S1_cached"

    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        matrix = collect_layer_matrix(calibration_loader, layer_idx)
        self._set_target_meta(matrix[0], layer_idx)
        self.mean = matrix.mean(dim=0).to(torch.float32)
        self._anchor_bytes = 0

    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        self._require_fitted()
        target = h_full_trace[self.layer_idx]
        return {"shape": tuple(target.shape), "device": str(target.device), "dtype": str(target.dtype)}

    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        self._require_fitted()
        shape = anchor_payload["shape"]
        return self.mean.to(device=self.device, dtype=self.dtype).expand(shape).clone()

    def cost_flops(self) -> int:
        return 0

    def cost_params(self) -> int:
        self._require_fitted()
        return int(self.hidden_dim)
