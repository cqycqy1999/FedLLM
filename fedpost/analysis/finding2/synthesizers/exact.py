from __future__ import annotations

from typing import Any, Iterable

import torch

from fedpost.analysis.finding2.synthesizers.base import Synthesizer, tensor_bytes


class ExactSynthesizer(Synthesizer):
    """S0 Exact.

    Anchor contains the full target vector h^(i).
    Params: 0. FLOPs: 0. Communication: D * bytes(dtype).
    """

    name = "S0_exact"

    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        for trace in calibration_loader:
            if layer_idx not in trace:
                raise KeyError(f"Layer {layer_idx} missing from calibration trace.")
            self._set_target_meta(trace[layer_idx], layer_idx)
            return
        raise ValueError("Calibration loader is empty.")

    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        self._require_fitted()
        h = h_full_trace[self.layer_idx]
        self._anchor_bytes = tensor_bytes(h)
        return {"h": h.detach().clone()}

    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        return anchor_payload["h"].to(device=self.device, dtype=self.dtype)

    def cost_flops(self) -> int:
        return 0

    def cost_params(self) -> int:
        return 0
