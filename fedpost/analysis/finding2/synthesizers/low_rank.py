from __future__ import annotations

from typing import Any, Iterable

import torch

from fedpost.analysis.finding2.synthesizers.base import (
    Synthesizer,
    collect_layer_matrix,
    flatten_trace_tensor,
    tensor_bytes,
)


class LowRankSynthesizer(Synthesizer):
    """S2 Low-rank PCA/SVD compression.

    Fit stores mean m and basis V_r from calibration SVD.
    Anchor contains z = (h - m) V_r. Reconstruction is m + z V_r^T.
    Params: D + D*r. FLOPs: D*r for anchor build and D*r for synthesis.
    Communication: r * bytes(dtype).
    """

    def __init__(self, rank: int, seed: int = 42):
        super().__init__(seed=seed)
        self.rank = rank
        self.name = f"S2_low_rank_r{rank}"

    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        matrix = collect_layer_matrix(calibration_loader, layer_idx)
        self._set_target_meta(matrix[0], layer_idx)
        self.mean = matrix.mean(dim=0)
        centered = matrix - self.mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        r = min(self.rank, vh.shape[0])
        self.basis = vh[:r].contiguous()

    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        self._require_fitted()
        h = flatten_trace_tensor(h_full_trace[self.layer_idx]).to(torch.float32)
        z = torch.matmul(h - self.mean, self.basis.T)
        z = z.reshape(*h_full_trace[self.layer_idx].shape[:-1], self.basis.shape[0])
        z = z.to(dtype=self.dtype)
        self._anchor_bytes = tensor_bytes(z)
        return {"z": z, "shape": tuple(h_full_trace[self.layer_idx].shape)}

    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        z = flatten_trace_tensor(anchor_payload["z"]).to(torch.float32)
        h = torch.matmul(z, self.basis) + self.mean
        h = h.reshape(anchor_payload["shape"])
        return h.to(device=self.device, dtype=self.dtype)

    def cost_flops(self) -> int:
        self._require_fitted()
        return int(2 * self.hidden_dim * self.basis.shape[0])

    def cost_params(self) -> int:
        self._require_fitted()
        return int(self.hidden_dim + self.hidden_dim * self.basis.shape[0])
