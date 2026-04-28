from __future__ import annotations

from typing import Any, Iterable

import torch

from fedpost.analysis.finding2.synthesizers.base import (
    Synthesizer,
    collect_layer_matrix,
    flatten_trace_tensor,
)


class AnchorCorrectionSynthesizer(Synthesizer):
    """S5 Anchor+Correction.

    Main method: low-rank reconstruction plus sparse top-k exact correction.
    Fit stores mean m and PCA basis V_r. Anchor contains z=(h-m)V_r plus
    top-k indices and values from h. Synthesis computes m+zV_r^T and overwrites
    top-k coordinates with exact values.
    Params: D + D*r. FLOPs: D*r for synthesis. Communication:
      r * bytes(dtype) + top_k * (bytes(dtype) + 8 bytes index).
    """

    def __init__(self, rank: int = 64, top_k: int = 64, seed: int = 42):
        super().__init__(seed=seed)
        self.rank = rank
        self.top_k = top_k
        self.name = f"S5_anchor_correction_r{rank}_k{top_k}"

    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        matrix = collect_layer_matrix(calibration_loader, layer_idx)
        self._set_target_meta(matrix[0], layer_idx)
        self.mean = matrix.mean(dim=0)
        centered = matrix - self.mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        r = min(self.rank, vh.shape[0])
        self.basis = vh[:r].contiguous()

    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        target = h_full_trace[self.layer_idx]
        h = flatten_trace_tensor(target).to(torch.float32)
        centered = h - self.mean
        z = centered @ self.basis.T
        k = min(self.top_k, h.shape[-1])
        values, indices = torch.topk(h.abs(), k=k, dim=-1)
        signed_values = h.gather(dim=-1, index=indices)
        self._anchor_bytes = int(
            z.numel() * torch.tensor([], dtype=self.dtype).element_size()
            + signed_values.numel() * torch.tensor([], dtype=self.dtype).element_size()
            + indices.numel() * 8
        )
        return {
            "z": z.reshape(*target.shape[:-1], self.basis.shape[0]).to(dtype=self.dtype),
            "indices": indices.reshape(*target.shape[:-1], k).to(torch.int64),
            "values": signed_values.reshape(*target.shape[:-1], k).to(dtype=self.dtype),
            "shape": tuple(target.shape),
        }

    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        z = flatten_trace_tensor(anchor_payload["z"]).to(torch.float32)
        h = z @ self.basis + self.mean
        indices = flatten_trace_tensor(anchor_payload["indices"]).to(torch.int64)
        values = flatten_trace_tensor(anchor_payload["values"]).to(torch.float32)
        h.scatter_(dim=-1, index=indices, src=values)
        h = h.reshape(anchor_payload["shape"])
        return h.to(device=self.device, dtype=self.dtype)

    def cost_flops(self) -> int:
        self._require_fitted()
        return int(self.hidden_dim * self.basis.shape[0])

    def cost_params(self) -> int:
        self._require_fitted()
        return int(self.hidden_dim + self.hidden_dim * self.basis.shape[0])
