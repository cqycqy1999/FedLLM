from __future__ import annotations

from typing import Any, Iterable

import torch

from fedpost.analysis.finding2.synthesizers.base import (
    Synthesizer,
    collect_layer_matrix,
    flatten_trace_tensor,
    tensor_bytes,
)


class DistilledRandomFeatureSynthesizer(Synthesizer):
    """S3 Distilled.

    Small deterministic random-feature model h^(0) -> h^(i):
      z = h0 R, h_hat = z W + b.
    R is seeded and fixed; W,b are fitted by ridge least squares.
    Params: D*r + r*D + D. FLOPs: D*r + r*D.
    Communication: full h^(0), D * bytes(dtype).
    """

    name = "S3_distilled"

    def __init__(self, feature_dim: int = 128, ridge: float = 1e-3, seed: int = 42):
        super().__init__(seed=seed)
        self.feature_dim = feature_dim
        self.ridge = ridge

    def fit(self, calibration_loader: Iterable[dict[int, torch.Tensor]], layer_idx: int, model=None) -> None:
        x = collect_layer_matrix(calibration_loader, 0)
        y = collect_layer_matrix(calibration_loader, layer_idx)
        self._set_target_meta(y[0], layer_idx)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        self.random_basis = torch.randn(
            x.shape[-1],
            self.feature_dim,
            generator=generator,
            dtype=torch.float32,
        ) / (self.feature_dim ** 0.5)
        z = x @ self.random_basis
        ones = torch.ones(z.shape[0], 1)
        design = torch.cat([z, ones], dim=1)
        eye = torch.eye(design.shape[1])
        lhs = design.T @ design + self.ridge * eye
        rhs = design.T @ y
        self.weights = torch.linalg.solve(lhs, rhs)

    def build_anchor(self, h_full_trace: dict[int, torch.Tensor]) -> Any:
        h0 = h_full_trace[0]
        self._anchor_bytes = tensor_bytes(h0)
        return {"h0": h0.detach().clone(), "shape": tuple(h_full_trace[self.layer_idx].shape)}

    def synthesize(self, anchor_payload: Any) -> torch.Tensor:
        h0 = flatten_trace_tensor(anchor_payload["h0"]).to(torch.float32)
        z = h0 @ self.random_basis
        design = torch.cat([z, torch.ones(z.shape[0], 1)], dim=1)
        h = design @ self.weights
        h = h.reshape(anchor_payload["shape"])
        return h.to(device=self.device, dtype=self.dtype)

    def cost_flops(self) -> int:
        self._require_fitted()
        return int(self.hidden_dim * self.feature_dim + self.feature_dim * self.hidden_dim)

    def cost_params(self) -> int:
        self._require_fitted()
        return int(self.hidden_dim * self.feature_dim + (self.feature_dim + 1) * self.hidden_dim)
