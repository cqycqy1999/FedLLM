from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from fedpost.analysis.finding2.activation import resolve_transformer_blocks


@dataclass
class CaptureConfig:
    layer_indices: list[int] | None = None
    to_cpu: bool = True
    detach: bool = True


class CaptureHiddenStates:
    """Capture residual stream tensors by layer index.

    Layer index 0 is the token embedding output. Layer index k > 0 is the
    output of transformer block k. This convention matches the existing
    residual-stream collector.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_indices: list[int] | None = None,
        to_cpu: bool = True,
        detach: bool = True,
    ):
        self.model = model
        self.cfg = CaptureConfig(layer_indices=layer_indices, to_cpu=to_cpu, detach=detach)
        self.records: dict[int, torch.Tensor] = {}
        self._handles = []
        self._target_layers = set(layer_indices) if layer_indices is not None else None

    def __enter__(self):
        self.clear()
        self._register()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def clear(self) -> None:
        self.records = {}

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def snapshot(self) -> dict[int, torch.Tensor]:
        return dict(self.records)

    def _register(self) -> None:
        if self._should_capture(0):
            embedding = self.model.get_input_embeddings()
            self._handles.append(
                embedding.register_forward_hook(
                    lambda _module, _inputs, output: self._store(0, output)
                )
            )

        for idx, (_name, block) in enumerate(resolve_transformer_blocks(self.model), start=1):
            if not self._should_capture(idx):
                continue

            def hook(_module, _inputs, output, layer_idx=idx):
                self._store(layer_idx, output)

            self._handles.append(block.register_forward_hook(hook))

    def _should_capture(self, layer_idx: int) -> bool:
        return self._target_layers is None or layer_idx in self._target_layers

    def _store(self, layer_idx: int, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if self.cfg.detach:
            hidden = hidden.detach()
        if self.cfg.to_cpu:
            hidden = hidden.cpu()
        self.records[layer_idx] = hidden
