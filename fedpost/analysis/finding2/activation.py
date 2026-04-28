from __future__ import annotations

from typing import Any

import torch


BLOCK_PATHS = (
    "model.layers",
    "transformer.h",
    "gpt_neox.layers",
    "transformer.blocks",
    "decoder.layers",
    "base_model.model.model.layers",
    "base_model.model.transformer.h",
    "base_model.model.gpt_neox.layers",
    "base_model.model.transformer.blocks",
    "base_model.model.decoder.layers",
)


class ActivationSubstitution:
    """Replace a residual-stream activation at a block boundary.

    `boundary_layer_idx` follows the hidden-state convention:
    0 is embedding output, k > 0 is the input to transformer block k.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        boundary_layer_idx: int,
        replacement: torch.Tensor,
        detach: bool = True,
    ):
        self.model = model
        self.boundary_layer_idx = boundary_layer_idx
        self.replacement = replacement.detach() if detach else replacement
        self.detach = detach
        self._handle = None

    def __enter__(self):
        if self.boundary_layer_idx < 0:
            raise ValueError("boundary_layer_idx must be non-negative")
        if self.boundary_layer_idx == 0:
            embedding = self.model.get_input_embeddings()

            def embedding_hook(_module, _inputs, output):
                return self._replacement_like(output)

            self._handle = embedding.register_forward_hook(embedding_hook)
            return self

        blocks = resolve_transformer_blocks(self.model)
        block_idx = self.boundary_layer_idx
        if block_idx >= len(blocks):
            raise ValueError(
                f"boundary_layer_idx={self.boundary_layer_idx} exceeds available blocks={len(blocks)}"
            )
        _name, block = blocks[block_idx]

        def pre_hook(_module, inputs):
            if not inputs:
                raise RuntimeError("Cannot substitute activation because block received no positional inputs.")
            hidden = inputs[0]
            return (self._replacement_like(hidden), *inputs[1:])

        self._handle = block.register_forward_pre_hook(pre_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _replacement_like(self, hidden: torch.Tensor) -> torch.Tensor:
        replacement = self.replacement.to(device=hidden.device, dtype=hidden.dtype)
        if tuple(replacement.shape) != tuple(hidden.shape):
            raise RuntimeError(
                f"Replacement shape {tuple(replacement.shape)} does not match activation shape {tuple(hidden.shape)}"
            )
        return replacement


def resolve_transformer_blocks(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    for path in BLOCK_PATHS:
        module = get_submodule(model, path)
        if isinstance(module, (torch.nn.ModuleList, list, tuple)):
            return [(f"{path}.{idx}", block) for idx, block in enumerate(module)]
    raise RuntimeError("Could not resolve transformer blocks.")


def get_submodule(model: torch.nn.Module, path: str) -> Any:
    current = model
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
