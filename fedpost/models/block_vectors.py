from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from fedpost.bpfedpeft.planner import BlockSpec


@dataclass
class BlockDepthVectors:
    identity: dict[int, torch.Tensor]
    residual: dict[int, torch.Tensor]

    def for_block(self, block_idx: int) -> dict[str, torch.Tensor | None]:
        return {
            "identity_vector": self.identity.get(block_idx),
            "residual_vector": self.residual.get(block_idx),
        }


def load_block_depth_vectors(path: str | None) -> BlockDepthVectors | None:
    if not path:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"BP-FedPEFT vector file not found: {path}")

    payload = torch.load(path, map_location="cpu")
    return BlockDepthVectors(
        identity={int(k): v.detach().cpu() for k, v in payload.get("identity", {}).items()},
        residual={int(k): v.detach().cpu() for k, v in payload.get("residual", {}).items()},
    )


@torch.no_grad()
def compute_block_depth_vectors(
    model,
    dataloader,
    blocks: list[BlockSpec],
    max_batches: int | None = None,
) -> BlockDepthVectors:
    """Estimate Eq. (4) identity vectors and final residual guidance vectors."""

    device = next(model.parameters()).device
    identity_sums: dict[int, torch.Tensor] = {}
    residual_sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}

    model.eval()
    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = outputs.hidden_states
        embeddings = hidden_states[0]
        final_hidden = hidden_states[-1]

        for block_idx, block in enumerate(blocks):
            block_input = hidden_states[block.start]
            block_output = hidden_states[block.end + 1]
            identity_vec = (block_input - embeddings).mean(dim=(0, 1)).detach().cpu()
            residual_vec = (final_hidden - block_output).mean(dim=(0, 1)).detach().cpu()

            if block_idx not in identity_sums:
                identity_sums[block_idx] = identity_vec
                residual_sums[block_idx] = residual_vec
                counts[block_idx] = 1
            else:
                identity_sums[block_idx] += identity_vec
                residual_sums[block_idx] += residual_vec
                counts[block_idx] += 1

    if not counts:
        raise ValueError("proxy dataloader produced no batches")

    identity = {idx: identity_sums[idx] / counts[idx] for idx in counts}
    residual = {idx: residual_sums[idx] / counts[idx] for idx in counts}
    return BlockDepthVectors(identity=identity, residual=residual)


def save_block_depth_vectors(vectors: BlockDepthVectors, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"identity": vectors.identity, "residual": vectors.residual}, path)
    return path
