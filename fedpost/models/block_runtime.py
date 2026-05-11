from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from fedpost.bpfedpeft.planner import BlockSpec


@dataclass(frozen=True)
class LayerHandle:
    parent: Any
    attr: str
    layers: Any


def unwrap_causal_lm(model):
    if hasattr(model, "get_base_model"):
        return model.get_base_model()
    return model


def infer_num_decoder_layers(model) -> int:
    return len(locate_decoder_layers(unwrap_causal_lm(model)).layers)


def locate_decoder_layers(causal_lm) -> LayerHandle:
    candidates = (
        ("model", "layers"),
        ("transformer", "h"),
        ("gpt_neox", "layers"),
        ("model.decoder", "layers"),
        ("decoder", "layers"),
    )
    for parent_path, attr in candidates:
        parent = _resolve_path(causal_lm, parent_path)
        if parent is not None and hasattr(parent, attr):
            layers = getattr(parent, attr)
            if hasattr(layers, "__len__") and len(layers) > 0:
                return LayerHandle(parent=parent, attr=attr, layers=layers)
    raise ValueError(
        "Could not locate decoder layers. Supported paths include "
        "model.layers, transformer.h, gpt_neox.layers, and decoder.layers."
    )


def locate_final_norm(causal_lm):
    for path in (
        "model.norm",
        "transformer.ln_f",
        "gpt_neox.final_layer_norm",
        "model.decoder.final_layer_norm",
        "decoder.final_layer_norm",
    ):
        module = _resolve_path(causal_lm, path)
        if module is not None:
            return module
    return None


def locate_lm_head(causal_lm):
    if hasattr(causal_lm, "get_output_embeddings"):
        output = causal_lm.get_output_embeddings()
        if output is not None:
            return output
    if hasattr(causal_lm, "lm_head"):
        return causal_lm.lm_head
    if hasattr(causal_lm, "embed_out"):
        return causal_lm.embed_out
    raise ValueError("Could not locate a language-model output head.")


@contextmanager
def active_decoder_layers(model, block: BlockSpec):
    causal_lm = unwrap_causal_lm(model)
    handle = locate_decoder_layers(causal_lm)
    original_layers = handle.layers
    subset = original_layers[block.start : block.end + 1]
    replacement = torch.nn.ModuleList(list(subset))
    setattr(handle.parent, handle.attr, replacement)
    try:
        yield
    finally:
        setattr(handle.parent, handle.attr, original_layers)


def forward_block_causal_lm(
    model,
    input_ids,
    attention_mask=None,
    labels=None,
    block: BlockSpec | None = None,
    identity_vector=None,
    residual_vector=None,
    ignore_index: int = -100,
):
    """Run a decoder-only LM on the active block with depth vectors."""

    if block is None:
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    causal_lm = unwrap_causal_lm(model)
    embeddings = causal_lm.get_input_embeddings()(input_ids)
    embeddings = _add_depth_vector(embeddings, identity_vector)

    with active_decoder_layers(model, block):
        outputs = model(
            input_ids=None,
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    hidden_states = outputs.hidden_states[-1]
    if residual_vector is None:
        logits = outputs.logits
    else:
        hidden_states = _add_depth_vector(hidden_states, residual_vector)
        final_norm = locate_final_norm(causal_lm)
        if final_norm is not None:
            hidden_states = final_norm(hidden_states)
        logits = locate_lm_head(causal_lm)(hidden_states)

    loss = None
    if labels is not None:
        loss = causal_lm_loss(logits, labels, ignore_index=ignore_index)

    return {
        "loss": loss,
        "logits": logits,
        "hidden_states": outputs.hidden_states,
    }


def causal_lm_loss(logits, labels, ignore_index: int = -100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def _add_depth_vector(hidden_states, vector):
    if vector is None:
        return hidden_states
    vector = vector.to(device=hidden_states.device, dtype=hidden_states.dtype)
    while vector.ndim < hidden_states.ndim:
        vector = vector.unsqueeze(0)
    return hidden_states + vector


def _resolve_path(obj, path: str):
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current
