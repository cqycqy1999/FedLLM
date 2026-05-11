from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fedpost.algorithms.base import FederatedAlgorithm
from fedpost.algorithms.fedavg import FedAvgAggregator
from fedpost.bpfedpeft.planner import (
    BlockSpec,
    blocks_from_end_layers,
    plan_equal_blocks,
)
from fedpost.bpfedpeft.state import state_for_block
from fedpost.models.block_runtime import infer_num_decoder_layers
from fedpost.models.block_vectors import load_block_depth_vectors
from fedpost.utils.registry import Registry


class BPFedPEFTAggregator(FedAvgAggregator):
    pass


@Registry.register("algorithm", "bpfedpeft")
class BPFedPEFTAlgorithm(FederatedAlgorithm):
    """Block-wise progressive FedPEFT scheduler."""

    aggregator_cls = BPFedPEFTAggregator

    def __init__(self, cfg, aggregator):
        super().__init__(cfg, aggregator)
        self.blocks: list[BlockSpec] | None = None
        self.depth_vectors = load_block_depth_vectors(getattr(cfg.bpfedpeft, "vector_path", None))

        self.phase = "anchor"
        self.block_idx: int | None = None
        self.rounds_in_block = 0
        self.completed = False
        self.prev_global_ema: float | None = None
        self.global_ema = 0.0

    def before_broadcast(self, server, clients, round_idx: int):
        self._ensure_plan(server)
        if self.block_idx is None:
            self.block_idx = len(self.blocks) - 1

    def make_broadcast_payload(self, server, round_idx: int):
        payload = server.get_broadcast_payload()
        block = self._active_block()
        payload.model_state = state_for_block(
            payload.model_state,
            block,
            include_unindexed=self.cfg.bpfedpeft.include_unindexed_parameters,
        )
        payload.algo_state = {
            "algorithm": "bpfedpeft",
            "phase": self.phase,
            "block_idx": self.block_idx,
            "block": asdict(block),
            "rounds_in_block": self.rounds_in_block,
            "local_stability_threshold": self.cfg.bpfedpeft.local_stability_threshold,
            "min_local_steps": self.cfg.bpfedpeft.min_local_steps,
            "use_block_forward": self.cfg.bpfedpeft.use_block_forward,
            **self._vector_payload(self.block_idx),
        }
        payload.metadata["blocks"] = [block.to_dict() for block in self.blocks]
        return payload

    def server_update(self, server, results):
        metrics = server.apply_updates(results)
        metrics.update(self._summarize_active_block(results))
        self._advance_schedule(metrics)
        return metrics

    def should_stop_training(self) -> bool:
        return self.completed

    def _ensure_plan(self, server) -> None:
        if self.blocks is not None:
            return

        cfg = self.cfg.bpfedpeft
        if cfg.block_end_layers:
            self.blocks = blocks_from_end_layers(
                cfg.block_end_layers,
                overlap_layers=cfg.overlap_layers,
                one_based=True,
            )
            return

        num_layers = infer_num_decoder_layers(server.model_bundle.model)
        num_blocks = cfg.num_blocks or min(4, num_layers)
        self.blocks = plan_equal_blocks(
            num_layers=num_layers,
            num_blocks=num_blocks,
            overlap_layers=cfg.overlap_layers,
        )

    def _active_block(self) -> BlockSpec:
        if self.blocks is None or self.block_idx is None:
            raise RuntimeError("BP-FedPEFT block plan has not been initialized")
        return self.blocks[self.block_idx]

    def _vector_payload(self, block_idx: int | None) -> dict[str, Any]:
        if block_idx is None or self.depth_vectors is None:
            return {"identity_vector": None, "residual_vector": None}
        return self.depth_vectors.for_block(block_idx)

    def _summarize_active_block(self, results) -> dict[str, Any]:
        valid_results = [result for result in results if result.success]
        if not valid_results:
            return {}

        total = sum(result.num_train_samples for result in valid_results)
        weighted_score = 0.0
        local_stability = 0.0
        for result in valid_results:
            weight = result.num_train_samples / total
            weighted_score += weight * float(result.metrics.get("update_significance", 0.0))
            local_stability += weight * float(result.metrics.get("local_stability", 0.0))

        block = self._active_block()
        return {
            "bpfedpeft/phase": self.phase,
            "bpfedpeft/block_idx": self.block_idx,
            "bpfedpeft/block_start": block.start,
            "bpfedpeft/block_end": block.end,
            "bpfedpeft/global_update_significance": weighted_score,
            "bpfedpeft/weighted_local_stability": local_stability,
            "bpfedpeft/rounds_in_block": self.rounds_in_block + 1,
        }

    def _advance_schedule(self, metrics: dict[str, Any]) -> None:
        self.rounds_in_block += 1

        if self.phase == "anchor":
            if self.rounds_in_block >= self.cfg.bpfedpeft.anchoring_rounds_per_block:
                self._advance_anchor_block()
            return

        score = float(metrics.get("bpfedpeft/global_update_significance", 0.0))
        alpha = self.cfg.bpfedpeft.global_stability_alpha
        self.prev_global_ema = self.global_ema
        self.global_ema = alpha * self.global_ema + (1.0 - alpha) * score
        delta = abs(self.global_ema - (self.prev_global_ema or 0.0))

        min_rounds = self.cfg.bpfedpeft.min_rounds_per_block
        max_rounds = self.cfg.bpfedpeft.max_rounds_per_block
        stable = self.rounds_in_block >= min_rounds and delta <= self.cfg.bpfedpeft.global_stability_delta
        exhausted = self.rounds_in_block >= max_rounds
        if stable or exhausted:
            self._advance_adaptation_block()

    def _advance_anchor_block(self) -> None:
        self.rounds_in_block = 0
        if self.block_idx is None:
            raise RuntimeError("block_idx is not initialized")
        if self.block_idx > 0:
            self.block_idx -= 1
            return

        self.phase = "adapt"
        self.block_idx = 0
        self.prev_global_ema = None
        self.global_ema = 0.0

    def _advance_adaptation_block(self) -> None:
        self.rounds_in_block = 0
        self.prev_global_ema = None
        self.global_ema = 0.0
        if self.block_idx is None or self.blocks is None:
            raise RuntimeError("block schedule is not initialized")
        if self.block_idx + 1 < len(self.blocks):
            self.block_idx += 1
        else:
            self.completed = True
