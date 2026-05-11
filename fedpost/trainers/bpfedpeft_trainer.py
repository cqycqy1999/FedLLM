from __future__ import annotations

from fedpost.bpfedpeft.planner import BlockSpec
from fedpost.bpfedpeft.state import (
    set_active_peft_block,
    state_delta_norm,
    state_for_block,
)
from fedpost.models.block_runtime import forward_block_causal_lm
from fedpost.trainers.base_trainer import BaseTrainer
from fedpost.utils.registry import Registry


@Registry.register("trainer", "bpfedpeft_sft")
class BPFedPEFTSFTTrainer(BaseTrainer):
    def __init__(self, cfg, model_bundle, model_manager, collator, logger=None):
        super().__init__(cfg, model_bundle, model_manager, collator, logger)
        self.algo_state = {}
        self.active_block: BlockSpec | None = None
        self.identity_vector = None
        self.residual_vector = None
        self._round_start_state = {}

    def set_algorithm_state(self, algo_state: dict) -> None:
        self.algo_state = dict(algo_state or {})
        block = self.algo_state.get("block")
        self.active_block = BlockSpec(**block) if block else None
        self.identity_vector = self.algo_state.get("identity_vector")
        self.residual_vector = self.algo_state.get("residual_vector")
        self._activate_block()

    def set_trainable_state(self, state: dict) -> None:
        super().set_trainable_state(state)
        self._activate_block()
        if self.optimizer is not None:
            self.optimizer = self.build_optimizer()

    def get_trainable_state(self) -> dict:
        state = super().get_trainable_state()
        if self.active_block is None:
            return state
        return state_for_block(
            state,
            self.active_block,
            include_unindexed=self.cfg.bpfedpeft.include_unindexed_parameters,
        )

    def train_one_round(self, dataset, round_idx: int):
        self._round_start_state = self.get_trainable_state()
        return self._run_local_training(dataset, round_idx)

    def compute_loss(self, batch: dict):
        batch = self._move_batch_to_device(batch)
        if self.algo_state.get("use_block_forward", True):
            outputs = forward_block_causal_lm(
                self.model,
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
                block=self.active_block,
                identity_vector=self.identity_vector,
                residual_vector=self.residual_vector,
            )
            loss = outputs["loss"]
        else:
            outputs = self.model(**batch)
            loss = outputs.loss
        return loss, {"loss": float(loss.detach().cpu())}

    def _run_local_training(self, dataset, round_idx: int):
        self.model.train()

        dataloader = self.build_dataloader(dataset)
        total_update_steps = self._estimate_total_update_steps(len(dataloader))
        self.lr_scheduler = self.build_lr_scheduler(total_update_steps)

        train_cfg = self._train_cfg()
        grad_accum_steps = train_cfg.grad_accum_steps
        local_steps = self._local_steps()
        metrics_list = []
        micro_step = 0
        optimizer_steps = 0
        local_stability = 0.0
        previous_state = self.get_trainable_state()

        self.optimizer.zero_grad()
        stop_training = False

        for _ in range(self._local_epochs()):
            for batch in dataloader:
                loss, metrics = self.compute_loss(batch)
                (loss / grad_accum_steps).backward()
                metrics_list.append(metrics)
                micro_step += 1

                should_step = micro_step % grad_accum_steps == 0
                if should_step:
                    self._optimizer_step()
                    optimizer_steps += 1
                    local_stability, previous_state = self._update_local_stability(previous_state)
                    if self._should_stop_local(optimizer_steps, local_steps, local_stability):
                        stop_training = True
                        break

            if stop_training:
                break

        if micro_step > 0 and micro_step % grad_accum_steps != 0:
            if local_steps is None or optimizer_steps < local_steps:
                self._optimizer_step()
                optimizer_steps += 1
                local_stability, previous_state = self._update_local_stability(previous_state)

        update = self.get_trainable_state()
        _, update_significance = state_delta_norm(self._round_start_state, update)
        metrics = self._aggregate_metrics(metrics_list)
        metrics.update({
            "optimizer_steps": float(optimizer_steps),
            "local_epochs": float(self._local_epochs()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "local_stability": float(local_stability),
            "update_significance": float(update_significance),
            "bpfedpeft/block_idx": float(self.algo_state.get("block_idx", -1)),
            "bpfedpeft/is_anchor": float(self.algo_state.get("phase") == "anchor"),
        })
        return update, metrics

    def _activate_block(self) -> None:
        if self.active_block is None:
            return
        set_active_peft_block(
            self.model,
            self.active_block,
            include_unindexed=self.cfg.bpfedpeft.include_unindexed_parameters,
        )

    def _update_local_stability(self, previous_state: dict) -> tuple[float, dict]:
        current_state = self.get_trainable_state()
        _, rel_delta = state_delta_norm(previous_state, current_state)
        stability = 1.0 / (1.0 + rel_delta)
        return stability, current_state

    def _should_stop_local(
        self,
        optimizer_steps: int,
        local_steps: int | None,
        local_stability: float,
    ) -> bool:
        if local_steps is not None and optimizer_steps >= local_steps:
            return True
        min_steps = int(self.algo_state.get("min_local_steps", 1))
        threshold = float(self.algo_state.get("local_stability_threshold", 1.1))
        return optimizer_steps >= min_steps and local_stability >= threshold

    def _local_epochs(self) -> int:
        if self.algo_state.get("phase") == "anchor":
            return self.cfg.bpfedpeft.anchoring_local_epochs
        return self.cfg.federated.local_epochs

    def _optimizer_step(self) -> None:
        if self.optimizer is None:
            raise RuntimeError("optimizer has not been initialized")
        super()._optimizer_step()
