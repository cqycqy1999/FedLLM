from __future__ import annotations

from typing import Any
import math

import torch
from torch.utils.data import DataLoader


class BaseTrainer:
    def __init__(self, cfg, model_bundle, model_manager, collator, logger=None):
        self.cfg = cfg
        self.model = model_bundle.model
        self.tokenizer = model_bundle.tokenizer
        self.reference_model = model_bundle.reference_model
        self.model_state_spec = model_bundle.model_state_spec
        self.model_manager = model_manager
        self.collator = collator
        self.logger = logger

        # Keep inactive client models on CPU; each round moves active clients onto a target device.
        self.device = torch.device("cpu")
        self.model.to(self.device)
        if self.reference_model is not None:
            self.reference_model.to(self.device)

        self.optimizer = None
        self.lr_scheduler = None

    def build_optimizer(self):
        train_cfg = self._train_cfg()
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        lr = self._learning_rate()

        if train_cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                trainable_params,
                lr=lr,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                eps=train_cfg.adam_epsilon,
                weight_decay=train_cfg.weight_decay,
            )
        if train_cfg.optimizer == "adam":
            return torch.optim.Adam(
                trainable_params,
                lr=lr,
                betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
                eps=train_cfg.adam_epsilon,
                weight_decay=train_cfg.weight_decay,
            )
        if train_cfg.optimizer == "sgd":
            return torch.optim.SGD(
                trainable_params,
                lr=lr,
                weight_decay=train_cfg.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer: {train_cfg.optimizer}")

    def activate_device(self, device: str | torch.device | None = None) -> None:
        target_device = self._resolve_device(device)
        if target_device.type == "cuda":
            torch.cuda.set_device(target_device)

        self.model.to(target_device)
        if self.reference_model is not None:
            self.reference_model.to(target_device)

        self.device = target_device
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = None

    def release_device(self) -> None:
        self.optimizer = None
        self.lr_scheduler = None
        if self.device.type != "cuda":
            return

        self.model.to("cpu")
        if self.reference_model is not None:
            self.reference_model.to("cpu")
        self.device = torch.device("cpu")
        torch.cuda.empty_cache()

    def build_dataloader(self, dataset):
        batch_size = self._train_cfg().batch_size
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.collator,
        )

    def set_trainable_state(self, state: dict[str, Any]) -> None:
        self.model_manager.load_trainable_state(self.model, state)

    def get_trainable_state(self) -> dict[str, Any]:
        return self.model_manager.get_trainable_state(self.model)

    def train_one_round(self, dataset, round_idx: int):
        return self._run_local_training(dataset, round_idx)

    def compute_loss(self, batch: dict):
        raise NotImplementedError

    def _move_batch_to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if hasattr(v, "to"):
                out[k] = v.to(self.device)
            else:
                out[k] = v
        return out

    def _aggregate_metrics(self, metrics_list: list[dict]) -> dict:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: sum(m[k] for m in metrics_list) / len(metrics_list) for k in keys}

    def _reach_local_budget(self, step_idx: int) -> bool:
        local_steps = self.cfg.federated.local_steps
        if local_steps is None:
            return False
        return (step_idx + 1) >= local_steps

    def _run_local_training(self, dataset, round_idx: int):
        self.model.train()
        if self.reference_model is not None:
            self.reference_model.eval()

        dataloader = self.build_dataloader(dataset)
        total_update_steps = self._estimate_total_update_steps(len(dataloader))
        self.lr_scheduler = self.build_lr_scheduler(total_update_steps)

        train_cfg = self._train_cfg()
        grad_accum_steps = train_cfg.grad_accum_steps
        local_steps = self._local_steps()
        metrics_list = []
        micro_step = 0
        optimizer_steps = 0

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
                    if local_steps is not None and optimizer_steps >= local_steps:
                        stop_training = True
                        break

            if stop_training:
                break

        if micro_step > 0 and micro_step % grad_accum_steps != 0:
            if local_steps is None or optimizer_steps < local_steps:
                self._optimizer_step()
                optimizer_steps += 1

        update = self.get_trainable_state()
        metrics = self._aggregate_metrics(metrics_list)
        metrics.update({
            "optimizer_steps": float(optimizer_steps),
            "local_epochs": float(self._local_epochs()),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        })
        return update, metrics

    def _optimizer_step(self) -> None:
        max_grad_norm = self._train_cfg().max_grad_norm
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_grad_norm,
            )
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def build_lr_scheduler(self, total_update_steps: int):
        train_cfg = self._train_cfg()
        scheduler_type = train_cfg.lr_scheduler
        if scheduler_type == "constant":
            return None

        total_update_steps = max(1, total_update_steps)
        warmup_steps = train_cfg.warmup_steps
        if warmup_steps == 0 and train_cfg.warmup_ratio > 0:
            warmup_steps = int(total_update_steps * train_cfg.warmup_ratio)

        if scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=train_cfg.learning_rate_decay,
            )

        final_lr_factor = train_cfg.learning_rate_decay

        def lr_lambda(current_step: int) -> float:
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))

            progress = (current_step - warmup_steps) / float(max(1, total_update_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)

            if scheduler_type == "linear":
                return final_lr_factor + (1.0 - final_lr_factor) * (1.0 - progress)
            if scheduler_type == "cosine":
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                return final_lr_factor + (1.0 - final_lr_factor) * cosine_factor
            raise ValueError(f"Unsupported lr_scheduler: {scheduler_type}")

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _estimate_total_update_steps(self, num_batches: int) -> int:
        total_micro_steps = num_batches * self._local_epochs()
        total_update_steps = math.ceil(total_micro_steps / self._train_cfg().grad_accum_steps)
        local_steps = self._local_steps()
        if local_steps is not None:
            total_update_steps = min(total_update_steps, local_steps)
        return max(1, total_update_steps)

    def _train_cfg(self):
        return self.cfg.sft if self.cfg.task == "sft" else self.cfg.dpo

    def _learning_rate(self) -> float:
        train_cfg = self._train_cfg()
        return train_cfg.learning_rate if train_cfg.learning_rate is not None else train_cfg.lr

    def _local_epochs(self) -> int:
        return self.cfg.federated.local_epochs

    def _local_steps(self) -> int | None:
        return self.cfg.federated.local_steps

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
