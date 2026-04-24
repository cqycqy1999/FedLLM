from __future__ import annotations

import traceback

from fedpost.federation.message import TrainResult


class Client:
    def __init__(self, context, trainer, dataset, logger=None):
        self.context = context
        self.client_id = context.client_id
        self.trainer = trainer
        self.dataset = dataset
        self.logger = logger

        self._round_idx = 0
        self._algo_state = {}

    def receive_broadcast(self, payload):
        self._round_idx = payload.round_idx
        self._algo_state = payload.algo_state
        self.trainer.set_trainable_state(payload.model_state)

    def run_round(self, payload, device=None) -> TrainResult:
        result = None
        activated = False
        try:
            self.trainer.activate_device(device)
            activated = True
            self.receive_broadcast(payload)
            result = self.local_train()
        except Exception as exc:
            result = self._failed_result(exc)
        finally:
            if activated:
                try:
                    self.trainer.release_device()
                except Exception as exc:
                    if result is None or result.success:
                        result = self._failed_result(exc)
                    else:
                        release_traceback = traceback.format_exc()
                        release_error = f"Device release failed after training error: {exc}"
                        result.error_msg = (
                            f"{result.error_msg}\n{release_error}"
                            if result.error_msg
                            else release_error
                        )
                        result.error_traceback = (
                            f"{result.error_traceback}\n\nDuring device release:\n{release_traceback}"
                            if result.error_traceback
                            else release_traceback
                        )
        return result

    def local_train(self) -> TrainResult:
        try:
            update, metrics = self.trainer.train_one_round(
                dataset=self.dataset,
                round_idx=self._round_idx,
            )
            return TrainResult(
                client_id=self.client_id,
                round_idx=self._round_idx,
                num_train_samples=self.context.num_samples,
                update=update,
                metrics=metrics,
                success=True,
            )
        except Exception as exc:
            return self._failed_result(exc)

    def _failed_result(self, exc: Exception) -> TrainResult:
        return TrainResult(
            client_id=self.client_id,
            round_idx=self._round_idx,
            num_train_samples=self.context.num_samples,
            update={},
            metrics={},
            success=False,
            error_msg=str(exc),
            error_traceback=traceback.format_exc(),
        )
