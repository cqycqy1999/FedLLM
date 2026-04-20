from __future__ import annotations

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
        except Exception as e:
            return TrainResult(
                client_id=self.client_id,
                round_idx=self._round_idx,
                num_train_samples=self.context.num_samples,
                update={},
                metrics={},
                success=False,
                error_msg=str(e),
            )