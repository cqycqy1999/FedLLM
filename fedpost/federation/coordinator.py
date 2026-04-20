from __future__ import annotations

import os


class Coordinator:
    def __init__(
        self,
        cfg,
        server,
        clients,
        sampler,
        algorithm,
        evaluator=None,
        recorder=None,
        logger=None,
    ):
        self.cfg = cfg
        self.server = server
        self.clients = clients
        self.sampler = sampler
        self.algorithm = algorithm
        self.evaluator = evaluator
        self.recorder = recorder
        self.logger = logger

    def train(self):
        all_round_metrics = []

        for round_idx in range(self.cfg.federated.rounds):
            round_metrics = self.run_round(round_idx)
            all_round_metrics.append(round_metrics)

            if self._should_eval(round_idx):
                eval_result = self.evaluate(round_idx)
                if self.recorder is not None:
                    self.recorder.record_eval(eval_result)

            if self._should_save(round_idx):
                ckpt_path = self._ckpt_path(round_idx)
                self.server.save_checkpoint(ckpt_path)

        return all_round_metrics

    def run_round(self, round_idx: int):
        self.server.round_idx = round_idx
        selected_clients = self.sampler.sample(self.clients, round_idx)

        self.algorithm.before_broadcast(self.server, selected_clients, round_idx)
        payload = self.algorithm.make_broadcast_payload(self.server, round_idx)

        results = []
        for client in selected_clients:
            client.receive_broadcast(payload)
            result = client.local_train()
            results.append(result)

        results = self.algorithm.after_local_train(results, round_idx)
        agg_metrics = self.algorithm.server_update(self.server, results)

        round_metrics = self._summarize_round(results, agg_metrics)
        if self.recorder is not None:
            self.recorder.record_round(round_idx, round_metrics, results)
        return round_metrics

    def evaluate(self, round_idx: int):
        model = self.server.evaluate_model()
        return self.evaluator.evaluate(model, round_idx=round_idx)

    def _summarize_round(self, results, agg_metrics):
        success_results = [r for r in results if r.success]
        avg_loss = 0.0
        if success_results:
            losses = [r.metrics.get("loss", 0.0) for r in success_results]
            avg_loss = sum(losses) / len(losses)

        return {
            "avg_client_loss": avg_loss,
            "num_selected_clients": len(results),
            **agg_metrics,
        }

    def _should_eval(self, round_idx: int) -> bool:
        return (round_idx + 1) % self.cfg.eval.eval_every == 0

    def _should_save(self, round_idx: int) -> bool:
        return (round_idx + 1) % self.cfg.eval.save_every == 0

    def _ckpt_path(self, round_idx: int) -> str:
        return os.path.join(self.cfg.output_dir, "checkpoints", f"round_{round_idx+1}.pt")