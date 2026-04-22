from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import torch

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

            # if self._should_eval(round_idx):
            #     # eval_result = self.evaluate(round_idx)
            #     # if self.recorder is not None:
            #     #     self.recorder.record_eval(eval_result)
            #     ckpt_path = self._ckpt_path(round_idx)
            #     self.server.save_checkpoint(ckpt_path)

            if self._should_save(round_idx):
                ckpt_path = self._ckpt_path(round_idx)
                self.server.save_checkpoint(ckpt_path)

        return all_round_metrics

    def run_round(self, round_idx: int):
        self.server.round_idx = round_idx
        selected_clients = self.sampler.sample(self.clients, round_idx)

        self.algorithm.before_broadcast(self.server, selected_clients, round_idx)
        payload = self.algorithm.make_broadcast_payload(self.server, round_idx)

        devices = self._training_devices()
        max_parallel = self._max_parallel_clients(len(devices))
        results = []
        for start_idx in range(0, len(selected_clients), max_parallel):
            batch_clients = selected_clients[start_idx:start_idx + max_parallel]
            batch_devices = devices[:len(batch_clients)]

            with ThreadPoolExecutor(max_workers=len(batch_clients)) as executor:
                futures = [
                    executor.submit(client.run_round, payload, device)
                    for client, device in zip(batch_clients, batch_devices)
                ]
                results.extend(future.result() for future in futures)

        results = self.algorithm.after_local_train(results, round_idx)
        agg_metrics = self.algorithm.server_update(self.server, results)

        export_artifacts = self.server.export_round_artifacts(round_idx)

        round_metrics = self._summarize_round(results, agg_metrics)
        round_metrics.update({
            "export_adapter_dir": export_artifacts.get("adapter_dir", ""),
            "export_merged_model_dir": export_artifacts.get("merged_model_dir", ""),
        })

        if self._should_eval(round_idx):
            eval_result = self.evaluate(round_idx, model_artifacts=export_artifacts)
            if self.recorder is not None:
                self.recorder.record_eval(eval_result)

        if self.recorder is not None:
            self.recorder.record_round(round_idx, round_metrics, results)

        return round_metrics
        # round_metrics = self._summarize_round(results, agg_metrics)
        # if self.recorder is not None:
        #     self.recorder.record_round(round_idx, round_metrics, results)
        # return round_metrics

    def evaluate(self, round_idx: int, model_artifacts: dict | None = None):
        model = self.server.evaluate_model()
        return self.evaluator.evaluate(
            model,
            round_idx=round_idx,
            model_artifacts=model_artifacts or {},
        )
        
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

    def _training_devices(self) -> list[str]:
        gpu_ids = self.cfg.federated.gpu_ids
        if gpu_ids:
            if not torch.cuda.is_available():
                raise RuntimeError("gpu_ids were configured but CUDA is not available")

            device_count = torch.cuda.device_count()
            invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= device_count]
            if invalid_gpu_ids:
                raise ValueError(
                    f"Configured gpu_ids {invalid_gpu_ids} exceed available CUDA devices ({device_count})"
                )
            return [f"cuda:{gpu_id}" for gpu_id in gpu_ids]

        if torch.cuda.is_available():
            return ["cuda:0"]
        return ["cpu"]

    def _max_parallel_clients(self, num_devices: int) -> int:
        configured_limit = self.cfg.federated.max_parallel_clients
        if configured_limit is None:
            return max(1, min(num_devices, self.cfg.federated.clients_per_round))
        return max(1, min(configured_limit, num_devices, self.cfg.federated.clients_per_round))
