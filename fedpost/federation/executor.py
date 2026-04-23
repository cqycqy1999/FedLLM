from __future__ import annotations

import traceback
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch.multiprocessing as mp

from fedpost.federation.message import TrainResult


def _run_client_task(client, payload, device) -> TrainResult:
    try:
        return client.run_round(payload, device)
    except Exception as exc:
        client_id = getattr(client, "client_id", "unknown")
        round_idx = getattr(payload, "round_idx", -1)
        num_samples = getattr(getattr(client, "context", None), "num_samples", 0)
        return TrainResult(
            client_id=client_id,
            round_idx=round_idx,
            num_train_samples=num_samples,
            update={},
            metrics={},
            success=False,
            error_msg=f"{exc}\n{traceback.format_exc()}",
        )


def _worker_loop(device, task_queue, result_queue) -> None:
    while True:
        task = task_queue.get()
        if task is None:
            break

        task_id, client, payload = task
        result = _run_client_task(client, payload, device)
        result_queue.put((task_id, result))


class ClientExecutor:
    def run_batch(self, clients: Sequence, payload, devices: Sequence[str]) -> list[TrainResult]:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass


class ThreadClientExecutor(ClientExecutor):
    def run_batch(self, clients: Sequence, payload, devices: Sequence[str]) -> list[TrainResult]:
        with ThreadPoolExecutor(max_workers=len(clients)) as executor:
            futures = [
                executor.submit(client.run_round, payload, device)
                for client, device in zip(clients, devices)
            ]
            return [future.result() for future in futures]


class MultiprocessingClientExecutor(ClientExecutor):
    def __init__(self, start_method: str = "spawn"):
        self.ctx = mp.get_context(start_method)
        self.task_queues = {}
        self.result_queue = self.ctx.Queue()
        self.processes = {}

    def _ensure_worker(self, device: str) -> None:
        process = self.processes.get(device)
        if process is not None and process.is_alive():
            return

        task_queue = self.ctx.Queue()
        process = self.ctx.Process(
            target=_worker_loop,
            args=(device, task_queue, self.result_queue),
        )
        process.start()
        self.task_queues[device] = task_queue
        self.processes[device] = process

    def run_batch(self, clients: Sequence, payload, devices: Sequence[str]) -> list[TrainResult]:
        task_ids = []
        task_devices = {}

        for idx, (client, device) in enumerate(zip(clients, devices)):
            self._ensure_worker(device)
            task_id = (getattr(payload, "round_idx", -1), idx, getattr(client, "client_id", str(idx)))
            task_ids.append(task_id)
            task_devices[task_id] = device
            self.task_queues[device].put((task_id, client, payload))

        pending = set(task_ids)
        results_by_id = {}
        while pending:
            try:
                task_id, result = self.result_queue.get(timeout=1.0)
            except queue.Empty:
                self._raise_if_worker_died(pending, task_devices)
                continue
            if task_id in pending:
                results_by_id[task_id] = result
                pending.remove(task_id)

        return [results_by_id[task_id] for task_id in task_ids]

    def _raise_if_worker_died(self, pending, task_devices) -> None:
        for task_id in list(pending):
            device = task_devices[task_id]
            process = self.processes.get(device)
            if process is not None and not process.is_alive():
                raise RuntimeError(
                    f"Worker for {device} exited with code {process.exitcode} while running task {task_id}"
                )

    def shutdown(self) -> None:
        for device, task_queue in self.task_queues.items():
            process = self.processes.get(device)
            if process is not None and process.is_alive():
                task_queue.put(None)

        for process in self.processes.values():
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)

        self.task_queues.clear()
        self.processes.clear()


def build_client_executor(cfg) -> ClientExecutor:
    if cfg.federated.client_execution == "multiprocessing":
        return MultiprocessingClientExecutor(start_method=cfg.federated.mp_start_method)
    return ThreadClientExecutor()
