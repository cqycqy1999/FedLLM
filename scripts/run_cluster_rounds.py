from __future__ import annotations

import argparse
import json
import math
import os
import random
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedpost.utils.config import ConfigLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="A100 server orchestrator for SSH-based edge-client BP-FedPEFT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cluster", required=True, help="JSON manifest with one or more edge clients and a server.")
    parser.add_argument("--work-dir", default="outputs/cluster_run")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument("--selected-client-ids", nargs="*", default=None)
    parser.add_argument("--strict-client-count", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = ConfigLoader.from_yaml(args.config)
    cluster = load_cluster(args.cluster)
    clients = select_manifest_clients(cluster["clients"], args.selected_client_ids)
    reconcile_client_count(
        cfg,
        clients,
        clients_per_round=args.clients_per_round,
        strict_client_count=args.strict_client_count,
    )
    ConfigLoader.validate(cfg)
    os.makedirs(args.work_dir, exist_ok=True)

    num_rounds = args.rounds if args.rounds is not None else cfg.federated.rounds
    if args.dry_run:
        for round_idx in range(num_rounds):
            selected_clients = sample_clients(clients, cfg, round_idx)
            round_dir = os.path.join(args.work_dir, f"round_{round_idx + 1:04d}")
            broadcast_path = os.path.join(round_dir, "broadcast.pt")
            for client in selected_clients:
                print_commands(build_client_commands(client, args.config, broadcast_path, round_dir))
        return

    import torch
    import fedpost.algorithms  # noqa: F401
    from fedpost.federation.message import BroadcastPayload
    from fedpost.federation.server import Server
    from fedpost.evaluation.paper import PaperBenchmarkEvaluator
    from fedpost.models.loader import HFModelManager
    from fedpost.utils.recorder import Recorder
    from fedpost.utils.registry import Registry
    from fedpost.utils.seed import set_seed

    set_seed(cfg.seed)
    model_manager = HFModelManager(cfg)
    model_bundle = model_manager.build()
    algo_cls = Registry.get("algorithm", cfg.federated.algorithm)
    aggregator = algo_cls.aggregator_cls(cfg)
    algorithm = algo_cls(cfg, aggregator)
    server = Server(cfg, model_manager, model_bundle, aggregator)
    recorder = Recorder(args.work_dir)
    recorder.save_config(cfg)
    evaluator = PaperBenchmarkEvaluator(cfg, model_bundle.tokenizer) if cfg.eval.tasks else None

    for round_idx in range(num_rounds):
        if hasattr(algorithm, "should_stop_training") and algorithm.should_stop_training():
            break

        server.round_idx = round_idx
        selected_clients = sample_clients(clients, cfg, round_idx)
        algorithm.before_broadcast(server, selected_clients, round_idx)
        payload = algorithm.make_broadcast_payload(server, round_idx)

        round_dir = os.path.join(args.work_dir, f"round_{round_idx + 1:04d}")
        os.makedirs(round_dir, exist_ok=True)
        broadcast_path = os.path.join(round_dir, "broadcast.pt")
        torch.save(payload_to_dict(payload), broadcast_path)

        with ThreadPoolExecutor(max_workers=len(selected_clients)) as pool:
            futures = [
                pool.submit(run_remote_client, client, args.config, broadcast_path, round_dir)
                for client in selected_clients
            ]
            result_paths = [future.result() for future in futures]

        results = [torch.load(path, map_location="cpu", weights_only=False) for path in result_paths]
        agg_metrics = algorithm.server_update(server, results)
        round_metrics = summarize_round(results, agg_metrics)
        eval_result = None
        if evaluator is not None and cfg.eval.eval_every > 0 and (round_idx + 1) % cfg.eval.eval_every == 0:
            eval_result = evaluator.evaluate(server.evaluate_model(), round_idx, model_artifacts={})
            recorder.record_eval(eval_result)
            round_metrics.update({f"eval/{key}": value for key, value in eval_result.metrics.items()})

        recorder.record_round(round_idx, round_metrics, results)
        recorder.record_round_summary(
            round_idx,
            round_metrics,
            eval_result=eval_result,
            primary_metric=cfg.eval.summary_primary_metric,
        )

        if cfg.eval.save_every > 0 and (round_idx + 1) % cfg.eval.save_every == 0:
            server.save_checkpoint(os.path.join(round_dir, "server_checkpoint.pt"))
        print(json.dumps({"round": round_idx + 1, **round_metrics}, ensure_ascii=False))


def load_cluster(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cluster = json.load(f)
    if "clients" not in cluster or not cluster["clients"]:
        raise ValueError("cluster manifest must contain a non-empty clients list")
    seen = set()
    for client in cluster["clients"]:
        client_id = client.get("client_id")
        if not client_id:
            raise ValueError("Every cluster client must define client_id")
        if client_id in seen:
            raise ValueError(f"Duplicate client_id in cluster manifest: {client_id}")
        seen.add(client_id)
    return cluster


def select_manifest_clients(clients: list[dict], selected_client_ids: list[str] | None) -> list[dict]:
    if not selected_client_ids:
        return clients
    wanted = set(selected_client_ids)
    selected = [client for client in clients if client["client_id"] in wanted]
    missing = wanted - {client["client_id"] for client in selected}
    if missing:
        raise ValueError(f"selected client ids are not in the cluster manifest: {sorted(missing)}")
    if not selected:
        raise ValueError("selected client set is empty")
    return selected


def reconcile_client_count(
    cfg,
    clients: list[dict],
    clients_per_round: int | None = None,
    strict_client_count: bool = False,
) -> None:
    manifest_count = len(clients)
    configured_count = cfg.federated.num_clients
    if strict_client_count and configured_count != manifest_count:
        raise ValueError(
            f"Config num_clients={configured_count} does not match cluster manifest "
            f"count={manifest_count}. Remove --strict-client-count to auto-adapt."
        )

    cfg.federated.num_clients = manifest_count
    if clients_per_round is not None:
        if clients_per_round <= 0:
            raise ValueError("--clients-per-round must be positive")
        cfg.federated.clients_per_round = min(clients_per_round, manifest_count)
        cfg.federated.proportion = None
    elif cfg.federated.proportion is None:
        cfg.federated.clients_per_round = min(cfg.federated.clients_per_round, manifest_count)


def sample_clients(clients: list[dict], cfg, round_idx: int) -> list[dict]:
    if cfg.federated.proportion is not None:
        count = max(1, min(len(clients), math.ceil(len(clients) * cfg.federated.proportion)))
    else:
        count = min(len(clients), cfg.federated.clients_per_round)
    rnd = random.Random(cfg.seed + round_idx)
    return rnd.sample(clients, k=count)


def run_remote_client(client: dict, config_path: str, broadcast_path: str, round_dir: str) -> str:
    commands = build_client_commands(client, config_path, broadcast_path, round_dir)
    for command in commands["push"]:
        run(command)
    run(commands["ssh"])
    for command in commands["pull"]:
        run(command)
    return commands["local_result"]


def build_client_commands(client: dict, config_path: str, broadcast_path: str, round_dir: str) -> dict:
    host = client["host"]
    user = client.get("user")
    target = f"{user}@{host}" if user else host
    client_id = client["client_id"]
    remote_repo = client["remote_repo"]
    remote_work = client.get("remote_work_dir", f"~/bpfedpeft_runs/{client_id}")
    remote_config = client.get("remote_config", os.path.join(remote_repo, config_path))
    remote_data = client["remote_data"]
    device = client.get("device", "cuda:0")

    local_result = os.path.join(round_dir, f"{client_id}.result.pt")
    remote_broadcast = f"{remote_work}/broadcast.pt"
    remote_result = f"{remote_work}/{client_id}.result.pt"

    worker_cmd = " ".join([
        "cd", shlex.quote(remote_repo), "&&",
        "python", "scripts/run_edge_worker.py",
        "--config", shlex.quote(remote_config),
        "--client-id", shlex.quote(client_id),
        "--client-data", shlex.quote(remote_data),
        "--broadcast", shlex.quote(remote_broadcast),
        "--output", shlex.quote(remote_result),
        "--device", shlex.quote(device),
    ])

    return {
        "local_result": local_result,
        "push": [
            ["ssh", target, "mkdir", "-p", remote_work],
            ["rsync", "-az", broadcast_path, f"{target}:{remote_broadcast}"],
        ],
        "ssh": ["ssh", target, worker_cmd],
        "pull": [["rsync", "-az", f"{target}:{remote_result}", local_result]],
    }


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def print_commands(commands: dict) -> None:
    for section in ("push",):
        for command in commands[section]:
            print(" ".join(shlex.quote(part) for part in command))
    print(" ".join(shlex.quote(part) for part in commands["ssh"]))
    for command in commands["pull"]:
        print(" ".join(shlex.quote(part) for part in command))


def payload_to_dict(payload: BroadcastPayload) -> dict:
    return {
        "round_idx": payload.round_idx,
        "global_step": payload.global_step,
        "model_state": payload.model_state,
        "algo_state": payload.algo_state,
        "metadata": payload.metadata,
    }


def summarize_round(results, agg_metrics: dict) -> dict:
    success_results = [result for result in results if result.success]
    failed_results = [result for result in results if not result.success]
    avg_loss = 0.0
    if success_results:
        losses = [result.metrics.get("loss", 0.0) for result in success_results]
        avg_loss = sum(losses) / len(losses)
    return {
        "avg_client_loss": avg_loss,
        "num_selected_clients": len(results),
        "num_success_clients": len(success_results),
        "num_failed_clients": len(failed_results),
        "success_rate": len(success_results) / len(results) if results else 0.0,
        "successful_client_ids": [result.client_id for result in success_results],
        "failed_client_ids": [result.client_id for result in failed_results],
        **agg_metrics,
    }


if __name__ == "__main__":
    main()
