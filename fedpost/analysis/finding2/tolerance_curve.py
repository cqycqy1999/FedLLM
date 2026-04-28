from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

from fedpost.analysis.finding2.activation import ActivationSubstitution
from fedpost.analysis.finding2.train_utils import (
    Finding2TrainConfig,
    build_lora_model_and_tokenizer,
    capture_trace_for_batch,
    evaluate_loss,
    infer_input_device,
    load_sft_samples,
    make_dataloader,
    move_batch_to_device,
    save_lora_artifacts,
    set_determinism,
    write_json,
)


def run_tolerance_curve(args) -> dict:
    cfg = _train_config_from_args(args)
    set_determinism(cfg.seed)
    eps_values = _parse_eps(args.eps_values)

    train_samples = load_sft_samples(
        dataset_name=cfg.train_dataset_name,
        dataset_config=cfg.train_dataset_config,
        split=cfg.train_split,
        max_samples=cfg.train_samples,
        seed=cfg.seed,
    )
    eval_samples = load_sft_samples(
        dataset_name=cfg.eval_dataset_name or cfg.train_dataset_name,
        dataset_config=cfg.eval_dataset_config or cfg.train_dataset_config,
        split=cfg.eval_split,
        max_samples=cfg.eval_samples,
        seed=cfg.seed + 1,
    )
    if not train_samples or not eval_samples:
        raise RuntimeError("Part C requires non-empty train and eval samples.")

    runs = []
    for eps_idx, eps in enumerate(eps_values):
        runs.append(_run_eps_variant(cfg, args.layer_idx, eps, eps_idx, train_samples, eval_samples))

    baseline_loss = runs[0]["eval"]["loss_mean"]
    result = {
        "protocol": "finding2_part_c_tolerance_curve",
        "semantics": "all_lora_trainable_noisy_activation_substitution",
        "config": {
            **asdict(cfg),
            "layer_idx": args.layer_idx,
            "eps_values": eps_values,
            "loss_tolerance_ratio": args.loss_tolerance_ratio,
        },
        "runs": runs,
        "mu_star_loss_tolerance": _fit_mu_star(runs, baseline_loss, args.loss_tolerance_ratio),
    }
    write_json(os.path.join(cfg.output_dir, "tolerance_curve_result.json"), result)
    print(f"saved Part C result: {os.path.join(cfg.output_dir, 'tolerance_curve_result.json')}")
    return result


def _run_eps_variant(
    cfg: Finding2TrainConfig,
    layer_idx: int,
    eps: float,
    eps_idx: int,
    train_samples,
    eval_samples,
) -> dict:
    variant_dir = os.path.join(cfg.output_dir, f"eps_{eps:.3f}".replace(".", "p"))
    set_determinism(cfg.seed + eps_idx)
    model, tokenizer = build_lora_model_and_tokenizer(cfg)
    device = infer_input_device(model, cfg.device)
    train_loader = make_dataloader(
        train_samples,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=True,
    )
    eval_loader = make_dataloader(
        eval_samples,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    model.train()

    train_losses = []
    loader_iter = _cycle(train_loader)
    generator = torch.Generator(device="cpu").manual_seed(cfg.seed + 100_000 + eps_idx)

    for _step in range(cfg.train_steps):
        batch = move_batch_to_device(next(loader_iter), device)
        optimizer.zero_grad(set_to_none=True)

        trace = capture_trace_for_batch(model, batch, [layer_idx])
        h = trace[layer_idx].to(torch.float32)
        replacement = _noisy_activation(h, eps=eps, generator=generator)
        with ActivationSubstitution(model, boundary_layer_idx=layer_idx, replacement=replacement, detach=True):
            outputs = model(**batch, return_dict=True, use_cache=False)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")
        loss.backward()
        optimizer.step()
        train_losses.append(float(loss.detach().cpu()))

    eval_metrics = evaluate_loss(model, eval_loader, device=device)
    artifacts = save_lora_artifacts(model, tokenizer, variant_dir, adapter_name=cfg.adapter_name)
    train_tensor = torch.tensor(train_losses, dtype=torch.float64)
    result = {
        "eps": eps,
        "train": {
            "loss_mean": float(train_tensor.mean().item()),
            "loss_std": float(train_tensor.std(unbiased=False).item()),
            "num_steps": len(train_losses),
        },
        "eval": eval_metrics,
        "artifacts": artifacts,
    }
    write_json(os.path.join(variant_dir, "result.json"), result)
    return result


def _noisy_activation(h: torch.Tensor, eps: float, generator: torch.Generator) -> torch.Tensor:
    if eps == 0.0:
        return h.detach().clone()
    noise = torch.randn(h.shape, generator=generator, dtype=torch.float32)
    noise = noise / torch.linalg.vector_norm(noise, dim=-1, keepdim=True).clamp_min(1e-12)
    norm = torch.linalg.vector_norm(h, dim=-1, keepdim=True)
    return (h + eps * norm * noise).detach()


def _fit_mu_star(runs: list[dict], baseline_loss: float, tolerance_ratio: float) -> float | None:
    threshold = baseline_loss * tolerance_ratio
    sorted_runs = sorted(runs, key=lambda item: item["eps"])
    previous = sorted_runs[0]
    if previous["eval"]["loss_mean"] > threshold:
        return 0.0
    for current in sorted_runs[1:]:
        current_loss = current["eval"]["loss_mean"]
        if current_loss <= threshold:
            previous = current
            continue
        x0, y0 = previous["eps"], previous["eval"]["loss_mean"]
        x1, y1 = current["eps"], current_loss
        if y1 == y0:
            return x1
        alpha = (threshold - y0) / (y1 - y0)
        return float(x0 + alpha * (x1 - x0))
    return None


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _parse_eps(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _train_config_from_args(args) -> Finding2TrainConfig:
    if args.quick:
        args.train_samples = min(args.train_samples, 16)
        args.eval_samples = min(args.eval_samples, 8)
        args.train_steps = min(args.train_steps, 4)
        args.max_length = min(args.max_length, 128)
    return Finding2TrainConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        train_dataset_name=args.train_dataset_name,
        train_dataset_config=args.train_dataset_config,
        train_split=args.train_split,
        eval_dataset_name=args.eval_dataset_name,
        eval_dataset_config=args.eval_dataset_config,
        eval_split=args.eval_split,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_split_csv(args.lora_target_modules),
        adapter_name=args.adapter_name,
    )


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description="Finding2 Part C noisy-activation tolerance curve.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--eps_values", default="0.0,0.04,0.08,0.12,0.16,0.20,0.24,0.28,0.32")
    parser.add_argument("--loss_tolerance_ratio", type=float, default=1.05)
    parser.add_argument("--train_dataset_name", default="tatsu-lab/alpaca")
    parser.add_argument("--train_dataset_config", default=None)
    parser.add_argument("--train_split", default="train")
    parser.add_argument("--eval_dataset_name", default=None)
    parser.add_argument("--eval_dataset_config", default=None)
    parser.add_argument("--eval_split", default="train")
    parser.add_argument("--train_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--train_steps", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_target_modules", default=None)
    parser.add_argument("--adapter_name", default="default")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    run_tolerance_curve(parse_args())


if __name__ == "__main__":
    main()
