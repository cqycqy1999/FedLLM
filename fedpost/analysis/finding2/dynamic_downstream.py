from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Any

import torch

from fedpost.analysis.finding2.activation import ActivationSubstitution
from fedpost.analysis.finding2.train_utils import (
    Finding2TrainConfig,
    build_lora_model_and_tokenizer,
    build_synthesizer,
    capture_trace_for_batch,
    evaluate_loss,
    fit_synthesizer_from_hidden_trace,
    infer_input_device,
    load_sft_samples,
    make_dataloader,
    move_batch_to_device,
    save_lora_artifacts,
    set_determinism,
    write_json,
)


def run_dynamic_downstream(args) -> dict[str, Any]:
    cfg = _train_config_from_args(args)
    set_determinism(cfg.seed)

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
        raise RuntimeError("Part B requires non-empty train and eval samples.")

    baseline = _run_variant(
        cfg=cfg,
        variant_name="baseline",
        train_samples=train_samples,
        eval_samples=eval_samples,
        layer_idx=args.layer_idx,
        synthesizer=None,
    )

    synthesizer = build_synthesizer(
        name=args.synthesizer,
        seed=cfg.seed,
        low_rank_rank=args.low_rank_rank,
        s5_rank=args.s5_rank,
        s5_top_k=args.s5_top_k,
        distilled_feature_dim=args.distilled_feature_dim,
        neighbor_k=args.neighbor_k,
    )
    allow_small_calibration = bool(args.quick or args.allow_small_calibration)
    synthesizer, synthesizer_fit = fit_synthesizer_from_hidden_trace(
        hidden_trace_path=args.hidden_trace_path,
        synthesizer=synthesizer,
        layer_idx=args.layer_idx,
        calibration_vectors=args.calibration_vectors,
        seed=cfg.seed,
        allow_small_calibration=allow_small_calibration,
    )

    substituted = _run_variant(
        cfg=cfg,
        variant_name=f"substituted_{synthesizer.name}",
        train_samples=train_samples,
        eval_samples=eval_samples,
        layer_idx=args.layer_idx,
        synthesizer=synthesizer,
        neighbor_k=args.neighbor_k,
    )

    result = {
        "protocol": "finding2_part_b_dynamic_downstream",
        "semantics": "all_lora_trainable_activation_substitution",
        "config": {
            **asdict(cfg),
            "layer_idx": args.layer_idx,
            "synthesizer": args.synthesizer,
            "hidden_trace_path": args.hidden_trace_path,
            "calibration_vectors": args.calibration_vectors,
            "allow_small_calibration": allow_small_calibration,
        },
        "synthesizer_fit": synthesizer_fit,
        "baseline": baseline,
        "substituted": substituted,
        "loss_delta": substituted["eval"]["loss_mean"] - baseline["eval"]["loss_mean"],
        "loss_ratio": substituted["eval"]["loss_mean"] / max(1e-12, baseline["eval"]["loss_mean"]),
    }
    write_json(os.path.join(cfg.output_dir, "dynamic_downstream_result.json"), result)
    print(f"saved Part B result: {os.path.join(cfg.output_dir, 'dynamic_downstream_result.json')}")
    return result


def _run_variant(
    cfg: Finding2TrainConfig,
    variant_name: str,
    train_samples,
    eval_samples,
    layer_idx: int,
    synthesizer=None,
    neighbor_k: int = 1,
) -> dict[str, Any]:
    variant_dir = os.path.join(cfg.output_dir, variant_name)
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
    capture_layers = _capture_layers_for_synthesizer(synthesizer, layer_idx, neighbor_k)

    for step in range(cfg.train_steps):
        batch = move_batch_to_device(next(loader_iter), device)
        optimizer.zero_grad(set_to_none=True)

        if synthesizer is None:
            outputs = model(**batch, return_dict=True, use_cache=False)
        else:
            trace = capture_trace_for_batch(model, batch, capture_layers)
            anchor = synthesizer.build_anchor(trace)
            replacement = synthesizer.synthesize(anchor).detach()
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
        "variant": variant_name,
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


def _capture_layers_for_synthesizer(synthesizer, layer_idx: int, neighbor_k: int) -> list[int]:
    if synthesizer is None:
        return []
    layers = {layer_idx}
    if synthesizer.name == "S3_distilled":
        layers.add(0)
    if synthesizer.name.startswith("S4_neighbor"):
        layers.add(layer_idx - neighbor_k)
    return sorted(layer for layer in layers if layer >= 0)


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


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
    parser = argparse.ArgumentParser(description="Finding2 Part B dynamic downstream LoRA training.")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--hidden_trace_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--synthesizer", default="S5_anchor_correction")
    parser.add_argument("--calibration_vectors", type=int, default=2000)
    parser.add_argument(
        "--allow_small_calibration",
        action="store_true",
        help=(
            "Allow using fewer calibration vectors than requested. "
            "The runner records requested/used/available counts in the result JSON. "
            "--quick enables this automatically."
        ),
    )
    parser.add_argument("--low_rank_rank", type=int, default=64)
    parser.add_argument("--s5_rank", type=int, default=64)
    parser.add_argument("--s5_top_k", type=int, default=64)
    parser.add_argument("--distilled_feature_dim", type=int, default=128)
    parser.add_argument("--neighbor_k", type=int, default=1)
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
    run_dynamic_downstream(parse_args())


if __name__ == "__main__":
    main()
