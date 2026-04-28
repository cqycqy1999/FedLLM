from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


MODEL_ORDER = ["llama3_8b", "qwen2_7b", "mistral_7b"]
DATASET_ORDER = ["alpaca", "gsm8k", "mmlu"]

MODEL_LABELS = {
    "llama3_8b": "Llama-3-8B",
    "qwen2_7b": "Qwen2-7B",
    "mistral_7b": "Mistral-7B",
}

DATASET_LABELS = {
    "alpaca": "Alpaca",
    "gsm8k": "GSM8K",
    "mmlu": "MMLU",
}

MODEL_COLORS = {
    "llama3_8b": "#2f6f9f",
    "qwen2_7b": "#b64b3f",
    "mistral_7b": "#4f8f58",
}

DATASET_COLORS = {
    "alpaca": "#5b7aa5",
    "gsm8k": "#c07844",
    "mmlu": "#5f9a70",
}


@dataclass
class PlotConfig:
    input_root: str = "outputs/residual_stream"
    output_dir: str = "outputs/residual_stream/figures"
    dpi: int = 220


def plot_residual_stream_figures(cfg: PlotConfig) -> dict[str, str]:
    experiments = _load_all_metrics(cfg.input_root)
    _validate_complete_grid(experiments)
    os.makedirs(cfg.output_dir, exist_ok=True)

    outputs = {
        "cosine_heatmaps": os.path.join(cfg.output_dir, "a_cosine_heatmaps.png"),
        "layer_delta": os.path.join(cfg.output_dir, "b_layer_delta.png"),
        "cross_model_consistency": os.path.join(cfg.output_dir, "c_cross_model_consistency.png"),
        "rho_bar": os.path.join(cfg.output_dir, "d_rho_bar.png"),
    }

    _plot_cosine_heatmaps(experiments, outputs["cosine_heatmaps"], cfg.dpi)
    _plot_layer_delta(experiments, outputs["layer_delta"], cfg.dpi)
    _plot_cross_model_consistency(experiments, outputs["cross_model_consistency"], cfg.dpi)
    _plot_rho_bar(experiments, outputs["rho_bar"], cfg.dpi)

    for name, path in outputs.items():
        print(f"{name}: {path}")
    return outputs


def _load_all_metrics(input_root: str) -> dict[tuple[str, str], dict[str, Any]]:
    experiments = {}
    if not os.path.isdir(input_root):
        raise FileNotFoundError(f"Residual-stream input root does not exist: {input_root}")

    for dirname in sorted(os.listdir(input_root)):
        exp_dir = os.path.join(input_root, dirname)
        if not os.path.isdir(exp_dir) or "__" not in dirname:
            continue
        model_name, dataset_name = dirname.split("__", 1)
        metrics_path = os.path.join(exp_dir, "metrics.pt")
        if not os.path.exists(metrics_path):
            continue
        metrics = torch.load(metrics_path, map_location="cpu")
        experiments[(model_name, dataset_name)] = {
            "name": dirname,
            "dir": exp_dir,
            "metrics_path": metrics_path,
            "metrics": metrics,
        }
    return experiments


def _validate_complete_grid(experiments: dict[tuple[str, str], dict[str, Any]]) -> None:
    missing = [
        f"{model}__{dataset}"
        for model in MODEL_ORDER
        for dataset in DATASET_ORDER
        if (model, dataset) not in experiments
    ]
    if missing:
        raise FileNotFoundError(f"Missing residual-stream metrics for: {missing}")


def _plot_cosine_heatmaps(
    experiments: dict[tuple[str, str], dict[str, Any]],
    output_path: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(DATASET_ORDER),
        figsize=(11.5, 10.5),
        constrained_layout=True,
    )

    image = None
    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, dataset in enumerate(DATASET_ORDER):
            ax = axes[row_idx, col_idx]
            metrics = experiments[(model, dataset)]["metrics"]
            cosine = _to_numpy(metrics["cosine"]["mean"])
            image = ax.imshow(cosine, vmin=0.0, vmax=1.0, origin="lower", cmap="viridis")
            num_layers = cosine.shape[0]
            ticks = np.arange(0, num_layers, 4)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.plot(
                [-0.5, num_layers - 0.5],
                [-0.5, num_layers - 0.5],
                linestyle="--",
                color="white",
                linewidth=1.1,
                alpha=0.9,
            )
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Layer j")
            if col_idx == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\nLayer i")
            ax.set_title(DATASET_LABELS[dataset] if row_idx == 0 else "")

    cbar = fig.colorbar(image, ax=axes, shrink=0.82, pad=0.015)
    cbar.set_label("Cosine similarity")
    fig.suptitle("(a) Layer-pair cosine similarity $C_{ij}$", fontsize=14)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_layer_delta(
    experiments: dict[tuple[str, str], dict[str, Any]],
    output_path: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        len(DATASET_ORDER),
        figsize=(12.0, 9.5),
        sharey=False,
        constrained_layout=True,
    )

    for row_idx, model in enumerate(MODEL_ORDER):
        for col_idx, dataset in enumerate(DATASET_ORDER):
            ax = axes[row_idx, col_idx]
            metrics = experiments[(model, dataset)]["metrics"]
            mean = _to_numpy(metrics["delta"]["mean"])
            std = _to_numpy(metrics["delta"]["std"])
            x = np.arange(len(mean))
            lower = np.clip(mean - std, 1e-8, None)
            upper = np.clip(mean + std, 1e-8, None)
            color = MODEL_COLORS[model]

            ax.plot(x, mean, color=color, linewidth=1.8)
            ax.fill_between(x, lower, upper, color=color, alpha=0.22, linewidth=0)
            ax.axhline(
                float(metrics["delta_random"]["mean"]),
                color="0.45",
                linestyle="--",
                linewidth=1.1,
                label=r"$\delta_{random}$",
            )
            ax.set_yscale("log")
            ax.set_xticks(np.arange(0, len(mean), 4))
            ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
            if row_idx == len(MODEL_ORDER) - 1:
                ax.set_xlabel("Layer index i")
            if col_idx == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model]}\n$\\delta_i$")
            ax.set_title(DATASET_LABELS[dataset] if row_idx == 0 else "")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985))
    fig.suptitle("(b) Per-layer local change rate $\\delta_i$", fontsize=14)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_cross_model_consistency(
    experiments: dict[tuple[str, str], dict[str, Any]],
    output_path: str,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(
        1,
        len(DATASET_ORDER),
        figsize=(13.0, 4.2),
        sharey=False,
        constrained_layout=True,
    )

    for col_idx, dataset in enumerate(DATASET_ORDER):
        ax = axes[col_idx]
        for model in MODEL_ORDER:
            metrics = experiments[(model, dataset)]["metrics"]
            mean = _to_numpy(metrics["delta"]["mean"])
            std = _to_numpy(metrics["delta"]["std"])
            depth = np.linspace(0.0, 1.0, len(mean))
            lower = np.clip(mean - std, 1e-8, None)
            upper = np.clip(mean + std, 1e-8, None)
            color = MODEL_COLORS[model]
            ax.plot(depth, mean, label=MODEL_LABELS[model], color=color, linewidth=2.0)
            ax.fill_between(depth, lower, upper, color=color, alpha=0.12, linewidth=0)

        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Normalized depth i / L")
        if col_idx == 0:
            ax.set_ylabel("$\\delta_i$")
        ax.set_yscale("log")
        ax.set_xlim(0.0, 1.0)
        ax.grid(True, which="both", axis="y", linestyle=":", linewidth=0.6, alpha=0.6)

    axes[-1].legend(loc="best", frameon=False)
    fig.suptitle("(c) Cross-model consistency of local change rate", fontsize=14)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_rho_bar(
    experiments: dict[tuple[str, str], dict[str, Any]],
    output_path: str,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)

    x = np.arange(len(MODEL_ORDER))
    width = 0.23
    offsets = np.linspace(-width, width, len(DATASET_ORDER))

    for idx, dataset in enumerate(DATASET_ORDER):
        values = [
            float(experiments[(model, dataset)]["metrics"]["rho"]["mean"])
            for model in MODEL_ORDER
        ]
        errors = [
            float(experiments[(model, dataset)]["metrics"]["rho"]["std"])
            for model in MODEL_ORDER
        ]
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=width,
            yerr=errors,
            capsize=3,
            label=DATASET_LABELS[dataset],
            color=DATASET_COLORS[dataset],
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.2)
    ax.text(
        len(MODEL_ORDER) - 0.45,
        1.03,
        r"$\rho = 1$",
        color="0.25",
        fontsize=9,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
    ax.set_ylabel(r"Trajectory meandering $\rho$")
    ax.set_title("(d) Residual-stream trajectory meandering")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def parse_args() -> PlotConfig:
    parser = argparse.ArgumentParser(description="Plot residual-stream analysis figures.")
    parser.add_argument("--input_root", default="outputs/residual_stream")
    parser.add_argument("--output_dir", default="outputs/residual_stream/figures")
    parser.add_argument("--dpi", type=int, default=220)
    return PlotConfig(**vars(parser.parse_args()))


def main() -> None:
    plot_residual_stream_figures(parse_args())


if __name__ == "__main__":
    main()
