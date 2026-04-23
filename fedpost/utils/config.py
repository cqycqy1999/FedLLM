from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any
import yaml
import os


@dataclass
class ModelConfig:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = None
    trust_remote_code: bool = False
    torch_dtype: str = "float32"
    use_flash_attn: bool = False
    gradient_checkpointing: bool = False


@dataclass
class PEFTConfig:
    method: str = "none"
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: Optional[list[str]] = None
    adapter_name: str = "default"


@dataclass
class FederatedConfig:
    algorithm: str = "fedavg"
    num_clients: int = 4
    clients_per_round: int = 2
    rounds: int = 3
    local_epochs: int = 1
    local_steps: Optional[int] = None
    sample_mode: str = "uniform"
    gpu_ids: list[int] = field(default_factory=list)
    max_parallel_clients: Optional[int] = None


@dataclass
class SFTConfig:
    max_length: int = 128
    lr: float = 1e-4
    batch_size: int = 2
    grad_accum_steps: int = 1


@dataclass
class DPOConfig:
    max_length: int = 128
    max_prompt_length: int = 64
    lr: float = 1e-5
    batch_size: int = 1
    grad_accum_steps: int = 1
    beta: float = 0.1
    reference_mode: str = "frozen_copy"


@dataclass
class DataConfig:
    task: str = "sft"
    source: str = "hf"          # hf / local
    dataset_name: str = ""
    dataset_split: str = "train"
    partitioner: str = "iid"
    partition_seed: int = 42
    max_samples: Optional[int] = None


@dataclass
class EvalConfig:
    eval_every: int = 1
    save_every: int = 1
    evaluator_type: str = "alpaca_eval"

    run_alpaca_eval: bool = False
    alpaca_eval_model_name: str = "fedpost_model"
    alpaca_eval_annotators_config: str = "alpaca_eval_gpt4_turbo_fn"

    run_mt_bench: bool = False
    mt_bench_model_id: str = "fedpost_model"
    mt_bench_judge_model: str = "gpt-4"
    mt_bench_parallel: int = 2

    run_lm_eval: bool = False
    lm_eval_tasks: list[str] = field(default_factory=list)
    lm_eval_batch_size: str = "auto"
    lm_eval_device: Optional[str] = None

    eval_generation_max_new_tokens: int = 128

    summary_primary_metric: Optional[str] = None


@dataclass
class ExperimentConfig:
    task: str
    model: ModelConfig
    peft: PEFTConfig
    federated: FederatedConfig
    data: DataConfig
    eval: EvalConfig
    output_dir: str = "outputs/default"
    seed: int = 42
    sft: Optional[SFTConfig] = None
    dpo: Optional[DPOConfig] = None


class ConfigLoader:
    @staticmethod
    def from_yaml(path: str) -> ExperimentConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        return ExperimentConfig(
            task=raw["task"],
            model=ModelConfig(**raw["model"]),
            peft=PEFTConfig(**raw["peft"]),
            federated=FederatedConfig(**raw["federated"]),
            data=DataConfig(**raw["data"]),
            eval=EvalConfig(**raw["eval"]),
            output_dir=raw.get("output_dir", "outputs/default"),
            seed=raw.get("seed", 42),
            sft=SFTConfig(**raw["sft"]) if raw.get("sft") else None,
            dpo=DPOConfig(**raw["dpo"]) if raw.get("dpo") else None,
        )

    @staticmethod
    def validate(cfg: ExperimentConfig) -> None:
        if cfg.task not in {"sft", "dpo"}:
            raise ValueError(f"Unsupported task: {cfg.task}")

        if cfg.peft.method not in {"none", "lora"}:
            raise ValueError(f"Unsupported peft method: {cfg.peft.method}")

        if cfg.federated.clients_per_round > cfg.federated.num_clients:
            raise ValueError("clients_per_round cannot exceed num_clients")

        if any(gpu_id < 0 for gpu_id in cfg.federated.gpu_ids):
            raise ValueError("gpu_ids must contain non-negative integers only")

        if cfg.federated.max_parallel_clients is not None and cfg.federated.max_parallel_clients <= 0:
            raise ValueError("max_parallel_clients must be positive when provided")

        if cfg.task == "sft" and cfg.sft is None:
            raise ValueError("SFT config is required when task='sft'")

        if cfg.task == "dpo" and cfg.dpo is None:
            raise ValueError("DPO config is required when task='dpo'")

        if cfg.peft.method == "lora" and not cfg.peft.target_modules:
            raise ValueError("LoRA requires non-empty target_modules")

        if cfg.data.source != "hf":
            raise ValueError("This upgraded template currently supports data.source='hf' only.")
