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
    proportion: Optional[float] = None
    rounds: int = 3
    num_rounds: Optional[int] = None
    local_epochs: int = 1
    num_epochs: Optional[int] = None
    local_steps: Optional[int] = None
    num_steps: Optional[int] = None
    sample_mode: str = "uniform"
    client_execution: str = "thread"
    mp_start_method: str = "spawn"
    gpu_ids: list[int] = field(default_factory=list)
    max_parallel_clients: Optional[int] = None
    fail_fast: bool = False
    min_success_rate: float = 1.0
    early_stop_metric: Optional[str] = None
    early_stop_mode: str = "max"
    early_stop_patience: Optional[int] = None
    early_stop_min_delta: float = 0.0


@dataclass
class SFTConfig:
    max_length: int = 128
    lr: float = 1e-4
    learning_rate: Optional[float] = None
    batch_size: int = 2
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: Optional[float] = None
    lr_scheduler: str = "constant"
    learning_rate_decay: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0


@dataclass
class DPOConfig:
    max_length: int = 128
    max_prompt_length: int = 64
    lr: float = 1e-5
    learning_rate: Optional[float] = None
    batch_size: int = 1
    grad_accum_steps: int = 1
    optimizer: str = "adamw"
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: Optional[float] = None
    lr_scheduler: str = "constant"
    learning_rate_decay: float = 1.0
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
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
    save_adapter_every: Optional[int] = None
    merge_every: Optional[int] = None
    eval_requires_merged_model: bool = True
    eval_device: Optional[str] = None
    fail_on_eval_error: bool = False

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

        federated_raw = dict(raw["federated"])
        if federated_raw.get("num_rounds") is not None:
            federated_raw["rounds"] = federated_raw["num_rounds"]
        if federated_raw.get("num_epochs") is not None:
            federated_raw["local_epochs"] = federated_raw["num_epochs"]
        if federated_raw.get("num_steps") is not None:
            federated_raw["local_steps"] = federated_raw["num_steps"]

        return ExperimentConfig(
            task=raw["task"],
            model=ModelConfig(**raw["model"]),
            peft=PEFTConfig(**raw["peft"]),
            federated=FederatedConfig(**federated_raw),
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

        if cfg.federated.proportion is not None and not (0 < cfg.federated.proportion <= 1):
            raise ValueError("proportion must be in (0, 1] when provided")

        if cfg.federated.rounds <= 0:
            raise ValueError("rounds/num_rounds must be positive")

        if cfg.federated.local_epochs <= 0:
            raise ValueError("local_epochs/num_epochs must be positive")

        if cfg.federated.local_steps is not None and cfg.federated.local_steps <= 0:
            raise ValueError("local_steps/num_steps must be positive when provided")

        if any(gpu_id < 0 for gpu_id in cfg.federated.gpu_ids):
            raise ValueError("gpu_ids must contain non-negative integers only")

        if cfg.federated.max_parallel_clients is not None and cfg.federated.max_parallel_clients <= 0:
            raise ValueError("max_parallel_clients must be positive when provided")

        if not (0 < cfg.federated.min_success_rate <= 1):
            raise ValueError("min_success_rate must be in (0, 1]")

        if cfg.federated.early_stop_mode not in {"max", "min"}:
            raise ValueError("early_stop_mode must be either 'max' or 'min'")

        if cfg.federated.early_stop_patience is not None and cfg.federated.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be non-negative when provided")

        if cfg.federated.client_execution not in {"thread", "multiprocessing"}:
            raise ValueError("client_execution must be either 'thread' or 'multiprocessing'")

        if cfg.federated.mp_start_method not in {"spawn", "forkserver"}:
            raise ValueError("mp_start_method must be either 'spawn' or 'forkserver'")

        for field_name in ("save_adapter_every", "merge_every"):
            value = getattr(cfg.eval, field_name)
            if value is not None and value < 0:
                raise ValueError(f"{field_name} must be non-negative when provided")

        if cfg.task == "sft" and cfg.sft is None:
            raise ValueError("SFT config is required when task='sft'")

        if cfg.task == "dpo" and cfg.dpo is None:
            raise ValueError("DPO config is required when task='dpo'")

        train_cfg = cfg.sft if cfg.task == "sft" else cfg.dpo
        if train_cfg is not None:
            if train_cfg.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if train_cfg.grad_accum_steps <= 0:
                raise ValueError("grad_accum_steps must be positive")
            if (train_cfg.learning_rate if train_cfg.learning_rate is not None else train_cfg.lr) <= 0:
                raise ValueError("learning_rate/lr must be positive")
            if train_cfg.optimizer not in {"adamw", "adam", "sgd"}:
                raise ValueError("optimizer must be one of: adamw, adam, sgd")
            if train_cfg.lr_scheduler not in {"constant", "linear", "cosine", "exponential"}:
                raise ValueError("lr_scheduler must be one of: constant, linear, cosine, exponential")
            if train_cfg.learning_rate_decay < 0:
                raise ValueError("learning_rate_decay must be non-negative")
            if train_cfg.warmup_steps < 0:
                raise ValueError("warmup_steps must be non-negative")
            if not (0 <= train_cfg.warmup_ratio < 1):
                raise ValueError("warmup_ratio must be in [0, 1)")

        if cfg.peft.method == "lora" and not cfg.peft.target_modules:
            raise ValueError("LoRA requires non-empty target_modules")

        if cfg.data.source != "hf":
            raise ValueError("This upgraded template currently supports data.source='hf' only.")
