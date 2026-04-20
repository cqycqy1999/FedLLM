from __future__ import annotations

from dataclasses import dataclass
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
    data_path: str = ""
    file_type: str = "jsonl"   # json / jsonl
    partitioner: str = "iid"
    partition_seed: int = 42

    prompt_field: str = "prompt"
    response_field: str = "response"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"


@dataclass
class EvalConfig:
    eval_every: int = 1
    save_every: int = 1


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

        if cfg.task == "sft" and cfg.sft is None:
            raise ValueError("SFT config is required when task='sft'")

        if cfg.task == "dpo" and cfg.dpo is None:
            raise ValueError("DPO config is required when task='dpo'")

        if cfg.peft.method == "lora" and not cfg.peft.target_modules:
            raise ValueError("LoRA requires non-empty target_modules")

        if not cfg.data.data_path:
            raise ValueError("data.data_path is required")

        if cfg.data.file_type not in {"json", "jsonl"}:
            raise ValueError("data.file_type must be one of {'json', 'jsonl'}")

        if not os.path.exists(cfg.data.data_path):
            raise FileNotFoundError(f"Data file not found: {cfg.data.data_path}")