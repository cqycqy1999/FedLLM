from __future__ import annotations

import argparse
import gc
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from fedpost.evaluation.runners.alpaca_eval_runner import AlpacaEvalRunner
from fedpost.evaluation.runners.lm_eval_runner import LMEvalRunner
from fedpost.analysis.finding2.train_utils import parse_dtype, write_json


@dataclass
class Finding2EvaluationConfig:
    base_model_name_or_path: str
    adapter_dir: str
    output_dir: str
    eval_name: str = "finding2_adapter"
    run_lm_eval: bool = True
    lm_eval_tasks: list[str] | None = None
    lm_eval_batch_size: str = "auto"
    lm_eval_device: str | None = None
    run_alpaca_eval: bool = False
    alpaca_eval_limit: int | None = None
    alpaca_eval_annotators_config: str = "alpaca_eval_gpt4_turbo_fn"
    generation_max_new_tokens: int = 256
    dtype: str = "bfloat16"
    device: str | None = None
    device_map: str | None = None
    trust_remote_code: bool = False
    fail_on_error: bool = False
    quick: bool = False


def evaluate_finding2_adapter(cfg: Finding2EvaluationConfig) -> dict[str, Any]:
    _validate_inputs(cfg)
    os.makedirs(cfg.output_dir, exist_ok=True)

    if cfg.quick:
        if cfg.lm_eval_tasks is None:
            cfg.lm_eval_tasks = ["gsm8k"]
        cfg.alpaca_eval_limit = 16 if cfg.alpaca_eval_limit is None else min(cfg.alpaca_eval_limit, 16)

    result = {
        "protocol": "finding2_standard_adapter_evaluation",
        "config": asdict(cfg),
        "metrics": {},
        "artifacts": {},
        "failures": [],
    }

    if cfg.run_lm_eval:
        _run_lm_eval(cfg, result)

    if cfg.run_alpaca_eval:
        _run_alpaca_eval(cfg, result)

    result_path = os.path.join(cfg.output_dir, "evaluation_result.json")
    write_json(result_path, result)
    print(f"saved Finding2 evaluation result: {result_path}")
    if cfg.fail_on_error and result["failures"]:
        raise RuntimeError(f"Finding2 evaluation had failures: {result['failures']}")
    return result


def evaluate_from_yaml_config(
    config_path: str,
    adapter_name: str | None = None,
    quick: bool = False,
    fail_on_error: bool = False,
) -> list[dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    adapters = raw.get("adapters", [])
    if adapter_name is not None:
        adapters = [adapter for adapter in adapters if adapter.get("name") == adapter_name]
        if not adapters:
            raise ValueError(f"No adapter named {adapter_name} found in {config_path}")
    if not adapters:
        raise ValueError(f"No adapters configured in {config_path}")

    evaluators = raw.get("evaluators", {})
    results = []
    for adapter in adapters:
        cfg = Finding2EvaluationConfig(
            base_model_name_or_path=adapter.get("base_model_name_or_path", raw["base_model_name_or_path"]),
            adapter_dir=adapter["adapter_dir"],
            output_dir=adapter["output_dir"],
            eval_name=adapter.get("name", "finding2_adapter"),
            run_lm_eval=bool(evaluators.get("run_lm_eval", True)),
            lm_eval_tasks=_normalize_tasks(evaluators.get("lm_eval_tasks", ["gsm8k", "mmlu"])),
            lm_eval_batch_size=evaluators.get("lm_eval_batch_size", "auto"),
            lm_eval_device=evaluators.get("lm_eval_device"),
            run_alpaca_eval=bool(evaluators.get("run_alpaca_eval", False)),
            alpaca_eval_limit=evaluators.get("alpaca_eval_limit"),
            alpaca_eval_annotators_config=evaluators.get(
                "alpaca_eval_annotators_config",
                "alpaca_eval_gpt4_turbo_fn",
            ),
            generation_max_new_tokens=int(evaluators.get("generation_max_new_tokens", 256)),
            dtype=adapter.get("dtype", raw.get("dtype", "bfloat16")),
            device=adapter.get("device", raw.get("device")),
            device_map=adapter.get("device_map", raw.get("device_map")),
            trust_remote_code=bool(adapter.get("trust_remote_code", raw.get("trust_remote_code", False))),
            fail_on_error=bool(fail_on_error or raw.get("fail_on_error", False)),
            quick=quick,
        )
        results.append(evaluate_finding2_adapter(cfg))
    return results


def _run_lm_eval(cfg: Finding2EvaluationConfig, result: dict[str, Any]) -> None:
    tasks = cfg.lm_eval_tasks or ["gsm8k", "mmlu"]
    output_dir = os.path.join(cfg.output_dir, "lm_eval")
    runner = LMEvalRunner(output_dir)
    lm_result = runner.run(
        model_path=cfg.base_model_name_or_path,
        peft_path=cfg.adapter_dir,
        tasks=tasks,
        batch_size=cfg.lm_eval_batch_size,
        device=cfg.lm_eval_device,
        trust_remote_code=cfg.trust_remote_code,
        dtype=cfg.dtype,
    )
    result["artifacts"]["lm_eval_result_path"] = lm_result.get("result_path")
    result["artifacts"]["lm_eval_output_dir"] = output_dir
    result["artifacts"]["lm_eval_cmd"] = lm_result.get("cmd")
    result["metrics"]["lm_eval_returncode"] = int(lm_result.get("returncode", 1))

    if lm_result.get("returncode", 1) != 0:
        _record_failure(
            result,
            name="lm_eval",
            output_dir=output_dir,
            stdout=lm_result.get("stdout", ""),
            stderr=lm_result.get("stderr", ""),
        )
        return

    parsed = lm_result.get("parsed")
    if parsed and "results" in parsed:
        for task_name, task_metrics in parsed["results"].items():
            for metric_name, metric_value in task_metrics.items():
                if isinstance(metric_value, (float, int)):
                    result["metrics"][f"lm_eval/{task_name}/{metric_name}"] = float(metric_value)


def _run_alpaca_eval(cfg: Finding2EvaluationConfig, result: dict[str, Any]) -> None:
    output_dir = os.path.join(cfg.output_dir, "alpaca_eval")
    os.makedirs(output_dir, exist_ok=True)
    model = None
    try:
        model, tokenizer = _load_base_plus_adapter(cfg)
        outputs = _generate_alpaca_outputs(
            model=model,
            tokenizer=tokenizer,
            model_id=cfg.eval_name,
            max_new_tokens=cfg.generation_max_new_tokens,
            limit=cfg.alpaca_eval_limit,
        )
        runner = AlpacaEvalRunner(output_dir)
        alpaca_result = runner.run(
            outputs,
            annotators_config=cfg.alpaca_eval_annotators_config,
        )
        result["metrics"]["alpaca_eval_returncode"] = int(alpaca_result.get("returncode", 1))
        result["artifacts"]["alpaca_eval_result_dir"] = alpaca_result.get("result_dir")
        result["artifacts"]["alpaca_eval_outputs_path"] = alpaca_result.get("outputs_path")
        if alpaca_result.get("returncode", 1) != 0:
            _record_failure(
                result,
                name="alpaca_eval",
                output_dir=output_dir,
                stdout=alpaca_result.get("stdout", ""),
                stderr=alpaca_result.get("stderr", ""),
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load_base_plus_adapter(cfg: Finding2EvaluationConfig):
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name_or_path,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": parse_dtype(cfg.dtype),
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.device_map:
        kwargs["device_map"] = cfg.device_map
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_name_or_path, **kwargs)
    model = PeftModel.from_pretrained(base_model, cfg.adapter_dir)
    if cfg.device_map is None:
        device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.to(device)
    model.eval()
    return model, tokenizer


def _generate_alpaca_outputs(
    model,
    tokenizer,
    model_id: str,
    max_new_tokens: int,
    limit: int | None,
) -> list[dict[str, str]]:
    eval_path = hf_hub_download(
        repo_id="tatsu-lab/alpaca_eval",
        filename="alpaca_eval.json",
        repo_type="dataset",
    )
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_set = json.load(f)
    if limit is not None:
        eval_set = eval_set[:limit]

    device = next(model.parameters()).device
    outputs = []
    for example in eval_set:
        instruction = example["instruction"]
        prompt = f"User: {instruction}\nAssistant:"
        batch = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        outputs.append({
            "instruction": instruction,
            "output": answer,
            "generator": model_id,
        })
    return outputs


def _record_failure(
    result: dict[str, Any],
    name: str,
    output_dir: str,
    stdout: str,
    stderr: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    stdout_path = os.path.join(output_dir, f"{name}_stdout.txt")
    stderr_path = os.path.join(output_dir, f"{name}_stderr.txt")
    with open(stdout_path, "w", encoding="utf-8") as f:
        f.write(stdout)
    with open(stderr_path, "w", encoding="utf-8") as f:
        f.write(stderr)
    result["artifacts"][f"{name}_stdout_path"] = stdout_path
    result["artifacts"][f"{name}_stderr_path"] = stderr_path
    result["failures"].append({
        "name": name,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    })


def _validate_inputs(cfg: Finding2EvaluationConfig) -> None:
    if not os.path.exists(cfg.adapter_dir):
        raise FileNotFoundError(f"adapter_dir does not exist: {cfg.adapter_dir}")
    if not cfg.run_lm_eval and not cfg.run_alpaca_eval:
        raise ValueError("At least one evaluator must be enabled.")


def _split_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_tasks(value) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _split_csv(value)
    return [str(item) for item in value]


def _config_from_args(args) -> Finding2EvaluationConfig:
    return Finding2EvaluationConfig(
        base_model_name_or_path=args.base_model_name_or_path,
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        eval_name=args.eval_name,
        run_lm_eval=args.run_lm_eval,
        lm_eval_tasks=_split_csv(args.lm_eval_tasks),
        lm_eval_batch_size=args.lm_eval_batch_size,
        lm_eval_device=args.lm_eval_device,
        run_alpaca_eval=args.run_alpaca_eval,
        alpaca_eval_limit=args.alpaca_eval_limit,
        alpaca_eval_annotators_config=args.alpaca_eval_annotators_config,
        generation_max_new_tokens=args.generation_max_new_tokens,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        fail_on_error=args.fail_on_error,
        quick=args.quick,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Finding2 LoRA adapters with standard academic evaluators.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--adapter_name", default=None)
    parser.add_argument("--base_model_name_or_path", default=None)
    parser.add_argument("--adapter_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--eval_name", default="finding2_adapter")
    parser.add_argument("--run_lm_eval", action="store_true")
    parser.add_argument("--lm_eval_tasks", default="gsm8k,mmlu")
    parser.add_argument("--lm_eval_batch_size", default="auto")
    parser.add_argument("--lm_eval_device", default=None)
    parser.add_argument("--run_alpaca_eval", action="store_true")
    parser.add_argument("--alpaca_eval_limit", type=int, default=None)
    parser.add_argument("--alpaca_eval_annotators_config", default="alpaca_eval_gpt4_turbo_fn")
    parser.add_argument("--generation_max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--device_map", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--fail_on_error", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        evaluate_from_yaml_config(
            config_path=args.config,
            adapter_name=args.adapter_name,
            quick=args.quick,
            fail_on_error=args.fail_on_error,
        )
        return

    required = {
        "base_model_name_or_path": args.base_model_name_or_path,
        "adapter_dir": args.adapter_dir,
        "output_dir": args.output_dir,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required args for direct mode: {missing}. Or pass --config.")
    evaluate_finding2_adapter(_config_from_args(args))


if __name__ == "__main__":
    main()
