from __future__ import annotations

import argparse
import json
import os
from typing import Any

from fedpost.evaluation.runners.lm_eval_runner import LMEvalRunner


def run_lm_eval_cli(args) -> dict[str, Any]:
    os.makedirs(args.output_dir, exist_ok=True)
    runner = LMEvalRunner(args.output_dir)
    result = runner.run(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        tasks=_split_csv(args.tasks),
        batch_size=args.batch_size,
        device=args.device,
        model_backend=args.model_backend,
        model_args=_parse_key_values(args.model_arg),
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        log_samples=args.log_samples,
        include_path=args.include_path,
        use_cache=args.use_cache,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        seed=args.seed,
        predict_only=args.predict_only,
        timeout=args.timeout,
    )
    summary = {
        "protocol": "standard_lm_eval",
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "tasks": _split_csv(args.tasks),
        "returncode": result["returncode"],
        "cmd": result["cmd"],
        "result_path": result["result_path"],
        "metrics": _flatten_lm_eval_metrics(result.get("parsed")) if result["returncode"] == 0 else {},
        "attempts": result.get("attempts", []),
    }
    summary_path = os.path.join(args.output_dir, "standard_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved standard eval summary: {summary_path}")
    if result["returncode"] != 0:
        stdout_path = os.path.join(args.output_dir, "lm_eval_stdout.txt")
        stderr_path = os.path.join(args.output_dir, "lm_eval_stderr.txt")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(result.get("stdout", ""))
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(result.get("stderr", ""))
        print(f"lm-eval failed; stdout={stdout_path}, stderr={stderr_path}")
        raise SystemExit(result["returncode"])
    return summary


def _flatten_lm_eval_metrics(parsed: dict | None) -> dict[str, float]:
    if not parsed or "results" not in parsed:
        return {}
    metrics = {}
    for task_name, task_metrics in parsed["results"].items():
        for metric_name, metric_value in task_metrics.items():
            if isinstance(metric_value, (float, int)):
                metrics[f"{task_name}/{metric_name}"] = float(metric_value)
    return metrics


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_key_values(items: list[str] | None) -> dict[str, Any]:
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected --model_arg key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Run standard LLM evaluation via lm-evaluation-harness.")
    parser.add_argument("--model_path", required=True, help="Merged model path or base model name/path.")
    parser.add_argument("--adapter_path", default=None, help="Optional PEFT adapter dir for adapter-only eval.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated lm-eval task names, e.g. mmlu,gsm8k.")
    parser.add_argument("--model_backend", default="hf", choices=["hf", "vllm", "sglang"])
    parser.add_argument("--model_arg", action="append", default=[], help="Extra lm-eval model arg, repeated key=value.")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=None)
    parser.add_argument("--limit", default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--include_path", default=None)
    parser.add_argument("--use_cache", default=None)
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--fewshot_as_multiturn", type=lambda value: value.lower() == "true", default=None)
    parser.add_argument("--gen_kwargs", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--predict_only", action="store_true")
    parser.add_argument("--timeout", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    run_lm_eval_cli(parse_args())


if __name__ == "__main__":
    main()
