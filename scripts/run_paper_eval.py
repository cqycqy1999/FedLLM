from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedpost.evaluation.paper import PaperBenchmarkEvaluator
from fedpost.models.loader import HFModelManager
from fedpost.utils.config import ConfigLoader
from fedpost.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation: HumanEval, FinQA, MedQA-USMLE.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default=None, help="Optional merged model path to evaluate.")
    parser.add_argument("--adapter-dir", default=None, help="Optional PEFT adapter directory to load on the configured base model.")
    parser.add_argument("--tasks", nargs="+", choices=["humaneval", "finqa", "medqa"], default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--round-idx", type=int, default=0)
    args = parser.parse_args()

    cfg = ConfigLoader.from_yaml(args.config)
    if args.tasks:
        cfg.eval.tasks = args.tasks
    if args.max_samples is not None:
        cfg.eval.max_samples = args.max_samples
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    ConfigLoader.validate(cfg)
    if not cfg.eval.tasks:
        raise ValueError("No eval tasks configured. Set eval.tasks in YAML or pass --tasks.")

    if args.model_path is not None:
        cfg.model.model_name_or_path = args.model_path
        cfg.model.tokenizer_name_or_path = args.model_path
        cfg.peft.method = "none"
        cfg.peft.target_modules = None
    if args.adapter_dir is not None:
        cfg.peft.method = "none"
        cfg.peft.target_modules = None

    set_seed(cfg.seed)
    manager = HFModelManager(cfg)
    bundle = manager.build()
    if args.adapter_dir is not None:
        from peft import PeftModel

        bundle.model = PeftModel.from_pretrained(bundle.model, args.adapter_dir)

    evaluator = PaperBenchmarkEvaluator(cfg, bundle.tokenizer)
    result = evaluator.evaluate(
        bundle.model,
        round_idx=args.round_idx,
        model_artifacts={
            "model_path": args.model_path or cfg.model.model_name_or_path,
            "adapter_dir": args.adapter_dir or "",
        },
    )
    print(json.dumps({
        "round_idx": result.round_idx,
        "metrics": result.metrics,
        "artifacts": result.artifacts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
