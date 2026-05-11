from __future__ import annotations

import json
import os
from typing import Any

import torch

from fedpost.data.io import load_records
from fedpost.evaluation.base import EvalResult
from fedpost.evaluation.metrics import (
    evaluate_humaneval_completion,
    score_finqa_prediction,
    score_medqa_prediction,
)


class PaperBenchmarkEvaluator:
    """Evaluator for the configured benchmarks."""

    SUPPORTED_TASKS = {"humaneval", "finqa", "medqa"}

    def __init__(self, cfg, tokenizer):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.tasks = [task.lower() for task in cfg.eval.tasks]
        unknown = sorted(set(self.tasks) - self.SUPPORTED_TASKS)
        if unknown:
            raise ValueError(f"Unsupported eval tasks: {unknown}")

    def evaluate(self, model, round_idx: int, model_artifacts: dict | None = None):
        model.eval()
        metrics: dict[str, float] = {}
        artifacts: dict[str, str] = {}
        details: dict[str, Any] = {}
        with torch.no_grad():
            for task in self.tasks:
                task_metrics, prediction_path = self._evaluate_task(model, task, round_idx)
                metrics.update(task_metrics)
                artifacts[f"{task}_predictions"] = prediction_path
                details[task] = {"predictions": prediction_path}

        if model_artifacts:
            artifacts.update({f"model/{k}": v for k, v in model_artifacts.items()})
        return EvalResult(round_idx=round_idx, metrics=metrics, artifacts=artifacts, details=details)

    def _evaluate_task(self, model, task: str, round_idx: int) -> tuple[dict[str, float], str]:
        records = self._load_task_records(task)
        if self.cfg.eval.max_samples is not None:
            records = records[: self.cfg.eval.max_samples]
        if not records:
            raise ValueError(f"No records found for eval task {task}")

        rows = []
        correct = 0
        for idx, record in enumerate(records):
            prompt = self._build_prompt(task, record)
            prediction = self._generate(model, prompt)
            is_correct, extra = self._score(task, record, prompt, prediction)
            correct += int(is_correct)
            rows.append({
                "idx": idx,
                "task": task,
                "prompt": prompt,
                "prediction": prediction,
                "correct": bool(is_correct),
                **extra,
            })

        metric_name = {
            "humaneval": "humaneval/pass_at_1",
            "finqa": "finqa/execution_accuracy",
            "medqa": "medqa/accuracy",
        }[task]
        metrics = {
            metric_name: correct / len(records),
            f"{task}/num_examples": float(len(records)),
        }
        return metrics, self._write_predictions(task, round_idx, rows)

    def _load_task_records(self, task: str) -> list[dict[str, Any]]:
        local_path = getattr(self.cfg.eval, f"{task}_path", None)
        if local_path:
            return _load_local_records(local_path, self.cfg.eval.file_type)

        dataset_dir = self.cfg.eval.dataset_dir
        if dataset_dir:
            dataset_name = {
                "humaneval": "humaneval_eval",
                "finqa": "finqa_eval",
                "medqa": "medqa_usmle_eval",
            }[task]
            candidate = os.path.join(dataset_dir, dataset_name)
            if os.path.exists(candidate):
                return _load_local_records(candidate, self.cfg.eval.file_type)

        from datasets import load_dataset

        if task == "humaneval":
            ds = load_dataset(self.cfg.eval.humaneval_dataset, split=self.cfg.eval.humaneval_split)
        elif task == "finqa":
            ds = load_dataset(self.cfg.eval.finqa_dataset, split=self.cfg.eval.finqa_split)
        else:
            ds = load_dataset(
                self.cfg.eval.medqa_dataset,
                name=self.cfg.eval.medqa_config,
                split=self.cfg.eval.medqa_split,
            )
        return [dict(row) for row in ds]

    def _build_prompt(self, task: str, record: dict[str, Any]) -> str:
        if task == "humaneval":
            return str(record["prompt"])
        if task == "finqa":
            context = record.get("text") or record.get("context") or record.get("table") or ""
            question = record.get("query") or record.get("question") or ""
            return (
                "Answer the financial question using the provided report. "
                "Return either a FinQA arithmetic program or the final numeric answer.\n\n"
                f"Report:\n{context}\n\nQuestion: {question}\nAnswer:"
            )

        question = record.get("question") or record.get("sent1") or ""
        options = _format_options(record.get("options") or record.get("choices") or record)
        return (
            "Answer the medical multiple-choice question. "
            "Return only one letter: A, B, C, or D.\n\n"
            f"Question: {question}\n{options}\nAnswer:"
        )

    def _generate(self, model, prompt: str) -> str:
        device = next(model.parameters()).device
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self._prompt_max_length(),
        ).to(device)
        generate_kwargs = {
            "max_new_tokens": self.cfg.eval.max_new_tokens,
            "do_sample": self.cfg.eval.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.cfg.eval.do_sample:
            generate_kwargs.update({
                "temperature": self.cfg.eval.temperature,
                "top_p": self.cfg.eval.top_p,
            })
        output_ids = model.generate(**inputs, **generate_kwargs)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def _prompt_max_length(self) -> int | None:
        if self.cfg.eval.prompt_max_length is not None:
            return self.cfg.eval.prompt_max_length
        if self.cfg.sft is not None:
            return self.cfg.sft.max_length
        model_max_length = getattr(self.tokenizer, "model_max_length", None)
        if model_max_length is None or model_max_length > 100000:
            return None
        return int(model_max_length)

    def _score(
        self,
        task: str,
        record: dict[str, Any],
        prompt: str,
        prediction: str,
    ) -> tuple[bool, dict[str, Any]]:
        if task == "humaneval":
            passed, detail = evaluate_humaneval_completion(
                prompt=prompt,
                completion=prediction,
                test=record["test"],
                entry_point=record["entry_point"],
                timeout=self.cfg.eval.humaneval_timeout,
            )
            return passed, {"task_id": record.get("task_id"), "error": "" if passed else detail}
        if task == "finqa":
            reference = record.get("answer")
            return score_finqa_prediction(prediction, reference), {"reference": reference}
        return score_medqa_prediction(prediction, record), {"reference": record.get("answer")}

    def _write_predictions(self, task: str, round_idx: int, rows: list[dict[str, Any]]) -> str:
        out_dir = os.path.join(self.cfg.output_dir, "eval_predictions", f"round_{round_idx + 1}")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{task}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path


def _load_local_records(path: str, file_type: str) -> list[dict[str, Any]]:
    if os.path.isdir(path):
        try:
            from datasets import load_from_disk

            return [dict(row) for row in load_from_disk(path)]
        except Exception:
            records = []
            for name in sorted(os.listdir(path)):
                if name.endswith((".jsonl", ".json")):
                    records.extend(_load_local_records(os.path.join(path, name), file_type))
            if records:
                return records
            raise
    if path.endswith(".json"):
        return load_records(path, "json")
    if path.endswith(".jsonl"):
        return load_records(path, "jsonl")
    return load_records(path, file_type)


def _format_options(options: Any) -> str:
    values = []
    if isinstance(options, dict) and all(key in options for key in ("ending0", "ending1", "ending2", "ending3")):
        values = [("ABCD"[idx], str(options[f"ending{idx}"])) for idx in range(4)]
    elif isinstance(options, list):
        for item in options:
            if isinstance(item, dict):
                key = str(item.get("key", "")).strip()
                value = item.get("value", item.get("text", ""))
                values.append((key, str(value)))
            else:
                values.append(("", str(item)))
    elif isinstance(options, dict):
        for key in sorted(options.keys()):
            values.append((str(key), str(options[key])))

    lines = []
    for idx, (key, value) in enumerate(values[:4]):
        label = key.upper() if len(key) == 1 and key.upper() in "ABCD" else "ABCD"[idx]
        lines.append(f"{label}. {value}")
    return "\n".join(lines)
