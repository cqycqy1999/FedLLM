from __future__ import annotations

import csv
import json
import os
from dataclasses import fields, is_dataclass
from typing import Any

import torch


def _flatten(prefix: str, obj: dict | None) -> dict:
    if not obj:
        return {}
    out = {}
    for k, v in obj.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(f"{key}/", v))
        else:
            out[key] = v
    return out


class Recorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.round_file = os.path.join(self.output_dir, "round_metrics.jsonl")
        self.eval_file = os.path.join(self.output_dir, "eval_metrics.jsonl")
        self.summary_file = os.path.join(self.output_dir, "summary.jsonl")
        self.summary_csv = os.path.join(self.output_dir, "summary.csv")
        self.best_round_file = os.path.join(self.output_dir, "best_round.json")

    def record_round(self, round_idx: int, round_metrics: dict, client_results: list) -> None:
        payload = {
            "round_idx": round_idx,
            "round_metrics": self._to_jsonable(round_metrics),
            "client_results": [
                self._serialize_client_result(r) for r in client_results
            ],
        }
        with open(self.round_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_eval(self, eval_result) -> None:
        payload = self._to_jsonable(eval_result)
        with open(self.eval_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def record_round_summary(
        self,
        round_idx: int,
        round_metrics: dict,
        eval_result=None,
        model_artifacts: dict | None = None,
        primary_metric: str | None = None,
    ) -> None:
        row = {
            "round": round_idx + 1,
            **_flatten("train/", round_metrics),
        }

        if eval_result is not None:
            row.update(_flatten("eval/", getattr(eval_result, "metrics", {})))
            row.update(_flatten("artifact/", getattr(eval_result, "artifacts", {})))

        if model_artifacts:
            row.update(_flatten("export/", model_artifacts))

        with open(self.summary_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self._rewrite_summary_csv()
        if primary_metric:
            self._update_best_round(primary_metric)

    def _read_summary_rows(self) -> list[dict]:
        rows = []
        if not os.path.exists(self.summary_file):
            return rows
        with open(self.summary_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _rewrite_summary_csv(self) -> None:
        rows = self._read_summary_rows()
        if not rows:
            return

        fieldnames = sorted({k for row in rows for k in row.keys()})
        with open(self.summary_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _update_best_round(self, primary_metric: str) -> None:
        rows = self._read_summary_rows()
        candidates = []
        for row in rows:
            if primary_metric in row and isinstance(row[primary_metric], (int, float)):
                candidates.append(row)

        if not candidates:
            return

        best = max(candidates, key=lambda x: x[primary_metric])
        with open(self.best_round_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "primary_metric": primary_metric,
                    "best_round": best.get("round"),
                    "best_value": best.get(primary_metric),
                    "row": best,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def save_config(self, cfg) -> None:
        path = os.path.join(self.output_dir, "config_snapshot.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(cfg))

    def _serialize_client_result(self, result: Any) -> Any:
        if is_dataclass(result):
            payload = {
                field.name: getattr(result, field.name)
                for field in fields(result)
                if field.name != "update"
            }
            payload["update_summary"] = self._summarize_update(getattr(result, "update", {}))
            return self._to_jsonable(payload)

        if isinstance(result, dict):
            payload = {k: v for k, v in result.items() if k != "update"}
            payload["update_summary"] = self._summarize_update(result.get("update", {}))
            return self._to_jsonable(payload)

        return self._to_jsonable(result)

    def _summarize_update(self, update: Any) -> dict[str, Any]:
        if not isinstance(update, dict):
            return {"type": type(update).__name__}

        num_tensors = 0
        total_numel = 0
        total_bytes = 0

        for value in update.values():
            if isinstance(value, torch.Tensor):
                num_tensors += 1
                total_numel += value.numel()
                total_bytes += value.element_size() * value.numel()

        return {
            "num_entries": len(update),
            "num_tensors": num_tensors,
            "total_numel": total_numel,
            "total_bytes": total_bytes,
            "sample_keys": list(update.keys())[:10],
        }

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return {
                field.name: self._to_jsonable(getattr(value, field.name))
                for field in fields(value)
            }

        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            return tensor.item() if tensor.ndim == 0 else tensor.tolist()

        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (TypeError, ValueError):
                pass

        return value
