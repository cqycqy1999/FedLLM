from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass
from typing import Any

import torch


class Recorder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.round_file = os.path.join(self.output_dir, "round_metrics.jsonl")
        self.eval_file = os.path.join(self.output_dir, "eval_metrics.jsonl")

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
