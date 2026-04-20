from __future__ import annotations

import json
from typing import Any


def load_json(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of records.")
    return data


def load_jsonl(path: str) -> list[dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_idx}: {e}") from e
    return records


def load_records(path: str, file_type: str) -> list[dict[str, Any]]:
    if file_type == "json":
        return load_json(path)
    if file_type == "jsonl":
        return load_jsonl(path)
    raise ValueError(f"Unsupported file_type: {file_type}")