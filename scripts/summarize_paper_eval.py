from __future__ import annotations

import argparse
import json
import math
import os
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize benchmark metrics as mean/std over runs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Output directories containing eval_metrics.jsonl.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [read_last_eval(run_dir) for run_dir in args.runs]
    summary = summarize(rows)
    payload = {"num_runs": len(rows), "runs": args.runs, "metrics": summary}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


def read_last_eval(run_dir: str) -> dict:
    path = os.path.join(run_dir, "eval_metrics.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing eval metrics: {path}")
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    if last is None:
        raise ValueError(f"No eval rows found in {path}")
    return last.get("metrics", last)


def summarize(rows: list[dict]) -> dict[str, dict[str, float]]:
    numeric_keys = sorted({
        key
        for row in rows
        for key, value in row.items()
        if isinstance(value, (int, float))
    })
    out = {}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        mu = mean(values)
        var = sum((value - mu) ** 2 for value in values) / max(1, len(values) - 1)
        out[key] = {
            "mean": mu,
            "std": math.sqrt(var),
            "num_runs": len(values),
        }
    return out


if __name__ == "__main__":
    main()
