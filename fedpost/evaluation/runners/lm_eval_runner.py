from __future__ import annotations

import json
import os
import subprocess


class LMEvalRunner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        model_path: str,
        tasks: list[str],
        batch_size: str = "auto",
    ) -> dict:
        out_path = os.path.join(self.output_dir, "lm_eval_result.json")
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", ",".join(tasks),
            "--batch_size", batch_size,
            "--output_path", out_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        parsed = None
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "result_path": out_path,
            "parsed": parsed,
        }