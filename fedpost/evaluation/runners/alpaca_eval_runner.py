from __future__ import annotations

import json
import os
import subprocess


class AlpacaEvalRunner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self, model_outputs: list[dict], annotators_config: str) -> dict:
        outputs_path = os.path.join(self.output_dir, "alpaca_eval_outputs.json")
        with open(outputs_path, "w", encoding="utf-8") as f:
            json.dump(model_outputs, f, ensure_ascii=False, indent=2)

        result_dir = os.path.join(self.output_dir, "alpaca_eval_result")
        cmd = [
            "alpaca_eval",
            "--model_outputs", outputs_path,
            "--annotators_config", annotators_config,
            "--output_path", result_dir,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "result_dir": result_dir,
            "outputs_path": outputs_path,
        }