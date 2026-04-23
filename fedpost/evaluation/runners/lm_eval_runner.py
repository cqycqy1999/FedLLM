from __future__ import annotations

import json
import os
import subprocess
from typing import Optional


class LMEvalRunner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _candidate_cmds(
        self,
        model_path: str,
        tasks: list[str],
        batch_size: str,
        device: Optional[str],
    ) -> list[list[str]]:
        task_str = ",".join(tasks)
        base_args = [
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", task_str,
            "--batch_size", batch_size,
            "--output_path", self.output_dir,
        ]
        if device:
            base_args += ["--device", device]

        new_cli = ["lm_eval", "run", *base_args]
        old_cli = ["lm_eval", *base_args]
        module_cli = ["python", "-m", "lm_eval", *base_args]

        return [new_cli, old_cli, module_cli]

    def run(
        self,
        model_path: str,
        tasks: list[str],
        batch_size: str = "auto",
        device: Optional[str] = None,
    ) -> dict:
        last_proc = None
        last_cmd = None

        for cmd in self._candidate_cmds(model_path, tasks, batch_size, device):
            proc = subprocess.run(cmd, capture_output=True, text=True)
            last_proc = proc
            last_cmd = cmd
            if proc.returncode == 0:
                break

        parsed = None
        result_path = None

        if os.path.isdir(self.output_dir):
            for fname in os.listdir(self.output_dir):
                if fname.endswith(".json"):
                    candidate = os.path.join(self.output_dir, fname)
                    try:
                        with open(candidate, "r", encoding="utf-8") as f:
                            parsed = json.load(f)
                        result_path = candidate
                        break
                    except Exception:
                        pass

        return {
            "returncode": last_proc.returncode if last_proc else 1,
            "stdout": last_proc.stdout if last_proc else "",
            "stderr": last_proc.stderr if last_proc else "",
            "result_path": result_path,
            "parsed": parsed,
            "cmd": last_cmd,
        }