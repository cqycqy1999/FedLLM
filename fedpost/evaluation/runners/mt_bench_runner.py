from __future__ import annotations

import os
import subprocess


class MTBenchRunner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self,
        model_path: str,
        model_id: str,
        judge_model: str = "gpt-4",
    ) -> dict:
        answer_file = os.path.join(self.output_dir, f"{model_id}_answers.jsonl")

        gen_cmd = [
            "python",
            "-m",
            "fastchat.llm_judge.gen_model_answer",
            "--model-path", model_path,
            "--model-id", model_id,
            "--answer-file", answer_file,
        ]
        gen_proc = subprocess.run(gen_cmd, capture_output=True, text=True)

        judge_cmd = [
            "python",
            "-m",
            "fastchat.llm_judge.gen_judgment",
            "--model-list", model_id,
            "--judge-model", judge_model,
        ]
        judge_proc = subprocess.run(judge_cmd, capture_output=True, text=True)

        return {
            "gen_returncode": gen_proc.returncode,
            "judge_returncode": judge_proc.returncode,
            "gen_stdout": gen_proc.stdout,
            "gen_stderr": gen_proc.stderr,
            "judge_stdout": judge_proc.stdout,
            "judge_stderr": judge_proc.stderr,
            "answer_file": answer_file,
        }