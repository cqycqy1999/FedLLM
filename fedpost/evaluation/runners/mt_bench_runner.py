from __future__ import annotations

import os
import subprocess
import re


class MTBenchRunner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def _parse_score(stdout: str, model_id: str) -> float | None:
        """
        轻量解析 show_result 输出。
        """
        lines = stdout.splitlines()
        for line in lines:
            if model_id in line:
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                floats = []
                for x in nums:
                    try:
                        floats.append(float(x))
                    except Exception:
                        pass
                if floats:
                    return floats[-1]
        return None

    def run(
        self,
        model_path: str,
        model_id: str,
        judge_model: str = "gpt-4",
        parallel: int = 2,
    ) -> dict:
        gen_cmd = [
            "python", "-m", "fastchat.llm_judge.gen_model_answer",
            "--model-path", model_path,
            "--model-id", model_id,
        ]
        gen_proc = subprocess.run(gen_cmd, capture_output=True, text=True)

        judge_cmd = [
            "python", "-m", "fastchat.llm_judge.gen_judgment",
            "--model-list", model_id,
            "--parallel", str(parallel),
            "--judge-model", judge_model,
        ]
        judge_proc = subprocess.run(judge_cmd, capture_output=True, text=True)

        show_cmd = [
            "python", "-m", "fastchat.llm_judge.show_result",
            "--model-list", model_id,
        ]
        show_proc = subprocess.run(show_cmd, capture_output=True, text=True)
        parsed_score = self._parse_score(show_proc.stdout, model_id)

        answer_file = os.path.join("data", "mt_bench", "model_answer", f"{model_id}.jsonl")
        judgment_file = os.path.join("data", "mt_bench", "model_judgment", "gpt-4_single.jsonl")

        return {
            "gen_returncode": gen_proc.returncode,
            "judge_returncode": judge_proc.returncode,
            "show_returncode": show_proc.returncode,
            "gen_stdout": gen_proc.stdout,
            "gen_stderr": gen_proc.stderr,
            "judge_stdout": judge_proc.stdout,
            "judge_stderr": judge_proc.stderr,
            "show_stdout": show_proc.stdout,
            "show_stderr": show_proc.stderr,
            "answer_file": answer_file,
            "judgment_file": judgment_file,
            "mt_bench_score": parsed_score,
            "gen_cmd": gen_cmd,
            "judge_cmd": judge_cmd,
            "show_cmd": show_cmd,
        }