from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Optional


class LMEvalRunner:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _candidate_cmds(
        self,
        base_args: list[str],
    ) -> list[list[str]]:
        return [
            ["lm-eval", "run", *base_args],
            ["lm_eval", "run", *base_args],
            ["python", "-m", "lm_eval", "run", *base_args],
            ["lm-eval", *base_args],
            ["lm_eval", *base_args],
            ["python", "-m", "lm_eval", *base_args],
        ]

    def _build_base_args(
        self,
        model_backend: str,
        model_path: str,
        tasks: list[str],
        batch_size: str,
        device: Optional[str],
        model_args: dict[str, Any] | None,
        adapter_path: str | None,
        dtype: str | None,
        trust_remote_code: bool,
        num_fewshot: int | None,
        limit: int | float | str | None,
        log_samples: bool,
        include_path: str | None,
        use_cache: str | None,
        apply_chat_template: bool | str,
        fewshot_as_multiturn: bool | None,
        gen_kwargs: dict[str, Any] | str | None,
        seed: int | str | None,
        predict_only: bool,
    ) -> list[str]:
        if not tasks:
            raise ValueError("lm-eval tasks must not be empty.")

        resolved_model_args = dict(model_args or {})
        resolved_model_args.setdefault("pretrained", model_path)
        if adapter_path:
            resolved_model_args.setdefault("peft", adapter_path)
        if dtype:
            resolved_model_args.setdefault("dtype", dtype)
        if trust_remote_code:
            resolved_model_args.setdefault("trust_remote_code", True)

        base_args = [
            "--model", model_backend,
            "--model_args", *self._format_key_values(resolved_model_args),
            "--tasks", ",".join(tasks),
            "--batch_size", batch_size,
            "--output_path", self.output_dir,
        ]
        if device:
            base_args += ["--device", device]
        if num_fewshot is not None:
            base_args += ["--num_fewshot", str(num_fewshot)]
        if limit is not None:
            base_args += ["--limit", str(limit)]
        if log_samples:
            base_args += ["--log_samples"]
        if include_path:
            base_args += ["--include_path", include_path]
        if use_cache:
            base_args += ["--use_cache", use_cache]
        if apply_chat_template:
            base_args += ["--apply_chat_template"]
            if isinstance(apply_chat_template, str):
                base_args.append(apply_chat_template)
        if fewshot_as_multiturn is not None:
            base_args += ["--fewshot_as_multiturn", str(fewshot_as_multiturn).lower()]
        if gen_kwargs:
            if isinstance(gen_kwargs, str):
                base_args += ["--gen_kwargs", gen_kwargs]
            else:
                base_args += ["--gen_kwargs", *self._format_key_values(gen_kwargs)]
        if seed is not None:
            base_args += ["--seed", str(seed)]
        if predict_only:
            base_args += ["--predict_only"]
        return base_args

    @staticmethod
    def _format_key_values(values: dict[str, Any]) -> list[str]:
        formatted = []
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, bool):
                value = "True" if value else "False"
            formatted.append(f"{key}={value}")
        return formatted

    def _parse_result_json(self) -> tuple[str | None, dict | None]:
        candidates = []
        for root, _, files in os.walk(self.output_dir):
            for fname in files:
                if fname.endswith(".json"):
                    path = os.path.join(root, fname)
                    candidates.append(path)

        candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
        fallback_path = None
        fallback_payload = None
        for candidate in candidates:
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                continue
            if fallback_payload is None:
                fallback_path = candidate
                fallback_payload = payload
            if isinstance(payload, dict) and "results" in payload:
                return candidate, payload
        return fallback_path, fallback_payload

    def run(
        self,
        model_path: str,
        tasks: list[str],
        batch_size: str = "auto",
        device: Optional[str] = None,
        model_backend: str = "hf",
        model_args: dict[str, Any] | None = None,
        adapter_path: str | None = None,
        dtype: str | None = None,
        trust_remote_code: bool = False,
        num_fewshot: int | None = None,
        limit: int | float | str | None = None,
        log_samples: bool = False,
        include_path: str | None = None,
        use_cache: str | None = None,
        apply_chat_template: bool | str = False,
        fewshot_as_multiturn: bool | None = None,
        gen_kwargs: dict[str, Any] | str | None = None,
        seed: int | str | None = None,
        predict_only: bool = False,
        timeout: int | None = None,
    ) -> dict:
        last_proc = None
        last_cmd = None
        attempts = []

        base_args = self._build_base_args(
            model_backend=model_backend,
            model_path=model_path,
            tasks=tasks,
            batch_size=batch_size,
            device=device,
            model_args=model_args,
            adapter_path=adapter_path,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=log_samples,
            include_path=include_path,
            use_cache=use_cache,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            gen_kwargs=gen_kwargs,
            seed=seed,
            predict_only=predict_only,
        )

        for cmd in self._candidate_cmds(base_args):
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            except FileNotFoundError as exc:
                attempts.append({"cmd": cmd, "returncode": 127, "stderr": str(exc)})
                continue
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                proc = subprocess.CompletedProcess(cmd, returncode=124, stdout=stdout, stderr=stderr)
            last_proc = proc
            last_cmd = cmd
            attempts.append({
                "cmd": cmd,
                "returncode": proc.returncode,
                "stderr_tail": proc.stderr[-2000:],
            })
            if proc.returncode == 0:
                break

        result_path, parsed = self._parse_result_json()

        return {
            "returncode": last_proc.returncode if last_proc else 1,
            "stdout": last_proc.stdout if last_proc else "",
            "stderr": last_proc.stderr if last_proc else "",
            "result_path": result_path,
            "parsed": parsed,
            "cmd": last_cmd,
            "attempts": attempts,
        }
