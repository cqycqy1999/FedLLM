from __future__ import annotations

import math
import multiprocessing as mp
import re
import signal
from contextlib import contextmanager
from typing import Any


_CHOICE_RE = re.compile(r"\b([A-Da-d])\b")
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_PROGRAM_CALL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)")


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().strip().split())


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_choice(text: str) -> str | None:
    normalized = text.strip()
    answer_match = re.search(
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-D])\)?",
        normalized,
        re.IGNORECASE,
    )
    if answer_match:
        return answer_match.group(1).upper()

    match = _CHOICE_RE.search(normalized)
    return match.group(1).upper() if match else None


def extract_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text.replace("$", " "))
    if not matches:
        return None
    token = matches[-1]
    is_percent = token.endswith("%")
    token = token.rstrip("%").replace(",", "")
    try:
        value = float(token)
    except ValueError:
        return None
    return value / 100.0 if is_percent else value


def numeric_equal(prediction: Any, reference: Any, rel_tol: float = 1e-4, abs_tol: float = 1e-4) -> bool:
    pred = prediction if isinstance(prediction, (int, float)) else extract_number(str(prediction))
    ref = reference if isinstance(reference, (int, float)) else extract_number(str(reference))
    if pred is None or ref is None:
        return normalize_text(prediction) == normalize_text(reference)
    return math.isclose(float(pred), float(ref), rel_tol=rel_tol, abs_tol=abs_tol)


def execute_finqa_program(program: str) -> float | None:
    """Execute the small FinQA arithmetic DSL when a model emits a program."""
    values: list[float] = []
    for raw_step in _split_program_steps(program):
        value = _execute_finqa_step(raw_step.strip(), values)
        if value is None:
            continue
        values.append(value)
    return values[-1] if values else None


def score_finqa_prediction(prediction: str, reference: Any) -> bool:
    program_value = execute_finqa_program(prediction)
    if program_value is not None:
        return numeric_equal(program_value, reference)
    return numeric_equal(prediction, reference)


def score_medqa_prediction(prediction: str, record: dict[str, Any]) -> bool:
    pred_choice = extract_choice(prediction)
    gold_choice = _gold_choice(record)
    if pred_choice and gold_choice:
        return pred_choice == gold_choice

    gold_answer = record.get("answer")
    if isinstance(gold_answer, list) and gold_answer:
        gold_answer = gold_answer[0]
    return normalize_text(prediction).startswith(normalize_text(gold_answer))


def evaluate_humaneval_completion(
    prompt: str,
    completion: str,
    test: str,
    entry_point: str,
    timeout: float = 3.0,
) -> tuple[bool, str]:
    completion = _clean_humaneval_completion(prompt, completion)
    code = f"{prompt}{completion}\n{test}\ncheck({entry_point})\n"
    ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_humaneval_worker, args=(code, queue, timeout))
    process.start()
    process.join(timeout + 1.0)
    if process.is_alive():
        process.kill()
        process.join()
        return False, "timeout"
    if queue.empty():
        return False, "no_result"
    status, detail = queue.get()
    return bool(status), str(detail)


def _clean_humaneval_completion(prompt: str, completion: str) -> str:
    text = strip_code_fences(completion)
    if text.startswith(prompt):
        text = text[len(prompt):]
    lines = text.splitlines()
    kept = []
    for line in lines:
        if kept and line and not line.startswith((" ", "\t")) and re.match(r"(def|class)\s+", line):
            break
        if line.strip().startswith(("if __name__", "print(")):
            break
        kept.append(line)
    return "\n".join(kept).rstrip() + "\n"


def _humaneval_worker(code: str, queue, timeout: float) -> None:
    try:
        with _time_limit(timeout):
            namespace: dict[str, Any] = {}
            exec(code, namespace)
        queue.put((True, "passed"))
    except BaseException as exc:  # official HumanEval treats any exception as failure.
        queue.put((False, f"{type(exc).__name__}: {exc}"))


@contextmanager
def _time_limit(seconds: float):
    def handler(signum, frame):
        raise TimeoutError("timed out")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_handler)


def _split_program_steps(program: str) -> list[str]:
    text = strip_code_fences(program)
    if "Final answer" in text:
        text = text.split("Final answer", 1)[0]
    parts: list[str] = []
    depth = 0
    current = []
    for ch in text.replace("\n", ";"):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch in {";", ","} and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    calls = [part for part in parts if _PROGRAM_CALL_RE.search(part)]
    return calls or [text]


def _execute_finqa_step(step: str, previous_values: list[float]) -> float | None:
    match = _PROGRAM_CALL_RE.search(step)
    if not match:
        return extract_number(step)
    op = match.group(1).lower()
    args = [_resolve_arg(arg, previous_values) for arg in _split_args(match.group(2))]
    if any(arg is None for arg in args):
        return None
    nums = [float(arg) for arg in args if arg is not None]
    if not nums:
        return None

    if op in {"add", "sum"}:
        return sum(nums)
    if op in {"subtract", "sub", "minus", "diff"} and len(nums) >= 2:
        return nums[0] - nums[1]
    if op in {"multiply", "mul", "times", "product"}:
        result = 1.0
        for num in nums:
            result *= num
        return result
    if op in {"divide", "div"} and len(nums) >= 2 and nums[1] != 0:
        return nums[0] / nums[1]
    if op in {"average", "avg", "mean"}:
        return sum(nums) / len(nums)
    if op in {"greater", "greater_than", "max"} and len(nums) >= 2:
        return max(nums)
    if op in {"less", "less_than", "min"} and len(nums) >= 2:
        return min(nums)
    if op in {"exp"} and len(nums) >= 2:
        return nums[0] ** nums[1]
    return nums[-1]


def _split_args(text: str) -> list[str]:
    args = []
    depth = 0
    current = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _resolve_arg(arg: str, previous_values: list[float]) -> float | None:
    arg = arg.strip().strip("\"'")
    if arg.startswith("#"):
        try:
            return previous_values[int(arg[1:])]
        except (ValueError, IndexError):
            return None
    nested = _execute_finqa_step(arg, previous_values)
    if nested is not None:
        return nested
    return extract_number(arg)


def _gold_choice(record: dict[str, Any]) -> str | None:
    for key in ("answer_idx", "label", "gold", "target"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, int):
            return "ABCD"[value] if 0 <= value < 4 else None
        text = str(value).strip()
        if len(text) == 1 and text.upper() in "ABCD":
            return text.upper()

    answer = record.get("answer")
    if isinstance(answer, list) and answer:
        answer = answer[0]
    options = record.get("options") or record.get("choices")
    if answer is None or options is None:
        return None

    for idx, option in enumerate(_option_values(options)):
        if normalize_text(option) == normalize_text(answer):
            return "ABCD"[idx] if idx < 4 else None
    return None


def _option_values(options: Any) -> list[str]:
    if isinstance(options, dict):
        return [str(options[key]) for key in sorted(options.keys())]
    if isinstance(options, list):
        values = []
        for item in options:
            if isinstance(item, dict):
                values.append(str(item.get("value", item.get("text", item))))
            else:
                values.append(str(item))
        return values
    return []
