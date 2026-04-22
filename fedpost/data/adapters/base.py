from __future__ import annotations

from abc import ABC, abstractmethod


class BaseDatasetAdapter(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    def to_sft_sample(self, record: dict):
        raise NotImplementedError

    def to_dpo_sample(self, record: dict):
        raise NotImplementedError


def format_role(role: str) -> str:
    role = (role or "").strip().lower()
    if role == "user":
        return "User"
    if role == "assistant":
        return "Assistant"
    if role == "system":
        return "System"
    return role.capitalize() if role else "Unknown"


def render_messages_as_prompt(messages: list[dict]) -> str:
    """
    把 messages[:-1] 渲染成统一 prompt 文本。
    """
    parts = []
    for m in messages:
        role = format_role(m.get("role", "user"))
        content = str(m.get("content", "")).strip()
        if not content:
            continue
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts).strip()


def extract_last_assistant_message(messages: list[dict]) -> str | None:
    if not messages:
        return None
    last = messages[-1]
    if str(last.get("role", "")).lower() != "assistant":
        return None
    content = str(last.get("content", "")).strip()
    return content or None