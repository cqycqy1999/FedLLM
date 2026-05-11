from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTSample:
    prompt: str
    response: str
    metadata: dict | None = None
