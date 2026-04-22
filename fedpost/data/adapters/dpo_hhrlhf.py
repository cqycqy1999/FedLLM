from __future__ import annotations

from fedpost.data.adapters.base import BaseDatasetAdapter
from fedpost.data.processors import DPOSample
from fedpost.utils.registry import Registry


def _normalize_hh_prompt(prompt: str) -> str:
    prompt = prompt.replace("\n\nHuman:", "\n\nUser:")
    prompt = prompt.replace("\n\nAssistant:", "\n\nAssistant:")
    if prompt.startswith("Human:"):
        prompt = "User:" + prompt[len("Human:"):]
    return prompt.strip()


def _split_hh_transcript(text: str) -> tuple[str | None, str | None]:
    """
    HH-RLHF 常见格式：
    Human: ...
    
    Assistant: ...
    
    Human: ...
    
    Assistant: final_answer
    """
    if not text or not isinstance(text, str):
        return None, None

    markers = ["\n\nAssistant:", "Assistant:"]
    split_idx = -1
    split_marker = None
    for m in markers:
        idx = text.rfind(m)
        if idx > split_idx:
            split_idx = idx
            split_marker = m

    if split_idx < 0 or split_marker is None:
        return None, None

    prompt = text[:split_idx].strip()
    answer = text[split_idx + len(split_marker):].strip()

    if not prompt or not answer:
        return None, None

    prompt = _normalize_hh_prompt(prompt)
    return prompt, answer


@Registry.register("dataset_adapter", "Anthropic/hh-rlhf")
class HHRLHFAdapter(BaseDatasetAdapter):
    def to_sft_sample(self, record: dict):
        return None

    def to_dpo_sample(self, record: dict) -> DPOSample | None:
        chosen_text = record.get("chosen")
        rejected_text = record.get("rejected")
        if not chosen_text or not rejected_text:
            return None

        chosen_prompt, chosen_resp = _split_hh_transcript(chosen_text)
        rejected_prompt, rejected_resp = _split_hh_transcript(rejected_text)

        prompt = chosen_prompt or rejected_prompt
        if not prompt or not chosen_resp or not rejected_resp:
            return None

        return DPOSample(
            prompt=prompt,
            chosen=chosen_resp,
            rejected=rejected_resp,
            metadata={"source_dataset": "Anthropic/hh-rlhf"},
        )