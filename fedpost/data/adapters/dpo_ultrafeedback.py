from __future__ import annotations

from fedpost.data.adapters.base import (
    BaseDatasetAdapter,
    extract_last_assistant_message,
    render_messages_as_prompt,
)
from fedpost.data.processors import DPOSample
from fedpost.utils.registry import Registry


def _parse_messages_variant(messages_obj) -> tuple[str | None, str | None]:
    """
    输入形如:
    [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
    返回:
      prompt, assistant_response
    """
    if not isinstance(messages_obj, list) or len(messages_obj) == 0:
        return None, None

    response = extract_last_assistant_message(messages_obj)
    if response is None:
        return None, None

    prompt_messages = messages_obj[:-1]
    prompt = render_messages_as_prompt(prompt_messages)
    if not prompt:
        return None, None

    return prompt, response


@Registry.register("dataset_adapter", "HuggingFaceH4/ultrafeedback_binarized")
class UltraFeedbackBinarizedDPOAdapter(BaseDatasetAdapter):
    def to_sft_sample(self, record: dict):
        return None

    def to_dpo_sample(self, record: dict) -> DPOSample | None:
        chosen = record.get("chosen")
        rejected = record.get("rejected")

        # 1) chosen/rejected 为 messages list
        chosen_prompt, chosen_resp = _parse_messages_variant(chosen)
        rejected_prompt, rejected_resp = _parse_messages_variant(rejected)

        if chosen_prompt and chosen_resp and rejected_prompt and rejected_resp:
            prompt = chosen_prompt if chosen_prompt else rejected_prompt
            return DPOSample(
                prompt=prompt,
                chosen=chosen_resp,
                rejected=rejected_resp,
                metadata={
                    "source_dataset": "HuggingFaceH4/ultrafeedback_binarized",
                    "score_chosen": record.get("score_chosen"),
                    "score_rejected": record.get("score_rejected"),
                },
            )

        # 2) fallback: plain text columns
        prompt = record.get("prompt")
        if prompt and chosen and rejected and isinstance(chosen, str) and isinstance(rejected, str):
            return DPOSample(
                prompt=str(prompt).strip(),
                chosen=str(chosen).strip(),
                rejected=str(rejected).strip(),
                metadata={
                    "source_dataset": "HuggingFaceH4/ultrafeedback_binarized",
                    "score_chosen": record.get("score_chosen"),
                    "score_rejected": record.get("score_rejected"),
                },
            )

        return None