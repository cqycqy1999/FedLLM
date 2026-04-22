from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SFTSample:
    prompt: str
    response: str
    metadata: dict | None = None


@dataclass
class DPOSample:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict | None = None


def build_sft_sample_from_hf(record: dict, dataset_name: str) -> SFTSample | None:
    if dataset_name == "databricks/databricks-dolly-15k":
        instruction = record.get("instruction", "")
        context = record.get("context", "")
        response = record.get("response", "")

        if not instruction or not response:
            return None

        prompt = instruction.strip()
        if context and str(context).strip():
            prompt = f"{instruction.strip()}\n\nContext:\n{str(context).strip()}"

        return SFTSample(
            prompt=prompt,
            response=str(response).strip(),
            metadata={"category": record.get("category", None)},
        )

    raise ValueError(f"Unsupported SFT HF dataset: {dataset_name}")


def _extract_prompt_from_hh(text: str) -> tuple[str, str]:
    """
    HH-RLHF style text often looks like:
    Human: ...
    Assistant: ...
    We split off the last assistant response as completion.
    """
    marker = "\n\nAssistant:"
    if marker not in text:
        return "", text
    idx = text.rfind(marker)
    prompt = text[:idx].strip()
    answer = text[idx + len(marker):].strip()
    return prompt, answer


def build_dpo_sample_from_hf(record: dict, dataset_name: str) -> DPOSample | None:
    if dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        # Dataset card explicitly recommends chosen/rejected for DPO.
        chosen = record.get("chosen")
        rejected = record.get("rejected")

        if isinstance(chosen, list) and len(chosen) > 0:
            # messages format -> use final assistant content
            chosen_text = chosen[-1]["content"]
            prompt = "\n".join([m["content"] for m in chosen[:-1] if m["role"] != "assistant"]).strip()
        else:
            chosen_text = None
            prompt = None

        if isinstance(rejected, list) and len(rejected) > 0:
            rejected_text = rejected[-1]["content"]
            if prompt is None:
                prompt = "\n".join([m["content"] for m in rejected[:-1] if m["role"] != "assistant"]).strip()
        else:
            rejected_text = None

        if prompt and chosen_text and rejected_text:
            return DPOSample(
                prompt=prompt,
                chosen=chosen_text.strip(),
                rejected=rejected_text.strip(),
                metadata={
                    "source": dataset_name,
                    "score_chosen": record.get("score_chosen"),
                    "score_rejected": record.get("score_rejected"),
                },
            )

        # fallback if prompt/chosen/rejected plain text columns exist
        prompt = record.get("prompt")
        chosen_text = record.get("chosen")
        rejected_text = record.get("rejected")
        if prompt and chosen_text and rejected_text:
            return DPOSample(
                prompt=str(prompt).strip(),
                chosen=str(chosen_text).strip(),
                rejected=str(rejected_text).strip(),
                metadata={"source": dataset_name},
            )
        return None

    if dataset_name == "Anthropic/hh-rlhf":
        chosen = record.get("chosen")
        rejected = record.get("rejected")
        if not chosen or not rejected:
            return None

        prompt_c, chosen_resp = _extract_prompt_from_hh(chosen)
        prompt_r, rejected_resp = _extract_prompt_from_hh(rejected)
        prompt = prompt_c or prompt_r
        if not prompt or not chosen_resp or not rejected_resp:
            return None

        return DPOSample(
            prompt=prompt,
            chosen=chosen_resp,
            rejected=rejected_resp,
            metadata={"source": dataset_name},
        )

    raise ValueError(f"Unsupported DPO HF dataset: {dataset_name}")