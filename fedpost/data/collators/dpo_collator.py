from __future__ import annotations

import torch


class DPOCollator:
    def __init__(self, tokenizer, max_length: int, max_prompt_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def _build_texts(self, batch, field_name: str):
        prompt_texts = [f"User: {x.prompt}\nAssistant:" for x in batch]
        full_texts = [f"User: {x.prompt}\nAssistant: {getattr(x, field_name)}" for x in batch]
        return prompt_texts, full_texts

    def _tokenize_with_response_mask(self, prompt_texts, full_texts):
        encoded_full = self.tokenizer(
            full_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded_prompt = self.tokenizer(
            prompt_texts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors=None,
        )

        response_mask = torch.zeros_like(encoded_full["input_ids"], dtype=torch.long)
        for i, prompt_ids in enumerate(encoded_prompt["input_ids"]):
            prompt_len = min(len(prompt_ids), response_mask.shape[1])
            response_mask[i, prompt_len:] = 1
            if "attention_mask" in encoded_full:
                response_mask[i] = response_mask[i] * encoded_full["attention_mask"][i]

        return {
            "input_ids": encoded_full["input_ids"],
            "attention_mask": encoded_full["attention_mask"],
            "response_mask": response_mask,
        }

    def __call__(self, batch: list) -> dict:
        chosen_prompt, chosen_full = self._build_texts(batch, "chosen")
        rejected_prompt, rejected_full = self._build_texts(batch, "rejected")

        chosen = self._tokenize_with_response_mask(chosen_prompt, chosen_full)
        rejected = self._tokenize_with_response_mask(rejected_prompt, rejected_full)

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "chosen_response_mask": chosen["response_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
            "rejected_response_mask": rejected["response_mask"],
        }