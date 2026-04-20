from __future__ import annotations

import torch


class SFTCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list) -> dict:
        prompt_texts = [f"User: {x.prompt}\nAssistant:" for x in batch]
        full_texts = [f"User: {x.prompt}\nAssistant: {x.response}" for x in batch]

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
            max_length=self.max_length,
            return_tensors=None,
        )

        labels = encoded_full["input_ids"].clone()

        for i, prompt_ids in enumerate(encoded_prompt["input_ids"]):
            prompt_len = min(len(prompt_ids), labels.shape[1])
            labels[i, :prompt_len] = -100

        encoded_full["labels"] = labels
        return encoded_full