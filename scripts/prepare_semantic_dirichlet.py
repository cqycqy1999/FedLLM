from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Iterable


DEFAULT_LABELS = ["code", "finance", "medical", "math", "general"]

KEYWORDS = {
    "code": ["code", "python", "java", "function", "algorithm", "bug", "compile", "class", "api"],
    "finance": ["finance", "stock", "revenue", "cash", "loan", "interest", "market", "asset", "bank"],
    "medical": ["patient", "medical", "diagnosis", "treatment", "disease", "symptom", "drug", "doctor"],
    "math": ["calculate", "equation", "proof", "probability", "integer", "geometry", "solve", "number"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create semantic pseudo-labels and Dirichlet client shards.")
    parser.add_argument("--input", required=True, help="Local .json or .jsonl instruction data.")
    parser.add_argument("--output-dir", required=True, help="Directory for client_*.jsonl shards.")
    parser.add_argument("--num-clients", type=int, default=14)
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet concentration. Smaller means more non-IID.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--instruction-field", default="instruction")
    parser.add_argument("--input-field", default="input")
    parser.add_argument("--output-field", default="output")
    parser.add_argument("--label-field", default="semantic_label")
    parser.add_argument("--classifier", choices=["keyword", "hf-zero-shot"], default="keyword")
    parser.add_argument("--labels", nargs="*", default=DEFAULT_LABELS)
    parser.add_argument("--zero-shot-model", default="facebook/bart-large-mnli")
    args = parser.parse_args()

    if args.num_clients <= 0:
        raise ValueError("--num-clients must be positive")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive")

    records = load_records(args.input)
    labeler = build_labeler(args)
    for record in records:
        ensure_sft_fields(record, args)
        text = sample_text(record, args)
        record[args.label_field] = labeler(text)

    shards = dirichlet_partition(records, args.label_field, args.num_clients, args.alpha, args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    manifest = {
        "source": args.input,
        "num_records": len(records),
        "num_clients": args.num_clients,
        "alpha": args.alpha,
        "seed": args.seed,
        "label_field": args.label_field,
        "classifier": args.classifier,
        "labels": args.labels,
        "clients": [],
    }
    for client_idx, shard in enumerate(shards):
        client_id = f"client_{client_idx:03d}"
        path = os.path.join(args.output_dir, f"{client_id}.jsonl")
        write_jsonl(path, shard)
        counts = Counter(str(row.get(args.label_field, "unlabeled")) for row in shard)
        manifest["clients"].append({
            "client_id": client_id,
            "path": path,
            "num_records": len(shard),
            "label_counts": dict(sorted(counts.items())),
        })

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(manifest_path)


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "train", "records"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported JSON structure in {path}")


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_labeler(args):
    if args.classifier == "keyword":
        return lambda text: keyword_label(text, args.labels)

    from transformers import pipeline

    classifier = pipeline("zero-shot-classification", model=args.zero_shot_model)

    def label(text: str) -> str:
        if not text.strip():
            return "general"
        result = classifier(text[:2048], candidate_labels=args.labels)
        return str(result["labels"][0])

    return label


def keyword_label(text: str, labels: list[str]) -> str:
    lowered = text.lower()
    scores = {label: 0 for label in labels}
    for label, words in KEYWORDS.items():
        if label not in scores:
            continue
        for word in words:
            scores[label] += len(re.findall(rf"\b{re.escape(word)}\b", lowered))
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    return best_label if best_score > 0 else ("general" if "general" in labels else labels[0])


def sample_text(record: dict, args) -> str:
    prompt = record.get(args.prompt_field)
    response = record.get(args.response_field)
    if prompt is None and args.instruction_field in record:
        instruction = str(record.get(args.instruction_field, "")).strip()
        extra_input = str(record.get(args.input_field, "")).strip()
        prompt = f"{instruction}\n{extra_input}".strip()
    if response is None and args.output_field in record:
        response = record.get(args.output_field)
    return f"{prompt or ''}\n{response or ''}".strip()


def ensure_sft_fields(record: dict, args) -> None:
    if not str(record.get(args.prompt_field, "")).strip():
        instruction = str(record.get(args.instruction_field, "")).strip()
        extra_input = str(record.get(args.input_field, "")).strip()
        prompt = f"{instruction}\n{extra_input}".strip()
        if prompt:
            record[args.prompt_field] = prompt
    if not str(record.get(args.response_field, "")).strip() and args.output_field in record:
        record[args.response_field] = record.get(args.output_field)


def dirichlet_partition(
    records: list[dict],
    label_field: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[dict]]:
    rnd = random.Random(seed)
    by_label = defaultdict(list)
    for record in records:
        by_label[str(record.get(label_field, "unlabeled"))].append(record)

    shards = [[] for _ in range(num_clients)]
    for label_records in by_label.values():
        rnd.shuffle(label_records)
        weights = [rnd.gammavariate(alpha, 1.0) for _ in range(num_clients)]
        total = sum(weights)
        if total == 0:
            weights = [1.0] * num_clients
            total = float(num_clients)
        counts = [int(len(label_records) * weight / total) for weight in weights]
        while sum(counts) < len(label_records):
            counts[min(range(num_clients), key=lambda idx: counts[idx])] += 1

        cursor = 0
        for client_idx, count in enumerate(counts):
            shards[client_idx].extend(label_records[cursor:cursor + count])
            cursor += count

    rebalance_empty_shards(shards, rnd)
    for shard in shards:
        rnd.shuffle(shard)
    return shards


def rebalance_empty_shards(shards: list[list[dict]], rnd: random.Random) -> None:
    for shard in shards:
        if shard:
            continue
        donor = max(shards, key=len)
        if len(donor) <= 1:
            break
        shard.append(donor.pop(rnd.randrange(len(donor))))


if __name__ == "__main__":
    main()
