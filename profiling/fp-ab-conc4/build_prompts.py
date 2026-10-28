#!/usr/bin/env python3
"""Build eight distinct chat prompts padded to a target token count."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


SEEDS = [
    ("coastal-weather", "A coastal town is preparing for a rapid change in weather."),
    ("library-logistics", "A small library is redesigning how it moves returned books."),
    ("greenhouse-sensors", "A greenhouse operator is calibrating temperature and humidity sensors."),
    ("night-freight", "A freight depot is scheduling its overnight loading crews."),
    ("river-ferry", "A river ferry is adjusting its timetable for low water levels."),
    ("bakery-workflow", "A neighborhood bakery is reorganizing its morning preparation line."),
    ("trail-maintenance", "A park crew is prioritizing repairs after a storm damaged trails."),
    ("radio-workshop", "A technical workshop is restoring a vintage radio for a public exhibit."),
]

FILLER = (
    " The plan must keep ordinary work moving while equipment is checked. "
    "Supervisors record each decision, compare it with the usual procedure, "
    "and explain any tradeoff in plain terms."
)


def prompt_tokens(tokenizer, text: str) -> int:
    messages = [{"role": "user", "content": text}]
    encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
    return len(encoded)


def padded_text(tokenizer, topic: str, seed: str, target: int) -> str:
    base = f"{seed} Summarize the operational tradeoff in two sentences."
    if prompt_tokens(tokenizer, base) > target:
        raise ValueError(f"base prompt already exceeds target for {topic}")
    low, high = 0, 1
    while prompt_tokens(tokenizer, base + FILLER * high) <= target:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if prompt_tokens(tokenizer, base + FILLER * middle) <= target:
            low = middle
        else:
            high = middle
    text = base + FILLER * low
    actual = prompt_tokens(tokenizer, text)
    if actual == target:
        return text

    for token_id in range(256, tokenizer.vocab_size):
        piece = tokenizer.decode([token_id], skip_special_tokens=True)
        if not piece or not piece.startswith(" ") or not piece.strip().isascii() or not piece.strip().isalpha():
            continue
        candidate = text + piece
        candidate_tokens = prompt_tokens(tokenizer, candidate)
        if candidate_tokens == actual + 1:
            text = candidate
            actual = candidate_tokens
            if actual == target:
                return text
        elif candidate_tokens > target:
            continue
    raise ValueError(f"could not hit exact token target for {topic}: {actual}/{target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-prompt-tokens", type=int, default=512)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompts = []
    for index, (topic, seed) in enumerate(SEEDS, 1):
        prompts.append({
            "id": f"fp-{index:02d}-{topic}",
            "text": padded_text(tokenizer, topic, seed, args.target_prompt_tokens),
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "target_prompt_tokens": args.target_prompt_tokens,
        "tokenizer": args.model,
        "prompts": prompts,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(prompts)} prompts to {output}")


if __name__ == "__main__":
    main()
