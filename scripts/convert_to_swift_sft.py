#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM_PROMPT = (
    "You are simulating a real user in a human-AI conversation. Given the conversation history and the "
    "assistant's latest reply, first write the user's private thoughts, then write the user's next message. "
    "Follow the exact output format."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def thought_text(items: list[dict[str, Any]]) -> str:
    return "；".join(str(item.get("content", "")).strip() for item in items if str(item.get("content", "")).strip())


def format_history(messages: list[dict[str, Any]], *, max_history_turns: int | None = None) -> str:
    if max_history_turns is not None and max_history_turns > 0:
        messages = messages[-max_history_turns * 2 :]

    lines: list[str] = []
    for message in messages:
        role = "User" if message.get("type") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_input(history: list[dict[str, Any]], assistant_message: dict[str, Any], *, max_history_turns: int | None) -> str:
    history_text = format_history(history, max_history_turns=max_history_turns)
    latest_reply = str(assistant_message.get("content", "")).strip()
    if history_text:
        return f"[Conversation History]\n{history_text}\n\n[Assistant Latest Reply]\nAssistant: {latest_reply}"
    return f"[Conversation History]\n\n[Assistant Latest Reply]\nAssistant: {latest_reply}"


def build_output(assistant_message: dict[str, Any], next_user_message: dict[str, Any]) -> str:
    reaction = thought_text(assistant_message.get("reactions") or [])
    motivation = thought_text(next_user_message.get("reasons") or [])
    reply = str(next_user_message.get("content", "")).strip()
    return (
        "<thought>\n"
        f"[Reaction]: {reaction}\n"
        f"[Motivation]: {motivation}\n"
        "</thought>\n"
        "<reply>\n"
        f"{reply}\n"
        "</reply>"
    )


def build_samples(
    conversations: list[dict[str, Any]],
    *,
    max_history_turns: int | None,
    require_reaction: bool,
    require_reason: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    stats = {
        "conversations": len(conversations),
        "assistant_to_user_pairs": 0,
        "skipped_missing_reaction": 0,
        "skipped_missing_reason": 0,
        "samples": 0,
    }

    for conversation in conversations:
        messages = conversation.get("messages") or []
        for index, assistant_message in enumerate(messages[:-1]):
            next_user_message = messages[index + 1]
            if assistant_message.get("type") != "assistant" or next_user_message.get("type") != "user":
                continue

            stats["assistant_to_user_pairs"] += 1
            has_reaction = bool(thought_text(assistant_message.get("reactions") or []))
            has_reason = bool(thought_text(next_user_message.get("reasons") or []))
            if require_reaction and not has_reaction:
                stats["skipped_missing_reaction"] += 1
                continue
            if require_reason and not has_reason:
                stats["skipped_missing_reason"] += 1
                continue

            sample = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_input(
                            messages[:index],
                            assistant_message,
                            max_history_turns=max_history_turns,
                        ),
                    },
                    {"role": "assistant", "content": build_output(assistant_message, next_user_message)},
                ],
                "metadata": {
                    "conversation_id": conversation.get("id"),
                    "assistant_message_id": assistant_message.get("id"),
                    "next_user_message_id": next_user_message.get("id"),
                    "model_name": conversation.get("model_name"),
                    "model_provider": conversation.get("model_provider"),
                    "reaction_labels": [item.get("label") for item in assistant_message.get("reactions") or []],
                    "reason_labels": [item.get("label") for item in next_user_message.get("reasons") or []],
                },
            }
            samples.append(sample)

    stats["samples"] = len(samples)
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ThoughtTrace JSONL to ms-swift chat SFT JSONL.")
    parser.add_argument("--input", default="ThoughtTrace/data/ThoughtTrace.jsonl")
    parser.add_argument("--output-dir", default="ThoughtTrace/data/processed")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-turns", type=int, default=6)
    parser.add_argument("--allow-missing-reaction", action="store_true")
    parser.add_argument("--allow-missing-reason", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    conversations = load_jsonl(input_path)
    samples, stats = build_samples(
        conversations,
        max_history_turns=args.max_history_turns,
        require_reaction=not args.allow_missing_reaction,
        require_reason=not args.allow_missing_reason,
    )

    rng = random.Random(args.seed)
    rng.shuffle(samples)
    val_size = max(1, round(len(samples) * args.val_ratio)) if samples else 0
    val_rows = samples[:val_size]
    train_rows = samples[val_size:]

    write_jsonl(train_rows, output_dir / "user_sim_train.jsonl")
    write_jsonl(val_rows, output_dir / "user_sim_val.jsonl")
    write_jsonl(samples[:20], output_dir / "user_sim_preview.jsonl")

    stats.update(
        {
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "val_ratio": args.val_ratio,
            "max_history_turns": args.max_history_turns,
        }
    )
    (output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
