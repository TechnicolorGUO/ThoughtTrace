#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def convert_row(row: dict[str, Any], index: int, data_source: str, ability: str) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f"Row {index} must contain at least 2 chat messages")

    target_message = messages[-1]
    if target_message.get("role") != "assistant":
        raise ValueError(f"Row {index} last message must be an assistant message")

    prompt = messages[:-1]
    target = str(target_message.get("content", "")).strip()
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"metadata": metadata}

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": {
            "style": "none",
            "ground_truth": target,
        },
        "extra_info": {
            **metadata,
            "index": index,
            "target": target,
        },
    }


def convert_file(input_path: Path, output_path: Path, data_source: str, ability: str) -> None:
    rows = load_jsonl(input_path)
    converted = [convert_row(row, idx, data_source, ability) for idx, row in enumerate(rows)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(converted).to_parquet(str(output_path))
    print(f"Wrote {len(converted)} rows to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ms-swift chat SFT JSONL into verl RL/OPD parquet prompt data."
    )
    parser.add_argument("--train-input", default="data/processed_en/user_sim_train.jsonl")
    parser.add_argument("--val-input", default="data/processed_en/user_sim_val.jsonl")
    parser.add_argument("--train-output", default="data/processed_en/user_sim_train.parquet")
    parser.add_argument("--val-output", default="data/processed_en/user_sim_val.parquet")
    parser.add_argument("--data-source", default="thoughttrace")
    parser.add_argument("--ability", default="user_simulation")
    args = parser.parse_args()

    convert_file(Path(args.train_input), Path(args.train_output), args.data_source, args.ability)
    convert_file(Path(args.val_input), Path(args.val_output), args.data_source, args.ability)


if __name__ == "__main__":
    main()
