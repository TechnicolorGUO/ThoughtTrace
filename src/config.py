"""Tiny config loader. PyYAML is imported lazily so that pure-compute phases
(which never touch the config) can run without it installed locally."""

from __future__ import annotations

from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    import yaml  # lazy: only needed when an LLM/embedding phase runs

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
