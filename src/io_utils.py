"""I/O and schema helpers for the ThoughtTrace dataset.

Mirrors the upstream loader (`{id: conversation}` dict) and adds the turn /
thought accessors the reproduction phases rely on. Loading is the single source
of truth for schema constants (label sets, role names), so downstream phases
import them from here instead of hard-coding strings.

Conversation schema (one JSON object per line in data/ThoughtTrace.jsonl):

    conversation
    ├── id, model_name, model_provider, created_at, updated_at
    ├── task_summary, task_expectation
    ├── survey_answers[]            # demographics (age, gender, education, ...)
    └── messages[]
        ├── id, timestamp, type ("user"|"assistant"), content
        ├── reasons[]              # only on user messages  -> {content, timestamp, label}
        └── reactions[]            # only on assistant messages -> {content, timestamp, label}

At load time every message is annotated with `_turn`, its 1-indexed position in
the conversation, so `turn_index(msg)` is O(1) and matches the upstream
"turn" definition (each message is one turn; median is 8).
"""

from __future__ import annotations

import json
from pathlib import Path

# --- schema constants (verified against the full release) -------------------

USER = "user"
ASSISTANT = "assistant"

# 7 reason types (attached to user messages)
REASON_LABELS = (
    "task_motivation",
    "task_continuation",
    "context_grounding_and_constraints",
    "content_expectation",
    "social_and_others",
    "style_expectation",
    "task_reorientation",
)

# 5 reaction types (attached to assistant messages)
REACTION_LABELS = (
    "explicit_affirmation",
    "content_relevance",
    "presentation_style",
    "scope_fit",
    "partial_satisfaction",
)

# The three dissatisfaction reaction labels used to seed the alignment
# experiment (Phase 4) and the dissatisfaction analyses.
DISSATISFACTION_LABELS = frozenset(
    {"content_relevance", "presentation_style", "scope_fit"}
)

# Expected global totals for the full release (Phase 0 acceptance check).
EXPECTED_TOTALS = {
    "users": 1058,
    "conversations": 2155,
    "messages": 17058,
    "thoughts": 10174,
    "models": 20,
}

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "ThoughtTrace.jsonl"
EXAMPLES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "ThoughtTrace_examples.jsonl"
)


# --- loading ----------------------------------------------------------------

def load(path: str | Path = DEFAULT_PATH, quick: bool = False) -> dict[str, dict]:
    """Load the JSONL dataset into a ``{id: conversation}`` dict.

    Args:
        path: JSONL file to read. Ignored when ``quick`` is set.
        quick: load the small ``ThoughtTrace_examples.jsonl`` sample instead,
            for fast smoke tests (the spec's ``--quick`` mode).
    """
    path = EXAMPLES_PATH if quick else Path(path)
    data: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            _annotate_turns(conv)
            data[conv["id"]] = conv
    return data


def _annotate_turns(conv: dict) -> None:
    """Annotate each message with its 1-indexed turn position (in place)."""
    for i, msg in enumerate(conv.get("messages", []), start=1):
        msg["_turn"] = i


# --- message / turn accessors -----------------------------------------------

def messages(conv: dict) -> list[dict]:
    return conv.get("messages", [])


def turn_index(msg: dict) -> int:
    """1-indexed position of a message within its conversation."""
    return msg["_turn"]


def n_turns(conv: dict) -> int:
    """Number of turns (= messages) in the conversation."""
    return len(messages(conv))


def user_id(conv: dict) -> str:
    """Participant id, e.g. ``user804_task2_conversation1`` -> ``user804``."""
    return conv["id"].split("_", 1)[0]


def iter_user_turns(conv: dict):
    """Yield user messages in order."""
    for msg in messages(conv):
        if msg.get("type") == USER:
            yield msg


def iter_assistant_turns(conv: dict):
    """Yield assistant messages in order."""
    for msg in messages(conv):
        if msg.get("type") == ASSISTANT:
            yield msg


# --- thought accessors ------------------------------------------------------

def get_reasons(msg: dict) -> list[dict]:
    """Reasons attached to a user message (possibly empty)."""
    return msg.get("reasons") or []


def get_reactions(msg: dict) -> list[dict]:
    """Reactions attached to an assistant message (possibly empty)."""
    return msg.get("reactions") or []


def get_reason(msg: dict) -> dict | None:
    """First reason on a message, or ``None``. Convenience for the common
    single-thought case; use :func:`get_reasons` when a message may carry
    several."""
    rs = get_reasons(msg)
    return rs[0] if rs else None


def get_reaction(msg: dict) -> dict | None:
    """First reaction on a message, or ``None``."""
    rs = get_reactions(msg)
    return rs[0] if rs else None


def iter_thoughts(conv: dict):
    """Yield ``(msg, thought, kind)`` for every thought in the conversation,
    where ``kind`` is ``"reason"`` or ``"reaction"``."""
    for msg in messages(conv):
        for r in get_reasons(msg):
            yield msg, r, "reason"
        for r in get_reactions(msg):
            yield msg, r, "reaction"


def dissatisfaction_reactions(conv: dict) -> list[dict]:
    """All reaction thoughts whose gold label marks dissatisfaction
    ({content_relevance, presentation_style, scope_fit})."""
    out = []
    for msg in iter_assistant_turns(conv):
        for r in get_reactions(msg):
            if r.get("label") in DISSATISFACTION_LABELS:
                out.append(r)
    return out
