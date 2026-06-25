"""Shared prompt loading/filling.

Template files under ``prompts/`` start with a ``#`` developer-note block
(provenance, "replace with the exact Appendix text", placeholder docs) that must
NOT be sent to the model. ``load_prompt`` strips that leading block; ``fill``
substitutes ``{name}`` placeholders without str.format (templates and content
contain literal JSON braces that would break str.format).
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def load_prompt(name: str) -> str:
    text = (PROMPTS_DIR / name).read_text(encoding="utf-8")
    lines = text.splitlines()
    i = 0
    # drop the contiguous top-of-file comment/blank block; a '#' inside the
    # actual prompt body (after real content begins) is preserved.
    while i < len(lines) and (lines[i].lstrip().startswith("#") or not lines[i].strip()):
        i += 1
    return "\n".join(lines[i:]).strip()


def fill(template: str, **kwargs: str) -> str:
    out = template
    for k, v in kwargs.items():
        out = out.replace("{" + k + "}", v if v is not None else "")
    return out
