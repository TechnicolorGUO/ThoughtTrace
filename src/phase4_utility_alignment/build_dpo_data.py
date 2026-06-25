"""Phase 4 / 7A — build DPO preference data (Appendix D.9).

Three training arms, each producing {prompt, chosen, rejected} preference pairs
where `rejected` is the assistant's original (dissatisfying) response and `chosen`
is an LLM rewrite that addresses the dissatisfaction:

  * thought-guided (ThoughtTrace)  — dissatisfaction located via GOLD reaction
        labels {content_relevance, presentation_style, scope_fit}; rewrite uses
        the user's private reaction. Target 1,000.
  * message-guided (ThoughtTrace)  — same conversations, but dissatisfaction is
        CLASSIFIED from the user's follow-up message (LLM), and the rewrite uses
        that follow-up. Target 450 — intentionally smaller on the same convs, to
        show thoughts surface ~2.2x more dissatisfaction than messages.
  * message-guided (WildChat)      — same message-only pipeline over WildChat.
        Target 1,000. Needs the WildChat dataset; optional / off by default.

Design split (so the model-free logic is testable locally):
  * `collect_thought_guided` / `collect_message_candidates` are PURE — gold
    selection, filtering, context slicing. No model. Verifiable offline.
  * `generate_chosen_*` call the rewriter (any OpenAI-compatible endpoint).
  * `build_*` tie them together up to a target count.

Output: one JSONL per arm under outputs/phase4/, each line:
    {"prompt": [{role,content}...], "chosen": str, "rejected": str, "meta": {...}}
`prompt` is conversational (list of messages ending on a user turn); train_dpo.py
applies the tokenizer chat template.

Usage (rewrite step needs an endpoint):
    python -m src.phase4_utility_alignment.build_dpo_data --arm thought --n 1000
    python -m src.phase4_utility_alignment.build_dpo_data --arm message_tt --n 450
    python -m src.phase4_utility_alignment.build_dpo_data --arm message_wildchat --n 1000 --wildchat PATH
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .. import io_utils as io
from ..prompts import fill, load_prompt

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "phase4"
MIN_TURNS, MAX_TURNS = 2, 20
MIN_THOUGHT_WORDS = 6


# --- shared helpers ---------------------------------------------------------

def conversations_in_range(data: dict[str, dict]):
    for conv in data.values():
        if MIN_TURNS <= io.n_turns(conv) <= MAX_TURNS:
            yield conv


def thought_ok(text: str | None) -> bool:
    """Filter: non-empty, >= 6 words, has alphabetic chars."""
    if not text or not text.strip():
        return False
    if len(text.split()) < MIN_THOUGHT_WORDS:
        return False
    return any(ch.isalpha() for ch in text)


def _as_prompt(context_msgs: list[dict]) -> list[dict]:
    """Conversational prompt: role/content only, ending on a user turn."""
    return [{"role": m["type"], "content": m.get("content", "")} for m in context_msgs]


def _render(context_msgs: list[dict]) -> str:
    return "\n".join(
        f"{'User' if m['type'] == io.USER else 'Assistant'}: {m.get('content', '').strip()}"
        for m in context_msgs
    )


# --- thought-guided arm (gold dissatisfaction) ------------------------------

def collect_thought_guided(data: dict[str, dict]) -> list[dict]:
    """PURE: every gold dissatisfaction reaction (passing the filter) with a
    prior context. Returns candidates ready for rewriting."""
    out = []
    for conv in conversations_in_range(data):
        msgs = io.messages(conv)
        for i, m in enumerate(msgs):
            if m.get("type") != io.ASSISTANT or i == 0:
                continue  # need context before the dissatisfying turn
            for r in io.get_reactions(m):
                if r.get("label") in io.DISSATISFACTION_LABELS and thought_ok(r.get("content")):
                    out.append({
                        "conv_id": conv["id"], "index": i,
                        "context_msgs": msgs[:i],          # up to, NOT incl. the bad turn
                        "original": m.get("content", ""),  # rejected
                        "signal": r.get("content", ""),    # the reaction
                        "label": r.get("label"),
                        "arm": "thought_guided",
                    })
    return out


def generate_chosen_thought(cand: dict, client) -> str:
    prompt = fill(
        load_prompt("rewrite_thought_guided.txt"),
        context=_render(cand["context_msgs"]),
        original=cand["original"],
        thought=cand["signal"],
    )
    return client.complete(prompt).strip()


# --- message-guided arm (classified dissatisfaction) ------------------------

def collect_message_candidates(data: dict[str, dict]) -> list[dict]:
    """PURE: every (assistant -> next-user) pair with prior context. The
    dissatisfied/satisfied decision is made later by the LLM classifier."""
    out = []
    for conv in conversations_in_range(data):
        msgs = io.messages(conv)
        for i, m in enumerate(msgs):
            if m.get("type") != io.ASSISTANT or i == 0:
                continue
            if i + 1 < len(msgs) and msgs[i + 1].get("type") == io.USER:
                out.append({
                    "conv_id": conv["id"], "index": i,
                    "context_msgs": msgs[:i],
                    "original": m.get("content", ""),
                    "signal": msgs[i + 1].get("content", ""),  # the follow-up message
                    "arm": "message_guided",
                })
    return out


def classify_dissatisfied(cand: dict, judge) -> bool:
    prompt = fill(
        load_prompt("message_guided_classify.txt"),
        original=cand["original"], followup=cand["signal"],
    )
    out = judge.complete_json(prompt)
    return bool((out or {}).get("dissatisfied", False)) if isinstance(out, dict) else False


def generate_chosen_message(cand: dict, client) -> str:
    prompt = fill(
        load_prompt("rewrite_message_guided.txt"),
        context=_render(cand["context_msgs"]),
        original=cand["original"], followup=cand["signal"],
    )
    return client.complete(prompt).strip()


# --- builders (collect -> [classify] -> rewrite -> pair) --------------------

def _pair(cand: dict, chosen: str) -> dict:
    return {
        "prompt": _as_prompt(cand["context_msgs"]),
        "chosen": chosen,
        "rejected": cand["original"],
        "meta": {k: cand.get(k) for k in ("conv_id", "index", "arm", "label", "signal")},
    }


def build_thought_guided(data, client, n=1000, seed=0) -> list[dict]:
    cands = collect_thought_guided(data)
    random.Random(seed).shuffle(cands)
    pairs = []
    for c in cands:
        if len(pairs) >= n:
            break
        chosen = generate_chosen_thought(c, client)
        if chosen:
            pairs.append(_pair(c, chosen))
    return pairs


def build_message_guided(data, client, judge=None, n=450, seed=0,
                         candidates=None) -> list[dict]:
    judge = judge or client
    cands = candidates if candidates is not None else collect_message_candidates(data)
    random.Random(seed).shuffle(cands)
    pairs = []
    for c in cands:
        if len(pairs) >= n:
            break
        if not classify_dissatisfied(c, judge):
            continue
        chosen = generate_chosen_message(c, client)
        if chosen:
            pairs.append(_pair(c, chosen))
    return pairs


def write_pairs(pairs: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs),
                    encoding="utf-8")


# --- CLI --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4/7A — build DPO data")
    ap.add_argument("--arm", choices=["thought", "message_tt", "message_wildchat"],
                    required=True)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=None,
                    help="config yaml for the rewriter/judge endpoint (default: configs/default.yaml)")
    ap.add_argument("--wildchat", default=None, help="path to WildChat jsonl (message_wildchat)")
    args = ap.parse_args()

    from ..config import load_config
    from ..llm_client import LLMClient

    cfg = load_config(args.config) if args.config else load_config()
    client = LLMClient.from_config(cfg["llm"])
    judge = LLMClient.from_config(cfg["judge"]) if cfg.get("judge", {}).get("model") else client
    data = io.load(quick=args.quick)

    if args.arm == "thought":
        pairs = build_thought_guided(data, client, n=args.n or 1000, seed=args.seed)
        out = OUT_DIR / "dpo_thought_guided.jsonl"
    elif args.arm == "message_tt":
        pairs = build_message_guided(data, client, judge=judge, n=args.n or 450, seed=args.seed)
        out = OUT_DIR / "dpo_message_guided_tt.jsonl"
    else:
        raise SystemExit("message_wildchat: load WildChat into the message pipeline "
                         "via collect_message_candidates on the WildChat conversations "
                         "(pass --wildchat PATH); not wired by default.")

    write_pairs(pairs, out)
    print(f"arm={args.arm}  built {len(pairs)} DPO pairs -> {out}")


if __name__ == "__main__":
    main()
