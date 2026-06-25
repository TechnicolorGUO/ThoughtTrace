"""Phase 2C — thought-type distributions + (optional) labeler validation.

Two independent jobs:

1. ``distributions`` — compute the 7 reason-type and 5 reaction-type
   distributions DIRECTLY from the gold ``label`` fields (Figs 6-7). Pure
   stdlib; no model required. Dumps counts to JSON and renders bar charts if
   matplotlib is available.

2. ``validate`` — the optional labeler-validation pass. Re-label a deterministic
   sample of thoughts with Qwen3-8B and report agreement (accuracy / macro-F1 /
   confusion) against the gold labels. This needs the vLLM endpoint, so it runs
   on the server. It is the single best signal of how much to trust the
   LLM-labeled figures in Phase 1.

CLI:
    python -m src.phase2_thought_props distributions            # local, no model
    python -m src.phase2_thought_props validate --kind reason --n 200   # server
    python -m src.phase2_thought_props validate --kind reaction --n 200
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from . import io_utils as io

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase2c"


# --- 1. gold-label distributions (no model) ---------------------------------

def compute_gold_distributions(data: dict[str, dict]) -> dict:
    reasons: Counter[str] = Counter()
    reactions: Counter[str] = Counter()
    for conv in data.values():
        for _msg, thought, kind in io.iter_thoughts(conv):
            label = thought.get("label", "<none>")
            (reasons if kind == "reason" else reactions)[label] += 1

    # keep the canonical label order from io_utils, descending by count
    reasons_out = {k: reasons.get(k, 0) for k in io.REASON_LABELS}
    reactions_out = {k: reactions.get(k, 0) for k in io.REACTION_LABELS}
    reasons_out = dict(sorted(reasons_out.items(), key=lambda kv: kv[1], reverse=True))
    reactions_out = dict(sorted(reactions_out.items(), key=lambda kv: kv[1], reverse=True))

    return {
        "reasons": reasons_out,
        "reactions": reactions_out,
        "n_reasons": sum(reasons.values()),
        "n_reactions": sum(reactions.values()),
    }


def run_distributions(data: dict[str, dict], out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    dist = compute_gold_distributions(data)
    (out_dir / "gold_distributions.json").write_text(
        json.dumps(dist, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # figures are best-effort (skip cleanly if matplotlib is missing)
    try:
        from .viz import barh_counts

        barh_counts(dist["reasons"], "Reason types (gold) — Fig 6",
                    out_dir / "fig6_reason_types.png", color="#4C72B0")
        barh_counts(dist["reactions"], "Reaction types (gold) — Fig 7",
                    out_dir / "fig7_reaction_types.png", color="#C44E52")
        figs = ["fig6_reason_types.png", "fig7_reaction_types.png"]
    except ImportError:
        figs = []

    return {"dist": dist, "figures": figs, "out_dir": str(out_dir)}


# --- 2. optional labeler validation (needs the model) -----------------------

def _load_prompt(name: str) -> str:
    from .prompts import load_prompt
    return load_prompt(name)


def _fill(template: str, message: str, thought: str) -> str:
    from .prompts import fill
    return fill(template, message=message, thought=thought)


def _sample_thoughts(data: dict[str, dict], kind: str, n: int, seed: int):
    """Deterministically sample up to ``n`` (message, thought) pairs of one kind."""
    pool = []
    for conv in data.values():
        for msg, thought, k in io.iter_thoughts(conv):
            if k == kind:
                pool.append((msg.get("content", ""), thought))
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def _macro_f1(gold: list[str], pred: list[str], labels) -> dict:
    """Per-class precision/recall/F1 + macro average, over a closed label set."""
    tp = Counter(); fp = Counter(); fn = Counter()
    for g, p in zip(gold, pred):
        if p == g:
            tp[g] += 1
        else:
            fp[p] += 1
            fn[g] += 1
    per_class = {}
    f1s = []
    for lab in labels:
        support = tp[lab] + fn[lab]
        prec = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        rec = tp[lab] / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
        # macro average only over labels present in the gold (sklearn-style):
        # zero-support classes would otherwise drag the mean down on small samples.
        if support > 0:
            f1s.append(f1)
    accuracy = sum(1 for g, p in zip(gold, pred) if g == p) / len(gold) if gold else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "per_class": per_class,
    }


def validate_labeler(
    data: dict[str, dict],
    client,
    kind: str = "reason",
    n: int = 200,
    seed: int = 0,
    out_dir: Path = OUT_DIR,
) -> dict:
    """Re-label a sample with the LLM and score agreement vs gold labels.

    ``client`` is an :class:`src.llm_client.LLMClient`. ``kind`` is
    ``"reason"`` or ``"reaction"``.
    """
    assert kind in ("reason", "reaction")
    labels = io.REASON_LABELS if kind == "reason" else io.REACTION_LABELS
    template = _load_prompt(f"{kind}_classify.txt")
    sample = _sample_thoughts(data, kind, n, seed)

    gold, pred, rows = [], [], []
    confusion: dict[str, Counter] = defaultdict(Counter)
    for message, thought in sample:
        g = thought.get("label")
        out = client.complete_json(_fill(template, message, thought.get("content", "")))
        p = (out or {}).get("label") if isinstance(out, dict) else None
        if p not in labels:
            p = "<invalid>"   # hallucinated / unparseable -> counts as a miss
        gold.append(g)
        pred.append(p)
        confusion[g][p] += 1
        rows.append({"gold": g, "pred": p, "message": message[:200],
                     "thought": thought.get("content", "")[:200]})

    metrics = _macro_f1(gold, pred, labels)
    result = {
        "kind": kind,
        "n": len(sample),
        "seed": seed,
        **metrics,
        "confusion": {g: dict(c) for g, c in confusion.items()},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"labeler_validation_{kind}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / f"labeler_validation_{kind}_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return result


# --- 3. Phase 2D — stage dynamics (Prop 4 / D.7) ----------------------------
# Pure gold-label. Reasons sit on user turns, reactions on assistant turns.
# Three views are computed locally; the topic / multi-turn-relationship
# cross-tabs (Figs A8-A10) additionally need Phase 1C/1D LLM labels and are
# wired separately once those exist.

STAGE_LABELS = ["Early", "Mid-Early", "Mid-Late", "Late"]


def stage_of(turn: int, n_turns: int) -> int:
    """Bin a 1-indexed turn into one of 4 normalized stages (0..3)."""
    if n_turns <= 1:
        return 0
    pos = (turn - 1) / (n_turns - 1)   # 0..1
    return min(int(pos * 4), 3)


def compute_stage_dynamics(data: dict[str, dict]) -> dict:
    """For each kind, a stage x thought-type count matrix (all thoughts)."""
    reason = {st: Counter() for st in STAGE_LABELS}
    reaction = {st: Counter() for st in STAGE_LABELS}
    for conv in data.values():
        T = io.n_turns(conv)
        for msg in io.messages(conv):
            st = STAGE_LABELS[stage_of(io.turn_index(msg), T)]
            for r in io.get_reasons(msg):
                reason[st][r.get("label", "<none>")] += 1
            for r in io.get_reactions(msg):
                reaction[st][r.get("label", "<none>")] += 1
    return {
        "reason": {st: {k: reason[st].get(k, 0) for k in io.REASON_LABELS}
                   for st in STAGE_LABELS},
        "reaction": {st: {k: reaction[st].get(k, 0) for k in io.REACTION_LABELS}
                     for st in STAGE_LABELS},
    }


def _label_sequence(conv: dict, kind: str) -> list[str]:
    """Representative thought label per turn, in turn order (first thought if a
    message carries several)."""
    seq = []
    for msg in io.messages(conv):
        thoughts = io.get_reasons(msg) if kind == "reason" else io.get_reactions(msg)
        if thoughts:
            seq.append(thoughts[0].get("label", "<none>"))
    return seq


def compute_transitions(data: dict[str, dict]) -> dict:
    """Within-conversation consecutive thought-type transitions (Sankey flows)."""
    out = {}
    for kind in ("reason", "reaction"):
        trans: Counter = Counter()
        for conv in data.values():
            seq = _label_sequence(conv, kind)
            for a, b in zip(seq, seq[1:]):
                trans[(a, b)] += 1
        out[kind] = {f"{a} -> {b}": c for (a, b), c in trans.most_common()}
    return out


def compute_length_crosstab(data: dict[str, dict]) -> dict:
    """Thought-type distribution bucketed by conversation length (Figs A11-A12)."""
    from .phase1_conversation_props import TURN_BUCKETS, TURN_BUCKET_LABELS

    def bucket(T: int) -> str:
        for (lo, hi), lab in zip(TURN_BUCKETS, TURN_BUCKET_LABELS):
            if lo <= T <= hi:
                return lab
        return TURN_BUCKET_LABELS[-1]

    reason = {lab: Counter() for lab in TURN_BUCKET_LABELS}
    reaction = {lab: Counter() for lab in TURN_BUCKET_LABELS}
    for conv in data.values():
        b = bucket(io.n_turns(conv))
        for _msg, thought, kind in io.iter_thoughts(conv):
            (reason if kind == "reason" else reaction)[b][thought.get("label")] += 1
    return {
        "reason": {b: {k: reason[b].get(k, 0) for k in io.REASON_LABELS}
                   for b in TURN_BUCKET_LABELS},
        "reaction": {b: {k: reaction[b].get(k, 0) for k in io.REACTION_LABELS}
                     for b in TURN_BUCKET_LABELS},
    }


def run_stage_dynamics(data: dict[str, dict], out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (OUT_DIR.parent / "phase2d")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "stage_dynamics": compute_stage_dynamics(data),
        "transitions": compute_transitions(data),
        "length_crosstab": compute_length_crosstab(data),
    }
    (out_dir / "stage_dynamics.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"result": result, "out_dir": str(out_dir)}


def _stage_share(stage_dyn: dict, kind: str, label: str) -> list[float]:
    """% share of one thought-type across the 4 stages (for quick inspection)."""
    out = []
    for st in STAGE_LABELS:
        d = stage_dyn[kind][st]
        tot = sum(d.values()) or 1
        out.append(round(100 * d.get(label, 0) / tot, 1))
    return out


# --- CLI --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2C")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("distributions", help="gold-label distributions (no model)")
    d.add_argument("--quick", action="store_true")

    v = sub.add_parser("validate", help="LLM labeler agreement vs gold (needs server)")
    v.add_argument("--kind", choices=["reason", "reaction"], required=True)
    v.add_argument("--n", type=int, default=200)
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--quick", action="store_true")

    s = sub.add_parser("stages", help="Phase 2D stage dynamics (no model)")
    s.add_argument("--quick", action="store_true")

    args = ap.parse_args()
    data = io.load(quick=getattr(args, "quick", False))

    if args.cmd == "distributions":
        res = run_distributions(data)
        dist = res["dist"]
        print("=== Reason types (gold) ===")
        for k, c in dist["reasons"].items():
            print(f"  {k:>34}: {c:>6} ({100*c/dist['n_reasons']:.1f}%)")
        print("=== Reaction types (gold) ===")
        for k, c in dist["reactions"].items():
            print(f"  {k:>34}: {c:>6} ({100*c/dist['n_reactions']:.1f}%)")
        print(f"\nWrote {res['out_dir']}/  figures: {res['figures'] or '(matplotlib missing)'}")

    elif args.cmd == "validate":
        from .config import load_config
        from .llm_client import LLMClient

        client = LLMClient.from_config(load_config()["llm"])
        res = validate_labeler(data, client, kind=args.kind, n=args.n, seed=args.seed)
        print(f"=== Labeler validation [{res['kind']}] n={res['n']} ===")
        print(f"  accuracy : {res['accuracy']:.3f}")
        print(f"  macro_f1 : {res['macro_f1']:.3f}")
        for lab, m in res["per_class"].items():
            print(f"  {lab:>34}: F1={m['f1']:.3f}  support={m['support']}")

    elif args.cmd == "stages":
        res = run_stage_dynamics(data)
        sd = res["result"]["stage_dynamics"]
        print("=== Stage dynamics: % share across [Early, Mid-Early, Mid-Late, Late] ===")
        print("reasons:")
        for lab in io.REASON_LABELS:
            print(f"  {lab:>34}: {_stage_share(sd, 'reason', lab)}")
        print("reactions:")
        for lab in io.REACTION_LABELS:
            print(f"  {lab:>34}: {_stage_share(sd, 'reaction', lab)}")
        print("\nTop reason transitions:")
        for k, c in list(res["result"]["transitions"]["reason"].items())[:5]:
            print(f"  {k}: {c}")
        print(f"\nWrote {res['out_dir']}/")


if __name__ == "__main__":
    main()
