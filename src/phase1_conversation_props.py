"""Phase 1 — conversation properties.

1A (this file, implemented) — Demographics (Appendix D.1): aggregate the
per-conversation survey fields into the 6-panel bar chart (Fig 2). Pure gold
metadata, no model.

1B (this file, implemented) — Length distributions (Appendix D.2): turns and
tokens per conversation, plus per-role message lengths (Figs 3a, A1, A2).
Baseline (WildChat / LMSYS) overlays are intentionally skipped; only
ThoughtTrace's own stats are computed. Pure-compute (tiktoken gpt-4o).

1C/1D (topic labeling / multi-turn relationship) are added later; they need the LLM.

Data realities handled here (verified against the full release):
  * 75 conversations have no survey_answers, 152 have no frequency -> counted as
    missing, never silently bucketed.
  * `age` contains out-of-range garbage (values like 2 and 366) -> kept only if
    18-100, others reported as `dropped_age`.
  * `occupation` is free-text with mixed case ("Student"/"student") and blanks
    -> case-normalized, blanks dropped, top-8 + "Other".
  * `purposes` is a free-text comma list -> keyword-grouped, multi-label (one
    conversation can count toward several purpose groups).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import io_utils as io

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase1a"
OUT_DIR_1B = Path(__file__).resolve().parent.parent / "outputs" / "phase1b"

AGE_BRACKETS = [(18, 24), (25, 34), (35, 44), (45, 54), (55, 64), (65, 200)]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# purposes free-text -> 8 canonical groups, matched by substring (multi-label).
PURPOSE_KEYWORDS = {
    "Learning": ["learn", "study", "understand", "education", "knowledge"],
    "Working": ["work", "job", "professional", "business", "career"],
    "Brainstorming": ["brainstorm", "idea", "creativ"],
    "Research": ["research", "search", "find", "investigat", "analy"],
    "Coding": ["cod", "program", "software", "develop", "debug", "script"],
    "Planning": ["plan", "schedul", "organi", "manage"],
    "Writing": ["writ", "draft", "edit", "essay", "email", "summari", "translat"],
    "Translation": ["translat", "language"],
}


def _survey(conv: dict) -> dict | None:
    sa = conv.get("survey_answers") or []
    return sa[0] if sa else None


def _clean(v) -> str | None:
    if v is None:
        return None
    v = str(v).strip()
    return v or None


def _age_bracket(raw) -> str | None:
    v = _clean(raw)
    if v is None:
        return None
    try:
        age = int(float(v))
    except ValueError:
        return None
    if not (18 <= age <= 100):
        return None  # out-of-range garbage
    for (lo, hi), label in zip(AGE_BRACKETS, AGE_LABELS):
        if lo <= age <= hi:
            return label
    return None


def _purpose_groups(raw) -> set[str]:
    v = _clean(raw)
    if v is None:
        return set()
    text = v.lower()
    return {g for g, kws in PURPOSE_KEYWORDS.items() if any(k in text for k in kws)}


def compute_demographics(data: dict[str, dict], occupation_top: int = 8) -> dict:
    age = Counter()
    gender = Counter()
    education = Counter()
    frequency = Counter()
    purposes = Counter()
    occ_norm = Counter()
    occ_casings: dict[str, Counter] = defaultdict(Counter)  # lower -> casing votes

    n_total = len(data)
    n_survey = 0
    missing = defaultdict(int)
    dropped_age = 0

    for conv in data.values():
        s = _survey(conv)
        if s is None:
            missing["survey"] += 1
            continue
        n_survey += 1

        ab = _age_bracket(s.get("age"))
        if ab is not None:
            age[ab] += 1
        elif _clean(s.get("age")) is None:
            missing["age"] += 1
        else:
            dropped_age += 1  # present but out-of-range / unparseable

        g = _clean(s.get("gender"))
        if g:
            gender[g] += 1
        else:
            missing["gender"] += 1

        e = _clean(s.get("education"))
        if e:
            education[e] += 1
        else:
            missing["education"] += 1

        f = _clean(s.get("frequency"))
        if f in {"1", "2", "3", "4", "5"}:
            frequency[f] += 1
        else:
            missing["frequency"] += 1

        o = _clean(s.get("occupation"))
        if o:
            key = o.lower()
            occ_norm[key] += 1
            occ_casings[key][o] += 1
        else:
            missing["occupation"] += 1

        for grp in _purpose_groups(s.get("purposes")):
            purposes[grp] += 1

    # occupation: top-N by normalized count, with display casing, + Other bucket
    top = occ_norm.most_common(occupation_top)
    occupation = {occ_casings[k].most_common(1)[0][0]: c for k, c in top}
    other = sum(c for k, c in occ_norm.items() if k not in {k for k, _ in top})
    if other:
        occupation["Other"] = other

    def ordered(counter, keys=None):
        if keys:
            return {k: counter.get(k, 0) for k in keys}
        return dict(counter.most_common())

    return {
        "n_total": n_total,
        "n_with_survey": n_survey,
        "missing": dict(missing),
        "dropped_age_out_of_range": dropped_age,
        "panels": {
            "age": ordered(age, AGE_LABELS),
            "gender": ordered(gender),
            "education": ordered(education),
            "occupation": occupation,
            "frequency": ordered(frequency, ["1", "2", "3", "4", "5"]),
            "purposes": ordered(purposes, list(PURPOSE_KEYWORDS)),
        },
    }


def run_demographics(data: dict[str, dict], out_dir: Path = OUT_DIR) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    demo = compute_demographics(data)
    (out_dir / "demographics.json").write_text(
        json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figure = None
    try:
        figure = _render_6panel(demo["panels"], out_dir / "fig2_demographics.png")
    except ImportError:
        pass
    return {"demo": demo, "figure": figure, "out_dir": str(out_dir)}


def _render_6panel(panels: dict, out_path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["age", "gender", "education", "occupation", "frequency", "purposes"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, name in zip(axes.flat, order):
        d = panels[name]
        items = list(d.items())
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(labels, vals, color="#4C72B0")
        ax.invert_yaxis()
        ax.set_title(name.capitalize())
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v}", va="center", fontsize=8)
        ax.margins(x=0.18)
    fig.suptitle("Demographics (Fig 2)", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ============================================================================
# 1B — Length distributions (D.2)
# ============================================================================

# histogram buckets
TURN_BUCKETS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 10**9)]
TURN_BUCKET_LABELS = ["1", "2", "3-4", "5-8", "9-16", "17-32", "33+"]
TOKEN_BUCKETS = [
    (0, 100), (101, 300), (301, 1000), (1001, 3000),
    (3001, 10000), (10001, 30000), (30001, 10**12),
]
TOKEN_BUCKET_LABELS = ["0-100", "101-300", "301-1k", "1k-3k", "3k-10k", "10k-30k", "30k+"]


def _summary(values: list[int]) -> dict:
    import statistics as st

    if not values:
        return {"n": 0}
    s = sorted(values)
    return {
        "n": len(s),
        "min": s[0],
        "p25": st.median(s[: len(s) // 2]) if len(s) > 1 else s[0],
        "median": st.median(s),
        "mean": round(st.mean(s), 2),
        "p75": st.median(s[(len(s) + 1) // 2:]) if len(s) > 1 else s[0],
        "max": s[-1],
    }


def _bucketize(values: list[int], buckets, labels) -> dict:
    out = {lab: 0 for lab in labels}
    for v in values:
        for (lo, hi), lab in zip(buckets, labels):
            if lo <= v <= hi:
                out[lab] += 1
                break
    return out


def compute_length_stats(data: dict[str, dict], tokenizer_name: str = "gpt-4o") -> dict:
    from .tokenizer import TokenCounter

    tok = TokenCounter(tokenizer_name)

    turns_per_conv: list[int] = []
    tokens_per_conv: list[int] = []
    user_msg_tokens: list[int] = []
    asst_msg_tokens: list[int] = []

    for conv in data.values():
        turns_per_conv.append(io.n_turns(conv))
        conv_tokens = 0
        for msg in io.messages(conv):
            t = tok(msg.get("content", ""))
            conv_tokens += t
            if msg.get("type") == io.USER:
                user_msg_tokens.append(t)
            elif msg.get("type") == io.ASSISTANT:
                asst_msg_tokens.append(t)
        tokens_per_conv.append(conv_tokens)

    return {
        "tokenizer": tokenizer_name,
        "tokens_exact": tok.is_exact,  # False -> word-count fallback (no tiktoken)
        "turns_per_conv": {
            "summary": _summary(turns_per_conv),
            "hist": _bucketize(turns_per_conv, TURN_BUCKETS, TURN_BUCKET_LABELS),
        },
        "tokens_per_conv": {
            "summary": _summary(tokens_per_conv),
            "hist": _bucketize(tokens_per_conv, TOKEN_BUCKETS, TOKEN_BUCKET_LABELS),
        },
        "user_msg_tokens": {"summary": _summary(user_msg_tokens)},
        "assistant_msg_tokens": {"summary": _summary(asst_msg_tokens)},
    }


def run_length(data: dict[str, dict], out_dir: Path = OUT_DIR_1B,
               tokenizer_name: str = "gpt-4o") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_length_stats(data, tokenizer_name)
    (out_dir / "length_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    figs = []
    try:
        from .viz import barh_counts

        barh_counts(stats["turns_per_conv"]["hist"], "Turns per conversation (Fig 3a)",
                    out_dir / "fig3a_turns.png", color="#4C72B0")
        barh_counts(stats["tokens_per_conv"]["hist"], "Tokens per conversation (Fig A1)",
                    out_dir / "figA1_tokens.png", color="#55A868")
        figs = ["fig3a_turns.png", "figA1_tokens.png"]
    except ImportError:
        pass
    return {"stats": stats, "figures": figs, "out_dir": str(out_dir)}


# ============================================================================
# CLI
# ============================================================================

def _print_demographics(res: dict) -> None:
    demo = res["demo"]
    print(f"conversations: {demo['n_total']}  with survey: {demo['n_with_survey']}")
    print(f"missing: {demo['missing']}  dropped out-of-range ages: {demo['dropped_age_out_of_range']}")
    for name, d in demo["panels"].items():
        print(f"\n--- {name} ---")
        for k, v in d.items():
            print(f"  {k:>16}: {v}")
    print(f"\nWrote {res['out_dir']}/  figure: {res['figure'] or '(matplotlib missing)'}")


def _print_length(res: dict) -> None:
    s = res["stats"]
    if not s["tokens_exact"]:
        print("WARNING: tiktoken not installed -> token counts are an APPROXIMATE "
              "word-count fallback. Run on the server for exact gpt-4o tokens.\n")
    print(f"turns/conv     : {s['turns_per_conv']['summary']}")
    print(f"  hist: {s['turns_per_conv']['hist']}")
    print(f"tokens/conv    : {s['tokens_per_conv']['summary']}")
    print(f"  hist: {s['tokens_per_conv']['hist']}")
    print(f"user-msg tokens: {s['user_msg_tokens']['summary']}")
    print(f"asst-msg tokens: {s['assistant_msg_tokens']['summary']}")
    print(f"\nWrote {res['out_dir']}/  figures: {res['figures'] or '(matplotlib missing)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 conversation properties")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("demographics", "length"):
        sp = sub.add_parser(name)
        sp.add_argument("--quick", action="store_true")
    ap.set_defaults(cmd="demographics")
    args = ap.parse_args()

    data = io.load(quick=getattr(args, "quick", False))
    if args.cmd == "length":
        _print_length(run_length(data))
    else:
        _print_demographics(run_demographics(data))


if __name__ == "__main__":
    main()
