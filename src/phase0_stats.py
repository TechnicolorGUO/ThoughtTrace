"""Phase 0 — data load & sanity stats.

Reproduces ``check_dataset_stats.ipynb`` programmatically and asserts the global
totals from the reproduction spec (1,058 users / 2,155 conversations /
17,058 messages / 10,174 thoughts / 20 models). Also reports per-model counts
(Appendix Table A1) and the gold reason/reaction label distributions.

Usage:
    python -m src.phase0_stats              # full dataset, assert totals
    python -m src.phase0_stats --quick      # run on the 20-conversation sample
    python -m src.phase0_stats --json out.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from . import io_utils as io


def compute_stats(data: dict[str, dict]) -> dict:
    users: set[str] = set()
    models: set[str] = set()
    n_messages = 0
    role_counts: Counter[str] = Counter()
    reason_labels: Counter[str] = Counter()
    reaction_labels: Counter[str] = Counter()

    # per-model: users, conversations, messages, thoughts
    per_model_users: dict[str, set[str]] = defaultdict(set)
    per_model = defaultdict(lambda: {"conversations": 0, "messages": 0, "thoughts": 0})

    for conv in data.values():
        uid = io.user_id(conv)
        model = conv.get("model_name", "<unknown>")
        users.add(uid)
        models.add(model)
        per_model_users[model].add(uid)
        per_model[model]["conversations"] += 1

        for msg in io.messages(conv):
            n_messages += 1
            role_counts[msg.get("type", "<unknown>")] += 1
            per_model[model]["messages"] += 1
            for r in io.get_reasons(msg):
                reason_labels[r.get("label", "<none>")] += 1
                per_model[model]["thoughts"] += 1
            for r in io.get_reactions(msg):
                reaction_labels[r.get("label", "<none>")] += 1
                per_model[model]["thoughts"] += 1

    per_model_out = {
        m: {"users": len(per_model_users[m]), **per_model[m]}
        for m in sorted(per_model, key=lambda m: per_model[m]["conversations"], reverse=True)
    }

    return {
        "totals": {
            "users": len(users),
            "conversations": len(data),
            "messages": n_messages,
            "thoughts": sum(reason_labels.values()) + sum(reaction_labels.values()),
            "models": len(models),
        },
        "roles": dict(role_counts),
        "reason_labels": dict(reason_labels.most_common()),
        "reaction_labels": dict(reaction_labels.most_common()),
        "per_model": per_model_out,
    }


def check_totals(stats: dict) -> list[str]:
    """Return a list of mismatch messages (empty == all totals match)."""
    problems = []
    for key, expected in io.EXPECTED_TOTALS.items():
        actual = stats["totals"].get(key)
        if actual != expected:
            problems.append(f"  {key}: expected {expected}, got {actual}")
    return problems


def _print_report(stats: dict, *, assert_totals: bool) -> None:
    t = stats["totals"]
    print("=== Global totals ===")
    for k in ("users", "conversations", "messages", "thoughts", "models"):
        print(f"  {k:>14}: {t[k]:,}")
    print(f"  roles: {stats['roles']}")

    print("\n=== Reason labels (7) ===")
    for k, v in stats["reason_labels"].items():
        print(f"  {k:>34}: {v:,}")
    print("\n=== Reaction labels (5) ===")
    for k, v in stats["reaction_labels"].items():
        print(f"  {k:>34}: {v:,}")

    print("\n=== Per-model (Appendix Table A1) ===")
    print(f"  {'model':<28}{'users':>7}{'convs':>7}{'msgs':>8}{'thoughts':>10}")
    for m, d in stats["per_model"].items():
        print(f"  {m:<28}{d['users']:>7}{d['conversations']:>7}{d['messages']:>8}{d['thoughts']:>10}")

    if assert_totals:
        problems = check_totals(stats)
        print("\n=== Totals check ===")
        if problems:
            print("MISMATCH:")
            print("\n".join(problems))
            raise SystemExit(1)
        print("OK — all global totals match the spec.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 0 dataset stats")
    ap.add_argument("--quick", action="store_true", help="use the 20-conv sample")
    ap.add_argument("--path", default=None, help="path to a JSONL file")
    ap.add_argument("--json", dest="json_out", default=None, help="write stats to JSON")
    args = ap.parse_args()

    data = io.load(args.path or io.DEFAULT_PATH, quick=args.quick)
    stats = compute_stats(data)
    _print_report(stats, assert_totals=not args.quick)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
