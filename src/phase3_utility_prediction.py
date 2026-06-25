"""Phase 3 — Utility 1: thoughts predict user behavior (Appendix D.8).

The with-thought vs without-thought contrast. For each candidate point in a
conversation, predict the user's NEXT message twice:

  * history-only       — context = raw dialogue up to the last assistant turn
  * thought-augmented  — same context, but the user's private thoughts (reasons
                         on their messages, reactions to the assistant) are
                         interleaved at each turn

Each prediction is scored 0-100 for semantic similarity to the actual next
message by an LLM judge. The headline metric is the DELTA (does adding thoughts
help?), reported with a bootstrap CI. Paper: +41.7% relative.

All steps are inference-only, so this runs against any OpenAI-compatible endpoint
(local Qwen3-8B via vLLM, or a hosted API like DeepSeek). Per the §Judge note,
pass a separate ``judge`` client when possible so predictor != judge.

Pipeline (D.8):
  1. candidates = assistant message followed by a user turn, carrying a reaction
  2. quality filter: keep only candidates whose reaction scores >= 4 (informative)
  3. predict next message under both contexts
  4. score each prediction 0-100 vs the actual next message
  5. report mean(history) vs mean(thought) and the bootstrapped delta

Usage (needs an endpoint):
    python -m src.phase3_utility_prediction --n 100
    python -m src.phase3_utility_prediction --n 100 --quick
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import io_utils as io
from .prompts import fill, load_prompt

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "phase3"
QUALITY_THRESHOLD = 4


# --- candidate construction -------------------------------------------------

def iter_candidates(data: dict[str, dict]):
    """Yield prediction candidates.

    A candidate is an assistant message that (a) is followed by a user turn (the
    target to predict) and (b) carries a reaction — the immediate thought before
    the user's next message, and the signal the augmented arm adds.
    """
    for conv in data.values():
        msgs = io.messages(conv)
        for i, m in enumerate(msgs):
            if m.get("type") != io.ASSISTANT:
                continue
            if i + 1 >= len(msgs) or msgs[i + 1].get("type") != io.USER:
                continue
            reaction = io.get_reaction(m)
            if not reaction or not (reaction.get("content") or "").strip():
                continue
            yield {
                "conv_id": conv["id"],
                "index": i,
                "context_msgs": msgs[: i + 1],   # up to & including this assistant turn
                "target": msgs[i + 1].get("content", ""),
                "reaction": reaction,
            }


def render_context(context_msgs: list[dict], augmented: bool) -> str:
    """Render the dialogue. When augmented, interleave the user's thoughts as
    [THOUGHT] lines (reasons after user turns, reactions after assistant turns)."""
    lines = []
    for m in context_msgs:
        role = "User" if m.get("type") == io.USER else "Assistant"
        lines.append(f"{role}: {m.get('content', '').strip()}")
        if not augmented:
            continue
        if m.get("type") == io.USER:
            for r in io.get_reasons(m):
                if (r.get("content") or "").strip():
                    lines.append(f"  [THOUGHT — user's reason]: {r['content'].strip()}")
        else:
            for r in io.get_reactions(m):
                if (r.get("content") or "").strip():
                    lines.append(f"  [THOUGHT — user's reaction]: {r['content'].strip()}")
    return "\n".join(lines)


# --- LLM steps --------------------------------------------------------------

def thought_quality(cand: dict, judge) -> int:
    """Score the candidate's reaction 1-5 for informativeness."""
    prompt = fill(
        load_prompt("thought_quality.txt"),
        message=cand["context_msgs"][-1].get("content", ""),
        thought=cand["reaction"].get("content", ""),
    )
    out = judge.complete_json(prompt)
    try:
        return int((out or {}).get("score", 0))
    except (TypeError, ValueError):
        return 0


def predict_next(cand: dict, client, augmented: bool) -> str:
    template = load_prompt(
        "nextmsg_predict_thoughts.txt" if augmented else "nextmsg_predict_context.txt"
    )
    prompt = fill(template, context=render_context(cand["context_msgs"], augmented))
    return client.complete(prompt).strip()


def similarity(predicted: str, actual: str, judge) -> float:
    prompt = fill(load_prompt("nextmsg_judge.txt"), predicted=predicted, actual=actual)
    out = judge.complete_json(prompt)
    try:
        return float((out or {}).get("score", 0))
    except (TypeError, ValueError):
        return 0.0


# --- bootstrap --------------------------------------------------------------

def bootstrap_delta_ci(deltas: list[float], n_boot: int = 2000, seed: int = 0,
                       alpha: float = 0.05) -> dict:
    if not deltas:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "excludes_zero": False}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(n_boot):
        resample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return {
        "mean": sum(deltas) / n,
        "lo": lo,
        "hi": hi,
        "excludes_zero": lo > 0 or hi < 0,
    }


# --- orchestration ----------------------------------------------------------

def run(data: dict[str, dict], client, judge=None, n: int | None = 100,
        seed: int = 0, out_dir: Path = OUT_DIR) -> dict:
    judge = judge or client
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = list(iter_candidates(data))
    rng = random.Random(seed)
    rng.shuffle(candidates)

    rows = []
    sims_hist, sims_thought, deltas = [], [], []
    n_considered = n_kept = 0

    for cand in candidates:
        if n is not None and n_kept >= n:
            break
        n_considered += 1
        q = thought_quality(cand, judge)
        if q < QUALITY_THRESHOLD:
            continue
        n_kept += 1

        pred_h = predict_next(cand, client, augmented=False)
        pred_t = predict_next(cand, client, augmented=True)
        s_h = similarity(pred_h, cand["target"], judge)
        s_t = similarity(pred_t, cand["target"], judge)

        sims_hist.append(s_h)
        sims_thought.append(s_t)
        deltas.append(s_t - s_h)
        rows.append({
            "conv_id": cand["conv_id"], "index": cand["index"],
            "quality": q, "sim_history": s_h, "sim_thought": s_t, "delta": s_t - s_h,
            "target": cand["target"][:200],
            "pred_history": pred_h[:200], "pred_thought": pred_t[:200],
        })

    mean_h = sum(sims_hist) / len(sims_hist) if sims_hist else 0.0
    mean_t = sum(sims_thought) / len(sims_thought) if sims_thought else 0.0
    ci = bootstrap_delta_ci(deltas, seed=seed)
    rel = 100 * (mean_t - mean_h) / mean_h if mean_h else 0.0

    result = {
        "n_candidates_total": len(candidates),
        "n_considered": n_considered,
        "n_kept_after_quality_filter": n_kept,
        "quality_threshold": QUALITY_THRESHOLD,
        "mean_sim_history": round(mean_h, 2),
        "mean_sim_thought": round(mean_t, 2),
        "delta": round(mean_t - mean_h, 2),
        "relative_gain_pct": round(rel, 1),
        "bootstrap_delta_ci": ci,
    }
    (out_dir / "prediction_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "prediction_rows.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 — Utility 1 next-message prediction")
    ap.add_argument("--n", type=int, default=100, help="kept candidates after filter")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    from .config import load_config
    from .llm_client import LLMClient

    cfg = load_config(args.config) if args.config else load_config()
    client = LLMClient.from_config(cfg["llm"])
    judge = LLMClient.from_config(cfg["judge"]) if cfg.get("judge", {}).get("model") else client

    data = io.load(quick=args.quick)
    res = run(data, client, judge=judge, n=args.n, seed=args.seed)
    print("=== Phase 3 — Utility 1 (next-message prediction) ===")
    print(f"  candidates total      : {res['n_candidates_total']}")
    print(f"  kept (quality >= {res['quality_threshold']})   : {res['n_kept_after_quality_filter']} "
          f"of {res['n_considered']} considered")
    print(f"  mean sim (history)    : {res['mean_sim_history']}")
    print(f"  mean sim (thought)    : {res['mean_sim_thought']}")
    print(f"  delta                 : {res['delta']}  ({res['relative_gain_pct']}% relative)")
    ci = res["bootstrap_delta_ci"]
    print(f"  bootstrap 95% CI delta: [{ci['lo']:.2f}, {ci['hi']:.2f}]  "
          f"excludes 0: {ci['excludes_zero']}")


if __name__ == "__main__":
    main()
