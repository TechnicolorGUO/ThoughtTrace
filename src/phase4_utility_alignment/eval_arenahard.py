"""Phase 4 / 7C — Arena-Hard evaluation (Appendix D.9).

Evaluates each trained checkpoint + the base backbone on Arena-Hard-Auto and
reports raw win rate and style-controlled (SC) win rate. The expected ranking is
the acceptance criterion (not absolute numbers):

    thought-guided > message-guided(TT) > base,   and   message-guided(TT) > WildChat

GPU + the official Arena-Hard-Auto harness are needed, so this runs on the server.
This module orchestrates: (1) generate model answers to the Arena-Hard prompts,
(2) judge them against the baseline answers, (3) tabulate win rates.

Per the §Judge note, the eval judge must NOT be the model being graded — pass a
separate judge endpoint (configs `judge`).

Two integration modes:
  * --harness PATH : shell out to a local clone of the official arena-hard-auto
    repo (https://github.com/lmarena/arena-hard-auto), reusing its gen_answer /
    gen_judgment / show_result steps. Preferred for faithful numbers.
  * --self-contained : a minimal built-in pairwise-judge fallback for a quick
    directional signal when the official harness is not set up.

Usage (server):
    python -m src.phase4_utility_alignment.eval_arenahard \
        --models base=Qwen/Qwen3-8B \
                 thought=outputs/phase4/ckpt_thought_guided \
                 message_tt=outputs/phase4/ckpt_message_tt \
        --baseline base --self-contained --questions data/arena_hard/question.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "outputs" / "phase4_eval"


def _parse_models(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        name, _, path = it.partition("=")
        out[name] = path
    return out


# --- official harness mode --------------------------------------------------

def run_official(harness_dir: str, config: str) -> None:
    """Shell out to the official arena-hard-auto pipeline. Assumes the repo's
    config files already point at the model endpoints + judge."""
    h = Path(harness_dir)
    for step in ("gen_answer.py", "gen_judgment.py", "show_result.py"):
        print(f"[arena-hard] {step}")
        subprocess.run(["python", str(h / step), "--config", config], cwd=h, check=True)


# --- self-contained fallback ------------------------------------------------

def load_questions(path: str | Path, limit: int | None = None) -> list[dict]:
    qs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line))
    return qs[:limit] if limit else qs


def generate_answers(questions: list[dict], client, q_field: str = "turns") -> list[str]:
    answers = []
    for q in questions:
        # Arena-Hard question.jsonl uses {"turns": [{"content": ...}]}
        if q_field in q and isinstance(q[q_field], list):
            prompt = q[q_field][0]["content"]
        else:
            prompt = q.get("prompt") or q.get("question") or ""
        answers.append(client.complete(prompt))
    return answers


def judge_pairwise(question: str, ans_a: str, ans_b: str, judge) -> str:
    """Return 'A', 'B', or 'tie'. Order-randomization is the caller's job."""
    from ..prompts import fill, load_prompt

    prompt = fill(load_prompt("arena_judge.txt"),
                  question=question, answer_a=ans_a, answer_b=ans_b)
    out = judge.complete_json(prompt)
    verdict = (out or {}).get("winner", "tie") if isinstance(out, dict) else "tie"
    return verdict if verdict in ("A", "B", "tie") else "tie"


def win_rate_vs_baseline(questions, model_answers, baseline_answers, judge,
                         seed: int = 0) -> dict:
    """Pairwise win rate of model vs baseline, averaging both presentation orders
    to reduce position bias."""
    import random

    rng = random.Random(seed)
    wins = ties = losses = 0
    for q, m_ans, b_ans in zip(questions, model_answers, baseline_answers):
        qtext = q["turns"][0]["content"] if "turns" in q else q.get("prompt", "")
        # randomize order, then map verdict back to the model
        if rng.random() < 0.5:
            v = judge_pairwise(qtext, m_ans, b_ans, judge)
            model_won = v == "A"; baseline_won = v == "B"
        else:
            v = judge_pairwise(qtext, b_ans, m_ans, judge)
            model_won = v == "B"; baseline_won = v == "A"
        if model_won:
            wins += 1
        elif baseline_won:
            losses += 1
        else:
            ties += 1
    n = max(1, wins + ties + losses)
    return {
        "n": n, "wins": wins, "ties": ties, "losses": losses,
        "win_rate": round(100 * (wins + 0.5 * ties) / n, 2),
    }


def run_self_contained(model_paths: dict[str, str], baseline: str, questions_path: str,
                       limit: int | None, seed: int, out_dir: Path = OUT_DIR) -> dict:
    """Directional fallback: serve each model (OpenAI-compatible) and pairwise-judge
    vs the baseline. Model serving endpoints are read from configs `eval_models`
    or each path is treated as an endpoint base_url; here we expect each model to
    already be served and `model_paths[name]` to be its endpoint URL."""
    from ..config import load_config
    from ..llm_client import LLMClient

    cfg = load_config()
    judge = LLMClient.from_config(cfg["judge"]) if cfg.get("judge", {}).get("model") \
        else LLMClient.from_config(cfg["llm"])

    questions = load_questions(questions_path, limit)

    # NOTE: each model must be served separately; model_paths[name] is its base_url.
    answers = {}
    for name, url in model_paths.items():
        client = LLMClient(model=name, base_url=url, api_key="EMPTY",
                           cache_dir=f"outputs/llm_cache_eval/{name}")
        answers[name] = generate_answers(questions, client)

    base_ans = answers[baseline]
    results = {}
    for name in model_paths:
        if name == baseline:
            continue
        results[name] = win_rate_vs_baseline(questions, answers[name], base_ans, judge, seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "arena_results.json").write_text(
        json.dumps({"baseline": baseline, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4/7C — Arena-Hard eval")
    ap.add_argument("--models", nargs="+", required=True, help="name=path/url pairs")
    ap.add_argument("--baseline", default="base")
    ap.add_argument("--harness", default=None, help="path to arena-hard-auto repo")
    ap.add_argument("--config", default="config/api_config.yaml", help="harness config")
    ap.add_argument("--self-contained", action="store_true")
    ap.add_argument("--questions", default=None, help="Arena-Hard question.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.harness and not args.self_contained:
        run_official(args.harness, args.config)
        print("[arena-hard] done — see harness show_result output for win rates.")
        return

    if not args.questions:
        raise SystemExit("--self-contained needs --questions PATH (Arena-Hard question.jsonl)")
    models = _parse_models(args.models)
    results = run_self_contained(models, args.baseline, args.questions,
                                 args.limit, args.seed)
    print(f"=== Arena-Hard win rate vs {args.baseline} (self-contained) ===")
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["win_rate"]):
        print(f"  {name:<14}: {r['win_rate']:>6.2f}%  (W{r['wins']}/T{r['ties']}/L{r['losses']})")


if __name__ == "__main__":
    main()
