"""Phase 3 tests: candidate construction, context rendering, bootstrap, and a
full run wired to deterministic fake clients (no server)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src import phase3_utility_prediction as p3  # noqa: E402


def test_candidates_are_assistant_then_user_with_reaction():
    data = io.load(quick=True)
    cands = list(p3.iter_candidates(data))
    assert cands
    for c in cands:
        msgs = c["context_msgs"]
        assert msgs[-1]["type"] == io.ASSISTANT          # ends on assistant turn
        assert io.get_reaction(msgs[-1]) is not None      # carries a reaction
        assert isinstance(c["target"], str)


def test_render_context_thoughts_only_when_augmented():
    data = io.load(quick=True)
    cand = next(p3.iter_candidates(data))
    plain = p3.render_context(cand["context_msgs"], augmented=False)
    aug = p3.render_context(cand["context_msgs"], augmented=True)
    assert "[THOUGHT" not in plain
    assert "[THOUGHT" in aug                              # reaction is interleaved
    assert len(aug) > len(plain)


def test_bootstrap_excludes_zero_for_clear_positive():
    deltas = [5.0] * 50
    ci = p3.bootstrap_delta_ci(deltas, n_boot=500, seed=0)
    assert ci["mean"] == 5.0
    assert ci["excludes_zero"] is True


def test_bootstrap_includes_zero_for_noise():
    deltas = [5, -5, 4, -4, 3, -3, 0, 1, -1, 0]
    ci = p3.bootstrap_delta_ci(deltas, n_boot=500, seed=0)
    assert ci["lo"] < 0 < ci["hi"]
    assert ci["excludes_zero"] is False


def test_full_run_with_fake_clients():
    """Predictor that always echoes the target -> thought arm scores >= history.
    Uses fakes so the orchestration (filter, both arms, scoring, CI) is covered
    without a server."""
    data = io.load(quick=True)

    class FakeClient:
        def complete(self, prompt):
            # 'predict' the gold target by lifting it out of the rendered context
            # is not possible here; just return a fixed string so both arms are equal.
            return "some predicted message"

        def complete_json(self, prompt):
            if '"score": <integer 1-5>' in prompt or "informative" in prompt:
                return {"score": 5}            # pass the quality filter
            return {"score": 50}               # similarity

    res = p3.run(data, FakeClient(), n=3, seed=0)
    assert res["n_kept_after_quality_filter"] == 3
    assert res["mean_sim_history"] == 50.0
    assert res["mean_sim_thought"] == 50.0
    assert res["delta"] == 0.0
