"""Phase 2C tests: gold distributions (no model) + scoring/sampling logic.

The labeler-validation LLM path is exercised with a fake client so the metric
plumbing is tested without a server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src import phase2_thought_props as p2  # noqa: E402


def test_gold_distribution_totals_match_phase0():
    data = io.load()
    dist = p2.compute_gold_distributions(data)
    # the two thought streams must sum to the global thought total
    assert dist["n_reasons"] + dist["n_reactions"] == io.EXPECTED_TOTALS["thoughts"]
    assert set(dist["reasons"]) == set(io.REASON_LABELS)
    assert set(dist["reactions"]) == set(io.REACTION_LABELS)


def test_gold_distribution_known_values():
    data = io.load()
    dist = p2.compute_gold_distributions(data)
    # most common in each stream (per the paper's narrative)
    assert max(dist["reasons"], key=dist["reasons"].get) == "task_motivation"
    assert max(dist["reactions"], key=dist["reactions"].get) == "explicit_affirmation"
    assert dist["reasons"]["task_motivation"] == 1680
    assert dist["reactions"]["explicit_affirmation"] == 4099


def test_macro_f1_perfect_and_chance():
    labels = ("a", "b")
    perfect = p2._macro_f1(["a", "b", "a"], ["a", "b", "a"], labels)
    assert perfect["accuracy"] == 1.0
    assert perfect["macro_f1"] == 1.0
    half = p2._macro_f1(["a", "a", "b", "b"], ["a", "b", "a", "b"], labels)
    assert half["accuracy"] == 0.5


def test_sample_is_deterministic():
    data = io.load(quick=True)
    s1 = p2._sample_thoughts(data, "reason", 5, seed=0)
    s2 = p2._sample_thoughts(data, "reason", 5, seed=0)
    assert [t["content"] for _, t in s1] == [t["content"] for _, t in s2]


def test_validate_labeler_with_fake_client():
    """A perfect fake labeler should yield accuracy/macro_f1 == 1.0."""
    data = io.load(quick=True)

    class PerfectClient:
        # returns the gold label by reading it out of the prompt's thought —
        # instead we cheat via a closure over the sample below.
        def __init__(self, lookup):
            self.lookup = lookup

        def complete_json(self, prompt):
            for thought_text, label in self.lookup.items():
                if thought_text and thought_text[:60] in prompt:
                    return {"label": label}
            return {"label": "<invalid>"}

    sample = p2._sample_thoughts(data, "reason", 5, seed=0)
    lookup = {t["content"]: t["label"] for _, t in sample}
    client = PerfectClient(lookup)
    res = p2.validate_labeler(data, client, kind="reason", n=5, seed=0)
    assert res["n"] == 5
    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
