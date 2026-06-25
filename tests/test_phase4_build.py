"""Phase 4 / 7A tests: DPO data construction (pure parts + fake-client builders)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src.phase4_utility_alignment import build_dpo_data as b  # noqa: E402


def test_thought_ok_filter():
    assert b.thought_ok("this reaction has at least six words here") is True
    assert b.thought_ok("too short") is False           # < 6 words
    assert b.thought_ok("") is False
    assert b.thought_ok(None) is False
    assert b.thought_ok("1 2 3 4 5 6 7 8") is False      # no alphabetic chars


def test_thought_guided_pool_size_and_shape():
    data = io.load()
    cands = b.collect_thought_guided(data)
    assert len(cands) == 1148                            # gold pool, target 1000 reachable
    for c in cands:
        assert c["label"] in io.DISSATISFACTION_LABELS
        assert c["context_msgs"]                         # has prior context
        assert c["context_msgs"][-1]["type"] == io.USER  # ends on the prompting user turn
        assert b.thought_ok(c["signal"])


def test_message_candidates_end_on_user_and_have_context():
    data = io.load()
    cands = b.collect_message_candidates(data)
    assert len(cands) == 6128
    for c in cands[:50]:
        assert c["context_msgs"][-1]["type"] == io.USER


def test_thought_pool_exceeds_message_target_ratio():
    """Thoughts surface ~2.2x more dissatisfaction than the message target."""
    data = io.load()
    assert len(b.collect_thought_guided(data)) / 450 > 2.0


def test_build_thought_guided_with_fake_rewriter():
    data = io.load(quick=True)

    class FakeRewriter:
        def complete(self, prompt):
            return "IMPROVED RESPONSE"

    pairs = b.build_thought_guided(data, FakeRewriter(), n=3, seed=0)
    assert len(pairs) <= 3
    for p in pairs:
        assert p["chosen"] == "IMPROVED RESPONSE"
        assert p["rejected"] and p["chosen"] != p["rejected"]
        assert p["prompt"][-1]["role"] == io.USER
        assert p["meta"]["arm"] == "thought_guided"


def test_build_message_guided_with_fake_clients():
    data = io.load(quick=True)

    class FakeClient:
        def complete(self, prompt):
            return "IMPROVED"

        def complete_json(self, prompt):
            return {"dissatisfied": True}      # mark everything dissatisfied

    pairs = b.build_message_guided(data, FakeClient(), n=2, seed=0)
    assert len(pairs) <= 2
    for p in pairs:
        assert p["meta"]["arm"] == "message_guided"
        assert p["chosen"] == "IMPROVED"
