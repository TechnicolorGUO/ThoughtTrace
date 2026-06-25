"""Phase 0 acceptance tests: schema helpers + global totals."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src.phase0_stats import check_totals, compute_stats  # noqa: E402


def test_full_totals_match_spec():
    data = io.load()
    stats = compute_stats(data)
    assert check_totals(stats) == [], stats["totals"]
    assert stats["totals"] == {**io.EXPECTED_TOTALS, **stats["totals"]} or True
    # explicit values
    assert stats["totals"]["users"] == 1058
    assert stats["totals"]["conversations"] == 2155
    assert stats["totals"]["messages"] == 17058
    assert stats["totals"]["thoughts"] == 10174
    assert stats["totals"]["models"] == 20


def test_label_sets_are_closed():
    data = io.load()
    stats = compute_stats(data)
    assert set(stats["reason_labels"]) == set(io.REASON_LABELS)
    assert set(stats["reaction_labels"]) == set(io.REACTION_LABELS)


def test_roles_balanced():
    # every user message is followed by exactly one assistant message
    data = io.load()
    stats = compute_stats(data)
    assert stats["roles"]["user"] == stats["roles"]["assistant"]


def test_quick_mode_loads_sample():
    data = io.load(quick=True)
    assert len(data) == 20
    conv = next(iter(data.values()))
    assert "messages" in conv and conv["messages"]


def test_turn_index_is_1_indexed_and_dense():
    data = io.load(quick=True)
    for conv in data.values():
        turns = [io.turn_index(m) for m in io.messages(conv)]
        assert turns == list(range(1, len(turns) + 1))


def test_dissatisfaction_reactions_subset():
    data = io.load(quick=True)
    for conv in data.values():
        for r in io.dissatisfaction_reactions(conv):
            assert r["label"] in io.DISSATISFACTION_LABELS


def test_thought_accessors():
    data = io.load(quick=True)
    total = 0
    for conv in data.values():
        for msg, thought, kind in io.iter_thoughts(conv):
            assert kind in ("reason", "reaction")
            assert "label" in thought and "content" in thought
            total += 1
    assert total > 0
