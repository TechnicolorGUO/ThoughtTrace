"""Phase 2D tests: stage binning + the three headline stage-dynamics trends."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src import phase2_thought_props as p2  # noqa: E402


def test_stage_of_edges():
    assert p2.stage_of(1, 1) == 0           # degenerate single turn
    assert p2.stage_of(1, 9) == 0           # first turn -> Early
    assert p2.stage_of(9, 9) == 3           # last turn -> Late
    assert p2.stage_of(5, 9) == 2           # midpoint -> Mid-Late side
    for t in range(1, 10):
        assert 0 <= p2.stage_of(t, 9) <= 3


def test_stage_counts_reconcile_with_gold_totals():
    data = io.load()
    sd = p2.compute_stage_dynamics(data)
    n_reason = sum(sum(d.values()) for d in sd["reason"].values())
    n_reaction = sum(sum(d.values()) for d in sd["reaction"].values())
    assert n_reason == 4498        # == Phase 2C n_reasons
    assert n_reaction == 5676      # == Phase 2C n_reactions


def test_task_motivation_declines_over_stages():
    data = io.load()
    sd = p2.compute_stage_dynamics(data)
    share = p2._stage_share(sd, "reason", "task_motivation")
    assert share[0] > share[-1]           # dominates early
    assert share == sorted(share, reverse=True)  # monotonic decline


def test_task_continuation_rises_over_stages():
    data = io.load()
    sd = p2.compute_stage_dynamics(data)
    share = p2._stage_share(sd, "reason", "task_continuation")
    assert share[-1] > share[0]           # takes over later


def test_explicit_affirmation_rises_over_stages():
    data = io.load()
    sd = p2.compute_stage_dynamics(data)
    share = p2._stage_share(sd, "reaction", "explicit_affirmation")
    assert share == sorted(share)         # steadily rises to convergence


def test_length_crosstab_reconciles():
    data = io.load()
    ct = p2.compute_length_crosstab(data)
    n_reason = sum(sum(d.values()) for d in ct["reason"].values())
    n_reaction = sum(sum(d.values()) for d in ct["reaction"].values())
    assert n_reason == 4498 and n_reaction == 5676
