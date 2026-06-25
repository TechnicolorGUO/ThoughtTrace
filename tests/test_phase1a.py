"""Phase 1A tests: demographics cleaning + reconciliation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src import phase1_conversation_props as p1  # noqa: E402


def test_age_bracket_filters_garbage():
    assert p1._age_bracket("2") is None       # too young
    assert p1._age_bracket("366") is None      # impossible
    assert p1._age_bracket("17") is None       # below 18
    assert p1._age_bracket("18") == "18-24"
    assert p1._age_bracket("30") == "25-34"
    assert p1._age_bracket("70") == "65+"
    assert p1._age_bracket("") is None
    assert p1._age_bracket(None) is None


def test_purpose_groups_multilabel():
    g = p1._purpose_groups("Coding, working, and translating documents")
    assert "Coding" in g and "Working" in g and "Translation" in g
    assert p1._purpose_groups("") == set()


def test_demographics_reconcile_to_survey_count():
    data = io.load()
    demo = p1.compute_demographics(data)
    assert demo["n_total"] == 2155
    assert demo["n_with_survey"] == 2080
    assert demo["n_with_survey"] + demo["missing"]["survey"] == demo["n_total"]

    # each panel + its missing (+ dropped, for age) must reconcile to n_with_survey
    n = demo["n_with_survey"]
    p = demo["panels"]
    assert sum(p["age"].values()) + demo["missing"].get("age", 0) \
        + demo["dropped_age_out_of_range"] == n
    assert sum(p["gender"].values()) + demo["missing"].get("gender", 0) == n
    assert sum(p["education"].values()) + demo["missing"].get("education", 0) == n
    assert sum(p["frequency"].values()) + demo["missing"].get("frequency", 0) == n
    # occupation top-8 + Other + missing == n
    assert sum(p["occupation"].values()) + demo["missing"].get("occupation", 0) == n


def test_occupation_top8_plus_other():
    data = io.load()
    demo = p1.compute_demographics(data)
    occ = demo["panels"]["occupation"]
    assert len([k for k in occ if k != "Other"]) == 8
    # display picks most-common casing: "Student" not "student"
    assert "Student" in occ


def test_frequency_only_1_to_5():
    data = io.load()
    demo = p1.compute_demographics(data)
    assert set(demo["panels"]["frequency"]) <= {"1", "2", "3", "4", "5"}
