"""Phase 1B tests: length stats. Turn counts need no tokenizer, so the headline
'median 8 turns' check runs anywhere; token assertions only check structure
(exact values depend on whether tiktoken is installed)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import io_utils as io  # noqa: E402
from src import phase1_conversation_props as p1  # noqa: E402


def test_median_8_turns():
    data = io.load()
    stats = p1.compute_length_stats(data)
    # the paper's headline length claim, independent of tokenizer
    assert stats["turns_per_conv"]["summary"]["median"] == 8
    assert stats["turns_per_conv"]["summary"]["n"] == 2155


def test_turn_hist_reconciles():
    data = io.load()
    stats = p1.compute_length_stats(data)
    assert sum(stats["turns_per_conv"]["hist"].values()) == 2155


def test_per_role_token_counts_have_full_support():
    data = io.load()
    stats = p1.compute_length_stats(data)
    # every user and assistant message contributes one length sample
    assert stats["user_msg_tokens"]["summary"]["n"] == 8529
    assert stats["assistant_msg_tokens"]["summary"]["n"] == 8529


def test_assistant_longer_than_user():
    data = io.load()
    stats = p1.compute_length_stats(data)
    u = stats["user_msg_tokens"]["summary"]["median"]
    a = stats["assistant_msg_tokens"]["summary"]["median"]
    assert a > u  # assistants write much more than users, regardless of tokenizer


def test_bucketize_edges():
    b = p1._bucketize([1, 2, 3, 4, 33, 100], p1.TURN_BUCKETS, p1.TURN_BUCKET_LABELS)
    assert b["1"] == 1 and b["2"] == 1 and b["3-4"] == 2 and b["33+"] == 2
