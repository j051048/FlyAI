import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "phase0"))
from ngram_draft import NgramDrafter, simulate_g


def test_ngram_adaptive_margin_short_context():
    """P0-1 regression test: ensure sequences <= 256 tokens do NOT starve table_entries."""
    drafter = NgramDrafter(ng=3)  # default margin=256, adaptive=True by default
    # Simulate a repetitive prompt + generated sequence of length 150
    # Repeating pattern: [10, 20, 30, 40]
    pattern = [10, 20, 30, 40]
    seq = (pattern * 38)[:150]

    # propose should index anchors and find match
    proposals = drafter.propose(seq, k=4)
    metrics = drafter.metrics()

    # Table entries MUST be populated (not starved to 0)
    assert metrics["table_entries"] > 0, "Adaptive margin failed: table_entries is 0 on 150-tok context"
    # Propose must return a real match, not fallback padding
    assert drafter.matched is True
    # The suffix of seq[:150] is [..., 40, 10, 20], so continuation is [30, 40, 10, 20]
    assert proposals == [30, 40, 10, 20]
    assert metrics["ngram_matched"] == 1
    assert metrics["ngram_silent"] == 0


def test_ngram_fixed_margin_env_override(monkeypatch):
    """Ensure explicit NGRAM_MARGIN override honors fixed cap if requested."""
    monkeypatch.setenv("NGRAM_MARGIN", "300")
    drafter = NgramDrafter(ng=3)
    assert drafter.adaptive is False
    assert drafter.margin_cap == 300

    seq = list(range(100))
    drafter.propose(seq, k=4)
    # With margin 300 and seq 100, stable <= 0 so table is empty
    assert len(drafter.table) == 0


def test_ngram_workload_metrics_and_silent_ratio():
    """P1-1: Check workload telemetry accounting (silent_ratio & accept reporting)."""
    drafter = NgramDrafter(ng=3)
    # 1. First round with no repetition -> silent fallback
    seq1 = [1, 2, 3, 4, 5]
    res1 = drafter.propose(seq1, k=4)
    assert res1 == [5, 5, 5, 5]
    drafter.note_accepted(0)

    # 2. Add repetitive segment -> match
    seq2 = [1, 2, 3, 4, 5, 1, 2, 3]
    res2 = drafter.propose(seq2, k=2)
    drafter.note_accepted(2)

    m = drafter.metrics()
    assert m["ngram_rounds"] == 2
    assert m["ngram_silent"] == 1
    assert m["ngram_matched"] == 1
    assert m["silent_ratio"] == 0.5
    assert m["mean_accept"] == 1.0


def test_simulate_g_short_context():
    """Ensure simulate_g works correctly on short sequence without division by zero."""
    pattern = [1, 2, 3, 4, 5]
    seq = (pattern * 20)[:100]
    g, traversals = simulate_g(seq, prompt_len=20, ng=3, k=4)
    assert g >= 1.0
    assert traversals > 0
