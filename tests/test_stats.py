"""Unit tests for src/eval/stats.py, verified against hand-computed examples."""

import pytest

from src.eval.stats import holm_bonferroni


def test_all_significant_ascending_order():
    # m=3, sorted ascending: 0.01, 0.02, 0.03 -> raw candidates 3*0.01=0.03,
    # 2*0.02=0.04, 1*0.03=0.03; running max keeps the last one at 0.04 too.
    pvalues = [0.01, 0.02, 0.03]
    adjusted = holm_bonferroni(pvalues)
    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


def test_running_max_enforces_monotonicity():
    # m=3, sorted ascending: 0.01, 0.02, 0.005 -> raw candidates 3*0.01=0.03,
    # 2*0.02=0.04, 1*0.005=0.005; but 0.005 sorts first (rank 0, mult 3) giving
    # 0.015, then 0.01 (rank 1, mult 2) giving 0.02, then 0.02 (rank 2, mult 1)
    # giving 0.02 -- running max keeps the sequence non-decreasing in rank order.
    pvalues = [0.01, 0.02, 0.005]
    adjusted = holm_bonferroni(pvalues)
    assert adjusted == pytest.approx([0.02, 0.02, 0.015])


def test_capped_at_one():
    pvalues = [0.9, 0.95]
    adjusted = holm_bonferroni(pvalues)
    assert adjusted == pytest.approx([1.0, 1.0])


def test_single_pvalue_unchanged():
    assert holm_bonferroni([0.03]) == pytest.approx([0.03])


def test_order_preserved_matches_input_order():
    pvalues = [0.5, 0.001, 0.2, 0.04]
    adjusted = holm_bonferroni(pvalues)
    assert len(adjusted) == len(pvalues)
    # smallest raw p-value (index 1) should get the smallest adjusted p-value
    assert adjusted[1] == min(adjusted)
