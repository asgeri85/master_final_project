"""Unit tests for src/eval/metrics.py, verified against hand-computed examples."""

import math

import pytest

from src.eval.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    rank_of_true_item,
)


def test_rank_1_true_item_at_top():
    ranked = [42, 7, 99, 3, 5, 6, 8, 9, 10, 11]
    true_item = 42
    k = 10
    assert rank_of_true_item(ranked, true_item) == 1
    assert hit_rate_at_k(ranked, true_item, k) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(1.0)
    assert mrr_at_k(ranked, true_item, k) == pytest.approx(1.0)


def test_rank_4_within_k_but_not_top():
    ranked = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    true_item = 40  # 4th position
    k = 10
    assert rank_of_true_item(ranked, true_item) == 4
    assert hit_rate_at_k(ranked, true_item, k) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(1.0 / math.log2(5))
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(0.430677, abs=1e-6)
    assert mrr_at_k(ranked, true_item, k) == pytest.approx(0.25)


def test_rank_15_outside_k():
    ranked = list(range(1, 16))  # 15 items
    true_item = 15  # last position, rank 15
    k = 10
    assert rank_of_true_item(ranked, true_item) == 15
    assert hit_rate_at_k(ranked, true_item, k) == pytest.approx(0.0)
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(0.0)
    assert mrr_at_k(ranked, true_item, k) == pytest.approx(0.0)


def test_true_item_absent_entirely_cold_start():
    ranked = [1, 2, 3, 4, 5]
    true_item = 999  # never in the ranking
    k = 10
    assert rank_of_true_item(ranked, true_item) is None
    assert hit_rate_at_k(ranked, true_item, k) == pytest.approx(0.0)
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(0.0)
    assert mrr_at_k(ranked, true_item, k) == pytest.approx(0.0)


def test_rank_exactly_at_boundary_k():
    ranked = list(range(1, 11))  # 10 items, true item at rank 10
    true_item = 10
    k = 10
    assert rank_of_true_item(ranked, true_item) == 10
    assert hit_rate_at_k(ranked, true_item, k) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(1.0 / math.log2(11))
    assert ndcg_at_k(ranked, true_item, k) == pytest.approx(0.289065, abs=1e-6)
    assert mrr_at_k(ranked, true_item, k) == pytest.approx(0.1)
