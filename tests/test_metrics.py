"""Unit tests for src/eval/metrics.py, verified against hand-computed examples."""

import math

import pytest

from src.eval.metrics import (
    average_multi,
    evaluate_pair_multi,
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


# LSApp's HR@1/HR@3/MRR@5 scheme (docs/00-karar-notu.md SS3) uses three
# different K values in one pass, unlike RetailRocket's single shared K.
_LSAPP_SPECS = [("hr", 1), ("hr", 3), ("mrr", 5)]


def test_evaluate_pair_multi_rank_2_hits_hr3_and_mrr5_not_hr1():
    ranked = [10, 42, 30, 40, 50]
    true_item = 42  # rank 2
    result = evaluate_pair_multi(ranked, true_item, _LSAPP_SPECS)
    assert result == {
        "hr@1": pytest.approx(0.0),
        "hr@3": pytest.approx(1.0),
        "mrr@5": pytest.approx(0.5),  # 1/2
    }


def test_evaluate_pair_multi_rank_1_hits_everything():
    ranked = [42, 7, 99, 3, 5]
    true_item = 42  # rank 1
    result = evaluate_pair_multi(ranked, true_item, _LSAPP_SPECS)
    assert result == {
        "hr@1": pytest.approx(1.0),
        "hr@3": pytest.approx(1.0),
        "mrr@5": pytest.approx(1.0),
    }


def test_evaluate_pair_multi_rank_outside_all_ks():
    ranked = list(range(1, 11))  # true item at rank 10
    true_item = 10
    result = evaluate_pair_multi(ranked, true_item, _LSAPP_SPECS)
    assert result == {"hr@1": pytest.approx(0.0), "hr@3": pytest.approx(0.0), "mrr@5": pytest.approx(0.0)}


def test_evaluate_pair_multi_true_item_absent():
    ranked = [1, 2, 3, 4, 5]
    true_item = 999
    result = evaluate_pair_multi(ranked, true_item, _LSAPP_SPECS)
    assert result == {"hr@1": pytest.approx(0.0), "hr@3": pytest.approx(0.0), "mrr@5": pytest.approx(0.0)}


def test_evaluate_pair_multi_unknown_metric_name_raises():
    with pytest.raises(ValueError):
        evaluate_pair_multi([1, 2, 3], 1, [("recall", 5)])


def test_average_multi_hand_computed():
    # pair A: true item at rank 1 -> hr@1=1, hr@3=1, mrr@5=1
    # pair B: true item at rank 2 -> hr@1=0, hr@3=1, mrr@5=0.5
    # pair C: true item absent    -> hr@1=0, hr@3=0, mrr@5=0
    pairs = [
        ([42, 7, 99], 42),
        ([10, 42, 30], 42),
        ([1, 2, 3], 999),
    ]
    result = average_multi(pairs, _LSAPP_SPECS)
    assert result == {
        "hr@1": pytest.approx((1.0 + 0.0 + 0.0) / 3),
        "hr@3": pytest.approx((1.0 + 1.0 + 0.0) / 3),
        "mrr@5": pytest.approx((1.0 + 0.5 + 0.0) / 3),
    }


def test_average_multi_empty_raises():
    with pytest.raises(ValueError):
        average_multi([], _LSAPP_SPECS)
