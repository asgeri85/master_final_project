"""Ranking metrics for next-item prediction: HR@K, NDCG@K, MRR@K.

No negative sampling: `ranked_items` is always assumed to be a ranking over
the full item catalog (Krichene & Rendle, 2020 — sampled-negative evaluation
biases model comparison).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np


def rank_of_true_item(ranked_items: Sequence[int], true_item: int) -> int | None:
    """1-indexed rank of true_item in ranked_items, or None if absent."""
    positions = np.flatnonzero(np.asarray(ranked_items) == true_item)
    if positions.size == 0:
        return None
    return int(positions[0]) + 1


def hit_rate_at_k(ranked_items: Sequence[int], true_item: int, k: int) -> float:
    """HR@K = 1.0 if the true item's rank is <= k, else 0.0."""
    rank = rank_of_true_item(ranked_items, true_item)
    return 1.0 if rank is not None and rank <= k else 0.0


def ndcg_at_k(ranked_items: Sequence[int], true_item: int, k: int) -> float:
    """NDCG@K for a single relevant item: 1/log2(rank+1) if rank <= k, else 0.0.

    IDCG@K is always 1 in the single-relevant-item case (the ideal ranking
    places it at position 1), so DCG@K/IDCG@K collapses to this form.
    """
    rank = rank_of_true_item(ranked_items, true_item)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def mrr_at_k(ranked_items: Sequence[int], true_item: int, k: int) -> float:
    """MRR@K = 1/rank if rank <= k, else 0.0."""
    rank = rank_of_true_item(ranked_items, true_item)
    if rank is None or rank > k:
        return 0.0
    return 1.0 / rank


def evaluate_pair(ranked_items: Sequence[int], true_item: int, k: int) -> dict[str, float]:
    """Compute HR/NDCG/MRR@K for one (ranked_items, true_item) pair.

    Shares a single rank lookup across all three metrics to avoid three
    independent O(catalog_size) scans per evaluation pair.
    """
    rank = rank_of_true_item(ranked_items, true_item)
    hit = rank is not None and rank <= k
    return {
        f"hr@{k}": 1.0 if hit else 0.0,
        f"ndcg@{k}": (1.0 / math.log2(rank + 1)) if hit else 0.0,
        f"mrr@{k}": (1.0 / rank) if hit else 0.0,
    }


def average_at_k(pairs: Iterable[tuple[Sequence[int], int]], k: int) -> dict[str, float]:
    """Mean HR@K/NDCG@K/MRR@K over many (ranked_items, true_item) pairs."""
    totals = {f"hr@{k}": 0.0, f"ndcg@{k}": 0.0, f"mrr@{k}": 0.0}
    count = 0
    for ranked_items, true_item in pairs:
        result = evaluate_pair(ranked_items, true_item, k)
        for key in totals:
            totals[key] += result[key]
        count += 1
    if count == 0:
        raise ValueError("average_at_k received no pairs to evaluate")
    return {key: value / count for key, value in totals.items()}


def evaluate_pair_multi(
    ranked_items: Sequence[int], true_item: int, specs: Sequence[tuple[str, int]]
) -> dict[str, float]:
    """Compute an arbitrary set of (metric_name, k) pairs for one eval pair.

    Unlike `evaluate_pair`, `specs` need not share a single K -- this is what
    LSApp's HR@1/HR@3/MRR@5 scheme needs (karar-notu SS3's LSApp exception),
    where the default single-K trio doesn't apply. Shares one rank lookup
    across every spec regardless of how many distinct K values it contains.
    """
    rank = rank_of_true_item(ranked_items, true_item)
    out: dict[str, float] = {}
    for name, k in specs:
        hit = rank is not None and rank <= k
        if name == "hr":
            out[f"hr@{k}"] = 1.0 if hit else 0.0
        elif name == "ndcg":
            out[f"ndcg@{k}"] = (1.0 / math.log2(rank + 1)) if hit else 0.0
        elif name == "mrr":
            out[f"mrr@{k}"] = (1.0 / rank) if hit else 0.0
        else:
            raise ValueError(f"unknown metric name {name!r}, expected one of 'hr', 'ndcg', 'mrr'")
    return out


def average_multi(
    pairs: Iterable[tuple[Sequence[int], int]], specs: Sequence[tuple[str, int]]
) -> dict[str, float]:
    """Mean of an arbitrary (metric_name, k) spec list over many eval pairs."""
    keys = [f"{name}@{k}" for name, k in specs]
    totals = dict.fromkeys(keys, 0.0)
    count = 0
    for ranked_items, true_item in pairs:
        result = evaluate_pair_multi(ranked_items, true_item, specs)
        for key in keys:
            totals[key] += result[key]
        count += 1
    if count == 0:
        raise ValueError("average_multi received no pairs to evaluate")
    return {key: value / count for key, value in totals.items()}
