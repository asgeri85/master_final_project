"""Evaluation harness: slide over session prefixes and score a model.

For a session with items [i1, ..., iL] (L >= 2), yields (L-1) (context, true_next)
pairs: ([i1], i2), ([i1, i2], i3), ..., ([i1, ..., i_{L-1}], iL).
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from src.eval.metrics import average_at_k


def iter_session_prefixes(split_df: pd.DataFrame) -> Iterator[dict]:
    """Yield one dict per (context prefix, true next item) within each session:
    {session_id, items, timestamps, true_item}. `items`/`timestamps` are
    parallel lists for the context so far (needed for temporal features that
    `iter_eval_pairs`'s plain item-id lists can't carry).

    Sessions with fewer than 2 surviving items are skipped (no next item to
    predict) — this also guards against any session left truncated to length
    1 by an upstream time-based split boundary.
    """
    for session_id, group in split_df.sort_values("timestamp").groupby("session_id", sort=False):
        items = group["itemid"].tolist()
        timestamps = group["timestamp"].tolist()
        if len(items) < 2:
            continue
        for t in range(1, len(items)):
            yield {
                "session_id": session_id,
                "items": items[:t],
                "timestamps": timestamps[:t],
                "true_item": items[t],
            }


def iter_eval_pairs(split_df: pd.DataFrame) -> Iterator[tuple[list[int], int]]:
    """Yield (context_items, true_next_item) pairs for every session in split_df."""
    for ctx in iter_session_prefixes(split_df):
        yield ctx["items"], ctx["true_item"]


def evaluate_model(model, split_df: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Run model.predict() over every eval pair in split_df; return averaged HR/NDCG/MRR@K."""

    def pairs():
        for context, true_item in iter_eval_pairs(split_df):
            yield model.predict(context), true_item

    return average_at_k(pairs(), k)


def count_eval_pairs(split_df: pd.DataFrame) -> int:
    """Number of (context, true_next_item) pairs iter_eval_pairs would yield."""
    lengths = split_df.groupby("session_id", sort=False).size()
    return int((lengths[lengths >= 2] - 1).sum())
