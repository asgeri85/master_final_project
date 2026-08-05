"""Item-frequency and session-length filtering (A2) for RetailRocket."""

from __future__ import annotations

import pandas as pd

from src.data.sessionize import recompute_session_stats


def filter_by_item_frequency(df: pd.DataFrame, min_freq: int) -> pd.DataFrame:
    """Drop rows whose itemid occurs fewer than min_freq times in the whole df."""
    counts = df["itemid"].value_counts()
    keep_items = counts[counts >= min_freq].index
    return df[df["itemid"].isin(keep_items)].copy()


def filter_by_session_length(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """Drop rows belonging to sessions shorter than min_len events.

    Requires `session_length` to already reflect the current row set
    (call sessionize.recompute_session_stats first if rows were removed).
    """
    return df[df["session_length"] >= min_len].copy()


def iteratively_filter(
    df: pd.DataFrame, min_item_freq: int, min_session_len: int
) -> tuple[pd.DataFrame, int]:
    """Apply A2's item-frequency and session-length filters to a fixed point.

    A single pass is not enough: dropping infrequent items can shrink a
    session below min_session_len, and dropping short sessions removes item
    occurrences, which can push an item's surviving count back below
    min_item_freq. Iterate both filters (recomputing session stats between
    them) until a pass removes no further rows, so both thresholds hold
    simultaneously in the final output. session_id groupings themselves are
    never recomputed here — only which rows survive within them changes.
    """
    iterations = 0
    while True:
        iterations += 1
        before = len(df)

        df = filter_by_item_frequency(df, min_item_freq)
        df = recompute_session_stats(df)
        df = filter_by_session_length(df, min_session_len)
        df = recompute_session_stats(df)

        if len(df) == before:
            return df, iterations
