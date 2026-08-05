"""Cold-start flagging (A4) for RetailRocket. Cold-start items are kept, not removed."""

from __future__ import annotations

import pandas as pd


def add_cold_start_flag(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flag itemids in val/test that never appeared in train as is_cold_start.

    Cold-start rows are NOT removed (A4) — they're flagged so a separate
    cold-start subset table can be computed later, e.g. `df[df.is_cold_start]`.
    """
    train_items = set(train["itemid"].unique().tolist())

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["is_cold_start"] = False
    val["is_cold_start"] = ~val["itemid"].isin(train_items)
    test["is_cold_start"] = ~test["itemid"].isin(train_items)
    return train, val, test
