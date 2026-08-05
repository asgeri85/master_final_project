"""Time-based train/val/test split (A3) for RetailRocket. Random split is forbidden."""

from __future__ import annotations

import pandas as pd

_DAY_MS = 24 * 60 * 60 * 1000


def time_based_split(
    df: pd.DataFrame, val_days: int, test_days: int, bucket_by: str = "session_start"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split whole sessions by their anchor time: last test_days -> test,
    the preceding val_days -> val, everything before that -> train.

    bucket_by="session_start" (recommended) anchors on a session's first
    event time; "session_end" anchors on its last event time.
    """
    if bucket_by not in ("session_start", "session_end"):
        raise ValueError(f"bucket_by must be 'session_start' or 'session_end', got {bucket_by!r}")

    max_ts = df["timestamp"].max()
    test_start = max_ts - test_days * _DAY_MS
    val_start = test_start - val_days * _DAY_MS

    agg = "min" if bucket_by == "session_start" else "max"
    anchor = df.groupby("session_id")["timestamp"].transform(agg)

    split_col = pd.Series("train", index=df.index, dtype="object")
    split_col[(anchor >= val_start) & (anchor < test_start)] = "val"
    split_col[anchor >= test_start] = "test"

    out = df.copy()
    out["split"] = split_col.astype("category")

    train = out[out["split"] == "train"].copy()
    val = out[out["split"] == "val"].copy()
    test = out[out["split"] == "test"].copy()
    return train, val, test
