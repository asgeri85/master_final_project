"""Session construction (A1) for RetailRocket."""

from __future__ import annotations

import pandas as pd


def build_sessions(df: pd.DataFrame, gap_minutes: int) -> pd.DataFrame:
    """Assign session_id to a stream already sorted by (visitorid, timestamp, _raw_row_id).

    A1: a new session starts when the visitor changes, or when the gap since
    the previous event exceeds `gap_minutes`.
    """
    gap_ms = gap_minutes * 60 * 1000
    prev_visitor = df["visitorid"].shift(1)
    prev_ts = df["timestamp"].shift(1)
    new_session = (df["visitorid"] != prev_visitor) | (df["timestamp"] - prev_ts > gap_ms)

    out = df.copy()
    out["session_id"] = new_session.cumsum().astype("int64")
    return recompute_session_stats(out)


def recompute_session_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute session_event_index (0-based position) and session_length per session_id.

    Must be re-run after any row removal, since filtering can leave a
    session's surviving rows with non-contiguous original positions.
    Assumes df rows are already in time order within each session_id.
    """
    out = df.copy()
    grouped = out.groupby("session_id", sort=False)
    out["session_event_index"] = grouped.cumcount().astype("int32")
    out["session_length"] = grouped["session_id"].transform("size").astype("int32")
    return out
