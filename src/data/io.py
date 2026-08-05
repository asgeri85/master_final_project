"""Raw event loading and processed split writing for RetailRocket."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_raw_events(path: str | Path) -> pd.DataFrame:
    """Read the raw RetailRocket events.csv with explicit dtypes.

    Adds `_raw_row_id` (original CSV row order) to serve as a deterministic
    sort tiebreaker for the ~2,600 (visitorid, timestamp) pairs that tie.
    """
    df = pd.read_csv(
        path,
        dtype={
            "timestamp": "int64",
            "visitorid": "int64",
            "event": "category",
            "itemid": "int64",
            "transactionid": "Int64",
        },
    )
    df["_raw_row_id"] = df.index.to_numpy()
    return df


def write_split(df: pd.DataFrame, path: str | Path, compression: str = "snappy") -> None:
    """Write a processed split to parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)
