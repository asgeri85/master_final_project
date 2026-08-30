"""Raw event loading and processed split writing for RetailRocket and LSApp."""

from __future__ import annotations

import json
import tarfile
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


def read_raw_events_lsapp(path: str | Path) -> pd.DataFrame:
    """Read the raw LSApp `lsapp.tsv`, which is actually a single-member POSIX
    tar archive (a 512-byte tar header followed by the real TSV content --
    detected Week 1, see docs/00-karar-notu.md SS4.2/SS16). Streamed via
    `tarfile` directly into `pandas`, without ever writing an extracted copy
    to disk.

    Raw columns are `user_id, session_id, timestamp, app_name, event_type`.
    Output is remapped to the same canonical schema `read_raw_events` produces
    for RetailRocket, so the rest of the pipeline (dedup/event-filter/
    sessionize/filter/split/coldstart) needs no LSApp-specific branching:
    `timestamp` (int64 epoch ms), `visitorid` (int64, from user_id), `event`
    (category, from event_type), `itemid` (int64, from app_name via a stable
    sorted-name mapping), `transactionid` (Int64, always NA -- LSApp has no
    transaction concept), `_raw_row_id` (original row order tiebreaker).

    The raw `session_id` column is dropped: LSApp's session_id is a single
    global counter, not a per-visitor gap-based session, so it is rebuilt
    from scratch by `sessionize.build_sessions` per A1 for consistency with
    RetailRocket.
    """
    with tarfile.open(path) as tf:
        members = tf.getmembers()
        if len(members) != 1:
            raise ValueError(f"expected a single-member tar archive at {path}, found {len(members)}")
        extracted = tf.extractfile(members[0])
        if extracted is None:
            raise ValueError(f"could not extract tar member from {path}")
        df = pd.read_csv(
            extracted,
            sep="\t",
            dtype={"user_id": "int64", "session_id": "int64", "app_name": "category", "event_type": "category"},
            parse_dates=["timestamp"],
        )

    df["_raw_row_id"] = df.index.to_numpy()
    sorted_app_names = sorted(df["app_name"].cat.categories.tolist())
    app_name_codes = df["app_name"].cat.set_categories(sorted_app_names).cat.codes.astype("int64")

    out = pd.DataFrame(
        {
            # .values.astype("datetime64[ms]") first normalizes to millisecond
            # resolution regardless of what pd.read_csv's date parser inferred
            # (pandas >=2 can infer us/s/ns resolution from string precision --
            # dividing a raw .astype("int64") by a fixed factor silently gives
            # the wrong unit if that guess changes across pandas versions).
            "timestamp": df["timestamp"].values.astype("datetime64[ms]").astype("int64"),
            "visitorid": df["user_id"],
            "event": df["event_type"],
            "itemid": app_name_codes,
            "transactionid": pd.array([pd.NA] * len(df), dtype="Int64"),
            "_raw_row_id": df["_raw_row_id"],
        }
    )
    out.attrs["item_id_to_app_name"] = dict(enumerate(sorted_app_names))
    return out


def write_item_mapping(item_id_to_name: dict[int, str], path: str | Path) -> None:
    """Persist an itemid -> original name mapping (e.g. LSApp's app_name) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({str(k): v for k, v in sorted(item_id_to_name.items())}, indent=2, ensure_ascii=False))


def write_split(df: pd.DataFrame, path: str | Path, compression: str = "snappy") -> None:
    """Write a processed split to parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)
