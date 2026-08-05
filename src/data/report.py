"""Summary reporting for the RetailRocket preprocessing pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _split_stats(df: pd.DataFrame, name: str, file_path: str | None) -> dict:
    stats: dict = {
        "n_rows": int(len(df)),
        "n_sessions": int(df["session_id"].nunique()),
        "n_unique_items": int(df["itemid"].nunique()),
        "n_unique_visitors": int(df["visitorid"].nunique()),
    }
    if name != "train" and "is_cold_start" in df.columns:
        n_cold = int(df["is_cold_start"].sum())
        stats["n_cold_start_rows"] = n_cold
        stats["pct_cold_start_rows"] = round(100 * n_cold / len(df), 2) if len(df) else 0.0
    if file_path and Path(file_path).exists():
        size_bytes = Path(file_path).stat().st_size
        stats["file_path"] = str(file_path)
        stats["file_size_bytes"] = size_bytes
        stats["file_size_mb"] = round(size_bytes / 1e6, 3)
    return stats


def build_report(
    funnel: dict,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    file_paths: dict[str, str],
) -> dict:
    """Build a JSON-serializable summary of the pipeline run."""
    splits = {
        "train": _split_stats(train, "train", file_paths.get("train")),
        "val": _split_stats(val, "val", file_paths.get("val")),
        "test": _split_stats(test, "test", file_paths.get("test")),
    }
    example_rows = {
        "train": train.head(5).to_dict(orient="records"),
        "val": val.head(5).to_dict(orient="records"),
        "test": test.head(5).to_dict(orient="records"),
    }
    return {"filtering_funnel": funnel, "splits": splits, "example_rows": example_rows}


def print_report(report: dict) -> None:
    """Print the report in a readable form to stdout."""
    print("=== Filtering funnel ===")
    for key, value in report["filtering_funnel"].items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print("\n=== Splits ===")
    for name, stats in report["splits"].items():
        print(f"[{name}]")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    print("\n=== Example rows ===")
    for name, rows in report["example_rows"].items():
        print(f"[{name}]")
        for row in rows:
            print(f"  {row}")
