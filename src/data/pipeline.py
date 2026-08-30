"""Orchestrates the A1-A4 preprocessing pipeline for RetailRocket and LSApp.

CLI usage:
  python -m src.data.pipeline --config configs/retailrocket_preprocess.yaml
  python -m src.data.pipeline --config configs/lsapp_preprocess.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.coldstart import add_cold_start_flag
from src.data.config import PreprocessConfig, load_config
from src.data.filter import iteratively_filter
from src.data.io import read_raw_events, read_raw_events_lsapp, write_item_mapping, write_split
from src.data.report import build_report, print_report
from src.data.sessionize import build_sessions
from src.data.split import time_based_split


def run_pipeline(config: PreprocessConfig) -> dict:
    """Run the full preprocessing pipeline and return the summary report."""
    funnel: dict[str, int] = {}

    if config.dataset.name == "lsapp":
        df = read_raw_events_lsapp(config.dataset.raw_events_path)
        item_id_to_app_name = df.attrs.get("item_id_to_app_name")
    else:
        df = read_raw_events(config.dataset.raw_events_path)
        item_id_to_app_name = None
    funnel["raw_rows"] = len(df)

    df = df.drop_duplicates(
        subset=["timestamp", "visitorid", "event", "itemid", "transactionid"]
    )
    funnel["rows_after_dedup"] = len(df)

    df = df[df["event"] == config.session.event_filter].copy()
    df = df.drop(columns=["event", "transactionid"])
    funnel["rows_after_event_filter"] = len(df)

    df = df.sort_values(["visitorid", "timestamp", "_raw_row_id"]).reset_index(drop=True)
    df = df.drop(columns=["_raw_row_id"])

    df = build_sessions(df, config.session.gap_minutes)
    funnel["sessions_before_item_filter"] = int(df["session_id"].nunique())

    df, n_iterations = iteratively_filter(
        df, config.filtering.min_item_frequency, config.filtering.min_session_length
    )
    funnel["filter_iterations_to_converge"] = n_iterations
    funnel["rows_after_iterative_filtering"] = len(df)
    funnel["sessions_final"] = int(df["session_id"].nunique())

    train, val, test = time_based_split(
        df, config.split.val_days, config.split.test_days, config.split.bucket_by
    )
    train, val, test = add_cold_start_flag(train, val, test)

    processed_dir = Path(config.dataset.processed_dir)
    file_paths = {
        "train": str(processed_dir / "train.parquet"),
        "val": str(processed_dir / "val.parquet"),
        "test": str(processed_dir / "test.parquet"),
    }
    write_split(train, file_paths["train"], config.io.parquet_compression)
    write_split(val, file_paths["val"], config.io.parquet_compression)
    write_split(test, file_paths["test"], config.io.parquet_compression)

    if item_id_to_app_name is not None:
        mapping_path = processed_dir / "item_id_to_app_name.json"
        write_item_mapping(item_id_to_app_name, mapping_path)
        file_paths["item_id_to_app_name"] = str(mapping_path)

    report = build_report(funnel, train, val, test, file_paths)
    if config.io.report_path:
        Path(config.io.report_path).write_text(
            json.dumps(report, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="RetailRocket/LSApp preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to preprocessing YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    report = run_pipeline(config)
    print_report(report)


if __name__ == "__main__":
    main()
