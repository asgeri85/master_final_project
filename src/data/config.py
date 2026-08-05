"""Typed config loading for the RetailRocket preprocessing pipeline.

All numeric thresholds (session gap, filtering, split windows) live in YAML,
never as literals in the pipeline modules (CLAUDE.md rule 5).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_events_path: str
    processed_dir: str


@dataclass(frozen=True)
class SessionConfig:
    event_filter: str
    gap_minutes: int


@dataclass(frozen=True)
class FilteringConfig:
    min_session_length: int
    min_item_frequency: int


@dataclass(frozen=True)
class SplitConfig:
    test_days: int
    val_days: int
    bucket_by: str


@dataclass(frozen=True)
class ColdStartConfig:
    enabled: bool
    reference_split: str


@dataclass(frozen=True)
class IOConfig:
    parquet_compression: str
    report_path: str


@dataclass(frozen=True)
class PreprocessConfig:
    dataset: DatasetConfig
    session: SessionConfig
    filtering: FilteringConfig
    split: SplitConfig
    cold_start: ColdStartConfig
    io: IOConfig


def load_config(path: str | Path) -> PreprocessConfig:
    """Load and validate a RetailRocket preprocessing config from a YAML file."""
    raw = yaml.safe_load(Path(path).read_text())
    return PreprocessConfig(
        dataset=DatasetConfig(**raw["dataset"]),
        session=SessionConfig(**raw["session"]),
        filtering=FilteringConfig(**raw["filtering"]),
        split=SplitConfig(**raw["split"]),
        cold_start=ColdStartConfig(**raw["cold_start"]),
        io=IOConfig(**raw["io"]),
    )
