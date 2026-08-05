"""Common model interface shared by every model in src/models/ (CLAUDE.md rule)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import pandas as pd


class RecommenderModel(Protocol):
    def fit(self, train: pd.DataFrame) -> None: ...

    def predict(self, session: Sequence[int]) -> list[int]: ...
