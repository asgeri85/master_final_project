"""POP: global popularity baseline. Ignores session content."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


class PopModel:
    """Ranks items by global interaction frequency in train, ties broken by ascending item id."""

    def __init__(self) -> None:
        self._ranked_items: list[int] = []

    def fit(self, train: pd.DataFrame) -> None:
        counts = train["itemid"].value_counts()
        ordered = sorted(counts.index.tolist(), key=lambda item: (-counts[item], item))
        self._ranked_items = ordered

    def predict(self, session: Sequence[int]) -> list[int]:
        return self._ranked_items
