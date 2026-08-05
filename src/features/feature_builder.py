"""Builds the 12 engineered features (karar-notu SS12a A5) for candidate re-ranking.

Feature groups (A5) and how the count of 4 named groups becomes exactly 12
columns: hour-of-day and day-of-week are cyclically encoded as sin/cos pairs
(2 dims each) rather than raw integers -- standard practice for periodic
features, confirmed with the user -- bringing Temporal from 3 named concepts
to 5 columns, and the total from 10 to 12.

Recency ("last 7 days") is a fixed snapshot relative to train's own time
window, computed once at fit() time -- mirrors how POP/S-POP/Item-KNN freeze
their statistics at fit() rather than recomputing per query.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

_DAY_MS = 24 * 60 * 60 * 1000

FEATURE_COLUMNS = [
    "item_global_view_count",
    "item_last_7d_view_count",
    "item_knn_score",
    "shared_session_count_with_last_item",
    "elapsed_time_in_session_ms",
    "hour_of_day_sin",
    "hour_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "session_length_so_far",
    "item_repeat_count_in_session",
    "session_unique_item_count",
]

RAW_FEATURE_COLUMNS = ["item_knn_score"]


class FeatureBuilder:
    """Precomputes A5 feature statistics from train, then builds per-candidate rows."""

    def __init__(self) -> None:
        self._global_view_count: dict[int, int] = {}
        self._last_7d_view_count: dict[int, int] = {}
        self._item_to_idx: dict[int, int] = {}
        self._raw_cooccurrence: sparse.csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self._global_view_count = train["itemid"].value_counts().to_dict()

        max_ts = train["timestamp"].max()
        recent = train[train["timestamp"] >= max_ts - 7 * _DAY_MS]
        self._last_7d_view_count = recent["itemid"].value_counts().to_dict()

        items = np.sort(train["itemid"].unique())
        self._item_to_idx = {int(item): idx for idx, item in enumerate(items)}
        n_items = len(items)

        pairs = train[["session_id", "itemid"]].drop_duplicates()
        item_idx = pairs["itemid"].map(self._item_to_idx).to_numpy()
        _, session_row = np.unique(pairs["session_id"].to_numpy(), return_inverse=True)
        n_sessions = session_row.max() + 1 if len(session_row) else 0

        incidence = sparse.csr_matrix(
            (np.ones(len(pairs), dtype=np.float32), (session_row, item_idx)),
            shape=(n_sessions, n_items),
        )
        cooccurrence = (incidence.T @ incidence).tocsr()
        cooccurrence.setdiag(0)
        cooccurrence.eliminate_zeros()
        self._raw_cooccurrence = cooccurrence

    def build_candidate_features(
        self,
        context_items: list[int],
        context_timestamps: list[int],
        candidate_items: list[int],
        knn_scores: dict[int, float],
    ) -> pd.DataFrame:
        """One row per candidate item, with all 12 A5 feature columns."""
        assert self._raw_cooccurrence is not None, "call fit() before build_candidate_features()"

        candidates = np.asarray(candidate_items, dtype=np.int64)
        n = len(candidates)

        last_item = context_items[-1]
        last_ts = context_timestamps[-1]
        elapsed_ms = context_timestamps[-1] - context_timestamps[0]

        dt = pd.Timestamp(last_ts, unit="ms", tz="UTC")
        hour = dt.hour + dt.minute / 60.0
        dow = dt.dayofweek  # Monday=0

        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * dow / 7)
        dow_cos = math.cos(2 * math.pi * dow / 7)

        session_length_so_far = len(context_items)
        session_unique_item_count = len(set(context_items))
        item_counts_in_context = Counter(context_items)

        global_view_count = np.fromiter(
            (self._global_view_count.get(int(i), 0) for i in candidates), dtype=np.float64, count=n
        )
        last_7d_view_count = np.fromiter(
            (self._last_7d_view_count.get(int(i), 0) for i in candidates), dtype=np.float64, count=n
        )
        item_knn_score = np.fromiter(
            (knn_scores.get(int(i), 0.0) for i in candidates), dtype=np.float64, count=n
        )
        repeat_count = np.fromiter(
            (item_counts_in_context.get(int(i), 0) for i in candidates), dtype=np.float64, count=n
        )

        last_item_idx = self._item_to_idx.get(last_item)
        if last_item_idx is None:
            shared_counts = np.zeros(n, dtype=np.float64)
        else:
            dense_row = np.asarray(self._raw_cooccurrence.getrow(last_item_idx).todense()).ravel()
            cand_idx = np.fromiter(
                (self._item_to_idx.get(int(i), -1) for i in candidates), dtype=np.int64, count=n
            )
            shared_counts = np.where(cand_idx >= 0, dense_row[np.clip(cand_idx, 0, None)], 0.0)

        return pd.DataFrame(
            {
                "itemid": candidates,
                "item_global_view_count": global_view_count,
                "item_last_7d_view_count": last_7d_view_count,
                "item_knn_score": item_knn_score,
                "shared_session_count_with_last_item": shared_counts,
                "elapsed_time_in_session_ms": np.full(n, elapsed_ms, dtype=np.float64),
                "hour_of_day_sin": np.full(n, hour_sin, dtype=np.float64),
                "hour_of_day_cos": np.full(n, hour_cos, dtype=np.float64),
                "day_of_week_sin": np.full(n, dow_sin, dtype=np.float64),
                "day_of_week_cos": np.full(n, dow_cos, dtype=np.float64),
                "session_length_so_far": np.full(n, session_length_so_far, dtype=np.float64),
                "item_repeat_count_in_session": repeat_count,
                "session_unique_item_count": np.full(n, session_unique_item_count, dtype=np.float64),
            }
        )
