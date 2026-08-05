"""XGBoost candidate re-ranker: the two feature-set variants for the H2 ablation.

XGBoost-raw uses only the Item-KNN candidate score (1 feature).
XGBoost-rich uses all 12 A5 features.
Both re-rank the SAME Item-KNN top-N candidate pool (karar-notu Katman-2
design), so H2's comparison isolates the effect of feature engineering alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.build_candidate_table import build_candidate_table
from src.features.feature_builder import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, FeatureBuilder
from src.models.item_knn import ItemKnnModel


class XGBoostRankerModel:
    """fit(train)/predict(session) per the project's common model interface.

    predict(session) only receives item ids (no timestamps), so temporal
    features fall back to neutral placeholders there; use
    predict_from_context(items, timestamps) for full-fidelity scoring during
    offline training/evaluation, where real timestamps are available.
    """

    def __init__(
        self,
        item_knn: ItemKnnModel,
        feature_builder: FeatureBuilder,
        feature_set: str,
        xgb_params: dict,
        n_candidates: int = 100,
        negatives_per_positive: int | None = 9,
        random_seed: int = 42,
    ) -> None:
        if feature_set not in ("raw", "rich"):
            raise ValueError(f"feature_set must be 'raw' or 'rich', got {feature_set!r}")
        self.item_knn = item_knn
        self.feature_builder = feature_builder
        self.feature_set = feature_set
        self.xgb_params = dict(xgb_params)
        self.n_candidates = n_candidates
        self.negatives_per_positive = negatives_per_positive
        self.random_seed = random_seed
        self.feature_columns = RAW_FEATURE_COLUMNS if feature_set == "raw" else FEATURE_COLUMNS
        self._booster: xgb.Booster | None = None

    def fit(self, train: pd.DataFrame) -> None:
        table = build_candidate_table(
            self.item_knn,
            self.feature_builder,
            train,
            n_candidates=self.n_candidates,
            include_label=True,
            negatives_per_positive=self.negatives_per_positive,
            random_seed=self.random_seed,
        )
        self.fit_from_table(table)

    def fit_from_table(self, table: pd.DataFrame) -> None:
        """Fit directly from an already-built candidate table (avoids rebuilding
        the table across repeated calls, e.g. Optuna trials sharing one table)."""
        X = table[self.feature_columns].to_numpy(dtype=np.float32)
        y = table["label"].to_numpy(dtype=np.float32)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_columns)
        self._booster = xgb.train(
            self._native_params(), dtrain, num_boost_round=self.xgb_params["n_estimators"]
        )

    def _native_params(self) -> dict:
        return {
            "max_depth": self.xgb_params["max_depth"],
            "learning_rate": self.xgb_params["learning_rate"],
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": self.random_seed,
        }

    def predict(self, session: Sequence[int]) -> list[int]:
        placeholder_timestamps = list(range(len(session)))
        return self.predict_from_context(list(session), placeholder_timestamps)

    def predict_from_context(self, items: list[int], timestamps: list[int]) -> list[int]:
        """Full-fidelity re-ranking using real context timestamps."""
        assert self._booster is not None, "call fit() before predict"
        candidates = self.item_knn.top_candidates(items, n=self.n_candidates)
        candidate_items = [item for item, _ in candidates]
        knn_scores = {item: score for item, score in candidates}

        features = self.feature_builder.build_candidate_features(
            items, timestamps, candidate_items, knn_scores
        )
        X = features[self.feature_columns].to_numpy(dtype=np.float32)
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_columns)
        scores = self._booster.predict(dmatrix)

        candidate_arr = np.asarray(candidate_items, dtype=np.int64)
        order = np.lexsort((candidate_arr, -scores))
        return candidate_arr[order].tolist()
