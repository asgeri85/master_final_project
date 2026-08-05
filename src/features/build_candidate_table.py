"""Builds the candidate table used to train/evaluate XGBoost-raw and XGBoost-rich.

Item-KNN generates the top-N candidate pool per session prefix (locked design,
karar-notu Katman-2: "Item-KNN -> top-100 aday uret -> XGBoost/RF ozniteligiyle
yeniden sirala"); this module enriches those candidates with the 12 A5
features and, for training data, attaches a binary relevance label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.harness import iter_session_prefixes
from src.features.feature_builder import FeatureBuilder
from src.models.item_knn import ItemKnnModel


def build_candidate_table(
    item_knn: ItemKnnModel,
    feature_builder: FeatureBuilder,
    split_df: pd.DataFrame,
    n_candidates: int = 100,
    include_label: bool = True,
    negatives_per_positive: int | None = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """One row per (session-prefix, candidate item), with the 12 A5 features.

    negatives_per_positive: if set (training only), each prefix's candidates
    are subsampled to at most 1 positive (if the true item is among them) +
    this many random negatives, keeping the training set tractable -- this is
    standard negative subsampling for training a pointwise ranker and does
    NOT violate the project's "no negative sampling" rule, which governs
    evaluation methodology, not training-set construction. Leave None for
    evaluation, so every candidate in the fixed top-N pool is scored.
    """
    rng = np.random.default_rng(random_seed)
    frames = []
    query_id = 0
    for ctx in iter_session_prefixes(split_df):
        candidates = item_knn.top_candidates(ctx["items"], n=n_candidates)
        candidate_items = [item for item, _ in candidates]
        knn_scores = {item: score for item, score in candidates}

        features = feature_builder.build_candidate_features(
            ctx["items"], ctx["timestamps"], candidate_items, knn_scores
        )
        features["query_id"] = query_id
        features["session_id"] = ctx["session_id"]
        features["true_item"] = ctx["true_item"]
        if include_label:
            features["label"] = (features["itemid"] == ctx["true_item"]).astype(np.int8)
            if negatives_per_positive is not None:
                features = _subsample_negatives(features, negatives_per_positive, rng)

        frames.append(features)
        query_id += 1

    return pd.concat(frames, ignore_index=True)


def _subsample_negatives(
    features: pd.DataFrame, negatives_per_positive: int, rng: np.random.Generator
) -> pd.DataFrame:
    positives = features[features["label"] == 1]
    negatives = features[features["label"] == 0]
    if len(negatives) > negatives_per_positive:
        keep_idx = rng.choice(negatives.index.to_numpy(), size=negatives_per_positive, replace=False)
        negatives = negatives.loc[keep_idx]
    return pd.concat([positives, negatives], ignore_index=True)
