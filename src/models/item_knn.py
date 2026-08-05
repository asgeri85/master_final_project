"""Item-KNN: item-item co-occurrence similarity baseline.

Similarity is built from session-item co-occurrence (cosine-normalized),
truncated to the top-`neighborhood_size` neighbors per item to keep memory
bounded regardless of catalog size (a dense V x V matrix would be
infeasible for a ~50k-235k item catalog).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.pop import PopModel


class ItemKnnModel:
    def __init__(self, neighborhood_size: int = 100) -> None:
        self.neighborhood_size = neighborhood_size
        self._pop = PopModel()
        self._pop_array: np.ndarray = np.array([], dtype=np.int64)
        self._item_to_idx: dict[int, int] = {}
        self._idx_to_item: np.ndarray = np.array([], dtype=np.int64)
        self._similarity: sparse.csr_matrix | None = None

    def fit(self, train: pd.DataFrame) -> None:
        self._pop.fit(train)
        self._pop_array = np.array(self._pop.predict(()), dtype=np.int64)

        items = np.sort(train["itemid"].unique())
        self._idx_to_item = items
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

        support = np.asarray(incidence.sum(axis=0)).ravel()
        support[support == 0] = 1.0  # items with 0 support have no nonzero rows to rescale
        inv_sqrt_support = sparse.diags(1.0 / np.sqrt(support))
        cooccurrence = (inv_sqrt_support @ cooccurrence @ inv_sqrt_support).tocsr()

        self._similarity = _top_k_per_row(cooccurrence, self.neighborhood_size)

    def _score_session(self, session: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Nonzero-scored items for a session, sorted (score desc, item id asc)."""
        assert self._similarity is not None, "call fit() before predict()"
        counts = Counter(session)
        known = {item: c for item, c in counts.items() if item in self._item_to_idx}

        if not known:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        scores = np.zeros(len(self._idx_to_item), dtype=np.float64)
        for item, count in known.items():
            row = self._similarity.getrow(self._item_to_idx[item])
            scores[row.indices] += count * row.data

        nonzero = np.flatnonzero(scores > 0)
        order = np.lexsort((self._idx_to_item[nonzero], -scores[nonzero]))
        idx_sorted = nonzero[order]
        return self._idx_to_item[idx_sorted], scores[idx_sorted]

    def predict(self, session: Sequence[int]) -> list[int]:
        scored_items, _ = self._score_session(session)
        if len(scored_items) == 0:
            return self._pop.predict(session)
        rest = self._pop_array[~np.isin(self._pop_array, scored_items)]
        return scored_items.tolist() + rest.tolist()

    def top_candidates(self, session: Sequence[int], n: int = 100) -> list[tuple[int, float]]:
        """Top-n (item, score) candidates for the Katman-2 re-ranking pipeline.

        Padded with global-popularity fallback items (score 0.0) if fewer
        than n items have a nonzero Item-KNN score.
        """
        scored_items, scores = self._score_session(session)
        if len(scored_items) >= n:
            return list(zip(scored_items[:n].tolist(), scores[:n].tolist()))

        result = list(zip(scored_items.tolist(), scores.tolist()))
        fallback = self._pop_array[~np.isin(self._pop_array, scored_items)]
        needed = n - len(result)
        result.extend((item, 0.0) for item in fallback[:needed].tolist())
        return result


def _top_k_per_row(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep only the top-k largest values per row of a sparse CSR matrix."""
    data, indices, indptr = matrix.data, matrix.indices, matrix.indptr
    new_data, new_indices, new_indptr = [], [], [0]
    for row in range(matrix.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_data, row_idx = data[start:end], indices[start:end]
        if len(row_data) > k:
            top_k = np.argpartition(row_data, -k)[-k:]
            row_data, row_idx = row_data[top_k], row_idx[top_k]
        new_data.append(row_data)
        new_indices.append(row_idx)
        new_indptr.append(new_indptr[-1] + len(row_data))
    return sparse.csr_matrix(
        (np.concatenate(new_data), np.concatenate(new_indices), new_indptr),
        shape=matrix.shape,
    )
