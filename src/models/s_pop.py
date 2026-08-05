"""S-POP: session-local popularity, falling back to global popularity."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models.pop import PopModel


class SPopModel:
    """Ranks items seen in the current session by in-session frequency, then
    appends the rest of the catalog in global-popularity order.

    Tie-break: in-session frequency desc -> global frequency desc -> item id asc.
    """

    def __init__(self) -> None:
        self._pop = PopModel()
        self._global_freq: dict[int, int] = {}
        self._pop_array: np.ndarray = np.array([], dtype=np.int64)

    def fit(self, train: pd.DataFrame) -> None:
        self._pop.fit(train)
        self._global_freq = train["itemid"].value_counts().to_dict()
        self._pop_array = np.array(self._pop.predict(()), dtype=np.int64)

    def predict(self, session: Sequence[int]) -> list[int]:
        session_counts = Counter(session)
        session_items = sorted(
            session_counts,
            key=lambda item: (-session_counts[item], -self._global_freq.get(item, 0), item),
        )
        # numpy set-difference: predict() is called once per eval pair (potentially
        # 10^4-10^5 times), so filtering the full catalog must avoid a per-call
        # Python-level scan over every item.
        session_arr = np.asarray(session_items, dtype=np.int64)
        rest = self._pop_array[~np.isin(self._pop_array, session_arr)]
        return session_items + rest.tolist()
