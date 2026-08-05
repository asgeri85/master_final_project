"""Smoke test for src/models/gru4rec.py: fit/predict shape correctness on tiny
synthetic data. Not mandated by CLAUDE.md (only eval/metrics.py is), but this
is the repo's first PyTorch model -- a minimal correctness check is cheap
insurance against padding/index bugs."""

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.gru4rec import Gru4RecModel  # noqa: E402
from src.models.pop import PopModel  # noqa: E402


def _tiny_train() -> pd.DataFrame:
    # 4 sessions, 6 distinct items, session_event_index starting at 0
    rows = [
        (0, 1, 10, 100, 0, 3),
        (1, 1, 11, 100, 1, 3),
        (2, 1, 12, 100, 2, 3),
        (3, 2, 10, 101, 0, 2),
        (4, 2, 13, 101, 1, 2),
        (5, 3, 14, 102, 0, 4),
        (6, 3, 15, 102, 1, 4),
        (7, 3, 10, 102, 2, 4),
        (8, 3, 11, 102, 3, 4),
        (9, 4, 12, 103, 0, 2),
        (10, 4, 13, 103, 1, 2),
    ]
    df = pd.DataFrame(
        rows,
        columns=["timestamp", "visitorid", "itemid", "session_id", "session_event_index", "session_length"],
    )
    df["split"] = "train"
    df["is_cold_start"] = False
    return df


def test_fit_predict_full_catalog_coverage():
    train = _tiny_train()
    model = Gru4RecModel(
        hidden_size=8, num_layers=1, dropout=0.1, max_epochs=2, batch_size=4, max_seq_len=5, checkpoint_dir=None
    )
    model.fit(train)

    ranked = model.predict([10, 11])
    assert isinstance(ranked, list)
    assert set(ranked) == set(train["itemid"].unique().tolist())
    assert len(ranked) == len(set(ranked))


def test_empty_and_oov_session_falls_back_to_pop():
    train = _tiny_train()
    model = Gru4RecModel(hidden_size=8, num_layers=1, max_epochs=1, batch_size=4, max_seq_len=5, checkpoint_dir=None)
    model.fit(train)
    pop = PopModel()
    pop.fit(train)

    assert model.predict([]) == pop.predict([])
    assert model.predict([99999]) == pop.predict([99999])
