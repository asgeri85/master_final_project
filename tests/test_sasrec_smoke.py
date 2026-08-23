"""Smoke test for src/models/sasrec.py: fit/predict shape correctness on tiny
synthetic data, plus SASRec-specific checks for the hidden_size/num_heads
repair strategy and the causal-mask/padding-mask invariant (a masking bug
here would silently leak future-token information into the training loss
without ever crashing)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.pop import PopModel  # noqa: E402
from src.models.sasrec import PAD_IDX, SasrecModel, _SASRecNet, repair_hidden_size  # noqa: E402


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
    model = SasrecModel(
        hidden_size=8, num_heads=2, num_blocks=1, dropout=0.1, max_epochs=2, batch_size=4, max_seq_len=5,
        checkpoint_dir=None,
    )
    model.fit(train)

    ranked = model.predict([10, 11])
    assert isinstance(ranked, list)
    assert set(ranked) == set(train["itemid"].unique().tolist())
    assert len(ranked) == len(set(ranked))


def test_empty_and_oov_session_falls_back_to_pop():
    train = _tiny_train()
    model = SasrecModel(
        hidden_size=8, num_heads=2, num_blocks=1, max_epochs=1, batch_size=4, max_seq_len=5, checkpoint_dir=None
    )
    model.fit(train)
    pop = PopModel()
    pop.fit(train)

    assert model.predict([]) == pop.predict([])
    assert model.predict([99999]) == pop.predict([99999])


def test_predict_many_matches_predict_one_by_one():
    # predict_many() re-implements predict()'s forward pass in a batched form
    # purely for eval speed -- this pins down that it must never drift from
    # calling predict() once per session, including the empty/OOV fallback path.
    train = _tiny_train()
    model = SasrecModel(
        hidden_size=8, num_heads=2, num_blocks=1, dropout=0.1, max_epochs=2, batch_size=4, max_seq_len=5,
        checkpoint_dir=None,
    )
    model.fit(train)

    sessions = [[10, 11], [10], [12, 13], [], [99999], [14, 15, 10], [11]]
    expected = [model.predict(s) for s in sessions]

    # eval_batch_size smaller than len(sessions) forces multiple chunks through
    # the batched path, exercising the same code as a real multi-thousand-pair eval.
    actual = model.predict_many(sessions, eval_batch_size=2)

    assert actual == expected


@pytest.mark.parametrize(
    "hidden_size, num_heads, low, high, expected",
    [
        (64, 4, 64, 256, 64),  # already divisible -> unchanged
        (65, 4, 64, 256, 68),  # round up within bounds
        (255, 4, 64, 255, 252),  # round-up (256) would exceed high (255) -> round down instead
        (254, 3, 64, 256, 255),  # round up to nearest multiple of 3
    ],
)
def test_repair_hidden_size(hidden_size, num_heads, low, high, expected):
    assert repair_hidden_size(hidden_size, num_heads, low, high) == expected


def test_repair_hidden_size_rejects_nonpositive_heads():
    with pytest.raises(ValueError):
        repair_hidden_size(64, 0, 64, 256)


def test_net_rejects_invalid_hidden_size():
    with pytest.raises(ValueError):
        _SASRecNet(n_items=10, hidden_size=65, num_heads=4, num_blocks=1, dropout=0.1, max_seq_len=5)


def test_causal_mask_blocks_future_and_ignores_padded_tail():
    # Two rows share an identical valid prefix [1, 2, 3] but differ in what
    # follows it: row 0 is padded with zeros, row 1 has different "future"
    # item ids in the same positions. If causal masking is correct, the
    # hidden states at the shared prefix positions must be identical --
    # position i must never be influenced by content at positions > i.
    torch.manual_seed(0)
    net = _SASRecNet(n_items=10, hidden_size=8, num_heads=2, num_blocks=2, dropout=0.0, max_seq_len=6)
    net.eval()

    x = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],
            [1, 2, 3, 4, 5, 6],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([3, 6], dtype=torch.long)

    with torch.no_grad():
        hidden = net.encode(x, lengths)

    prefix_len = 3
    assert torch.allclose(hidden[0, :prefix_len], hidden[1, :prefix_len], atol=1e-6)


def test_encode_output_shape():
    torch.manual_seed(0)
    for num_heads, num_blocks in [(1, 1), (2, 3), (4, 2)]:
        net = _SASRecNet(n_items=20, hidden_size=8, num_heads=num_heads, num_blocks=num_blocks, dropout=0.1, max_seq_len=5)
        x = torch.randint(0, 21, (3, 5))
        x[:, -1] = PAD_IDX  # ensure at least one padded position
        lengths = torch.tensor([4, 4, 4], dtype=torch.long)
        out = net.encode(x, lengths)
        assert out.shape == (3, 5, 8)
        assert not torch.isnan(out).any()
