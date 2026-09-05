"""Correctness test for src/ondevice/export_onnx.py's static-shape export
wrappers: since ONNX export can't be trivially compared against a "ground
truth" without running the ONNX runtime, the real risk is the wrapper's own
padding/gather logic drifting from _GRU4RecNet/_LSTMNet/_SASRecNet.encode()'s
pack_padded_sequence path. This test fits tiny models (same pattern as
test_gru4rec_smoke.py/test_sasrec_smoke.py) and checks the wrapper's full-padded
forward + explicit last_index gather produces identical logits to encode()+
project() at the true last position -- the invariant the whole conversion
pipeline depends on, checkable with no ONNX/TFLite/checkpoint involved."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.models.gru4rec import Gru4RecModel  # noqa: E402
from src.models.lstm import LstmModel  # noqa: E402
from src.models.sasrec import SasrecModel  # noqa: E402
from src.ondevice.export_onnx import build_export_wrapper, prepare_input  # noqa: E402


def _tiny_train() -> pd.DataFrame:
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


def _reference_logits(net, item_to_idx, known_items, max_seq_len):
    idx_seq = torch.tensor([[item_to_idx[i] for i in known_items]], dtype=torch.long)
    length = torch.tensor([len(known_items)], dtype=torch.long)
    with torch.no_grad():
        hidden = net.encode(idx_seq, length)
        last_hidden = hidden[0, length[0] - 1].unsqueeze(0)
        return net.project(last_hidden).squeeze(0)


@pytest.mark.parametrize(
    "model_name,build_model",
    [
        ("gru4rec", lambda: Gru4RecModel(hidden_size=8, num_layers=1, max_epochs=1, batch_size=4, max_seq_len=5, checkpoint_dir=None)),
        ("lstm", lambda: LstmModel(hidden_size=8, num_layers=1, max_epochs=1, batch_size=4, max_seq_len=5, checkpoint_dir=None)),
        (
            "sasrec",
            lambda: SasrecModel(
                hidden_size=8, num_heads=2, num_blocks=1, max_epochs=1, batch_size=4, max_seq_len=5, checkpoint_dir=None
            ),
        ),
    ],
)
def test_export_wrapper_matches_encode_project(model_name, build_model):
    train = _tiny_train()
    model = build_model()
    model.fit(train)
    net = model._net
    net.eval()
    wrapper = build_export_wrapper(model_name, net)

    for known_items in ([10, 11], [10, 11, 12], [14]):
        known_idx = [model.item_to_idx[i] for i in known_items]
        item_ids, last_index = prepare_input(known_idx, max_seq_len=model.max_seq_len)

        with torch.no_grad():
            wrapper_logits = wrapper(torch.from_numpy(item_ids), torch.from_numpy(last_index)).squeeze(0)
        expected_logits = _reference_logits(net, model.item_to_idx, known_items, model.max_seq_len)

        assert torch.allclose(wrapper_logits, expected_logits, atol=1e-5), (
            f"{model_name}: export wrapper diverges from encode()/project() for session {known_items}"
        )


def test_prepare_input_pads_and_truncates():
    item_ids, last_index = prepare_input([5, 6, 7], max_seq_len=5)
    assert item_ids.shape == (1, 5)
    assert item_ids.tolist() == [[5, 6, 7, 0, 0]]
    assert last_index.tolist() == [2]

    # session longer than max_seq_len: keep only the most recent max_seq_len items
    item_ids, last_index = prepare_input([1, 2, 3, 4, 5, 6, 7], max_seq_len=3)
    assert item_ids.tolist() == [[5, 6, 7]]
    assert last_index.tolist() == [2]


def test_prepare_input_rejects_empty_session():
    with pytest.raises(ValueError):
        prepare_input([], max_seq_len=5)
