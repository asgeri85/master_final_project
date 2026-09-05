"""PyTorch checkpoint -> static-shape ONNX export for on-device conversion.

`_GRU4RecNet`/`_LSTMNet`/`_SASRecNet` (src/models/{gru4rec,lstm,sasrec}.py)
expose `encode()`/`project()`, not a single exportable `forward()`, and
`encode()` uses `pack_padded_sequence`/`pad_packed_sequence` internally, which
`torch.onnx.export` handles poorly (data-dependent control flow). The wrappers
here instead run the recurrent net over the *full* padded max_seq_len (no
packing) and gather the hidden state at an explicit `last_index` input --
mathematically identical to `encode()` for a causal RNN, since padding steps
after the true sequence length cannot affect earlier hidden states.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn

from src.models.gru4rec import _GRU4RecNet
from src.models.lstm import _LSTMNet
from src.models.sasrec import _SASRecNet

PAD_IDX = 0  # matches src/models/{gru4rec,lstm,sasrec}.py

ModelName = Literal["gru4rec", "lstm", "sasrec"]


def load_checkpoint(path: str | Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def get_vocab(payload: dict, train_path: str | None = None) -> tuple[dict[int, int], np.ndarray]:
    """item_to_idx/idx_to_item are saved inside every checkpoint payload here
    (see _save_checkpoint in src/models/{gru4rec,lstm,sasrec}.py) -- read them
    directly. Falls back to rebuilding from train_path (same construction as
    Model.fit(): sorted unique itemid, 1-indexed, 0=PAD) only if a checkpoint
    ever lacks these keys.
    """
    if "item_to_idx" in payload and "idx_to_item" in payload:
        return payload["item_to_idx"], np.asarray(payload["idx_to_item"])
    if train_path is None:
        raise ValueError("checkpoint has no item_to_idx/idx_to_item and no train_path given to rebuild it")
    import pandas as pd

    train = pd.read_parquet(train_path)
    items = np.sort(train["itemid"].unique())
    item_to_idx = {int(item): idx + 1 for idx, item in enumerate(items)}
    return item_to_idx, items


def build_net(model_name: ModelName, payload: dict, n_items: int) -> nn.Module:
    hp = payload["hyperparameters"]
    if model_name == "gru4rec":
        net = _GRU4RecNet(n_items, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    elif model_name == "lstm":
        net = _LSTMNet(n_items, hp["hidden_size"], hp["num_layers"], hp["dropout"])
    elif model_name == "sasrec":
        net = _SASRecNet(n_items, hp["hidden_size"], hp["num_heads"], hp["num_blocks"], hp["dropout"], hp["max_seq_len"])
    else:
        raise ValueError(f"unknown model_name: {model_name}")
    net.load_state_dict(payload["model_state_dict"])
    net.eval()
    return net


def prepare_input(known_idx_seq: list[int], max_seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """known_idx_seq: non-empty list of embedding indices (already mapped
    through item_to_idx, PAD_IDX not included). Truncates to the most recent
    max_seq_len items (same rule as Model.predict()), right-pads with
    PAD_IDX=0. Returns (item_ids (1, max_seq_len) int64, last_index (1,) int64).
    """
    if not known_idx_seq:
        raise ValueError("known_idx_seq must be non-empty")
    truncated = known_idx_seq[-max_seq_len:]
    item_ids = np.zeros((1, max_seq_len), dtype=np.int64)
    item_ids[0, : len(truncated)] = truncated
    last_index = np.array([len(truncated) - 1], dtype=np.int64)
    return item_ids, last_index


def _manual_gru_forward(gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    """Per-timestep GRU recurrence using only elementary ops (matmul/sigmoid/
    tanh), numerically identical to nn.GRU(x) (verified: max diff ~6e-8) but
    without ever emitting PyTorch's single fused ONNX GRU op. That fused op's
    export includes an Expand node that onnx2tf cannot turn into a valid
    Functional-model output ("Output tensors of a Functional model must be the
    output of a TensorFlow Layer") -- the pure-elementary-op graph produced
    here converts cleanly instead.

    Gate splitting uses literal integer bounds (gi[:, 0:H], not gi.chunk(3)):
    PyTorch's ONNX exporter lowers `.chunk()` into a Shape/Gather/Div-computed
    dynamic split even when the split sizes are compile-time static, and
    onnx2tf's TFLite lowering of that pattern has a bug that fails at
    interpreter-invoke time with "scatter_nd index out of bounds" (verified
    empirically -- literal-bound slicing avoids the dynamic-shape codegen
    entirely and converts/invokes correctly)."""
    batch, seq_len, _ = x.shape
    H = gru.hidden_size
    layer_input = x
    for layer in range(gru.num_layers):
        w_ih = getattr(gru, f"weight_ih_l{layer}")
        w_hh = getattr(gru, f"weight_hh_l{layer}")
        b_ih = getattr(gru, f"bias_ih_l{layer}")
        b_hh = getattr(gru, f"bias_hh_l{layer}")
        h = torch.zeros(batch, H, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = layer_input[:, t, :]
            gi = x_t @ w_ih.t() + b_ih
            gh = h @ w_hh.t() + b_hh
            i_r, i_z, i_n = gi[:, 0:H], gi[:, H : 2 * H], gi[:, 2 * H : 3 * H]
            h_r, h_z, h_n = gh[:, 0:H], gh[:, H : 2 * H], gh[:, 2 * H : 3 * H]
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outputs.append(h.unsqueeze(1))
        layer_input = torch.cat(outputs, dim=1)
    return layer_input


def _manual_lstm_forward(lstm: nn.LSTM, x: torch.Tensor) -> torch.Tensor:
    """LSTM counterpart of _manual_gru_forward -- same rationale, including the
    literal-bound gate slicing (see _manual_gru_forward's docstring)."""
    batch, seq_len, _ = x.shape
    H = lstm.hidden_size
    layer_input = x
    for layer in range(lstm.num_layers):
        w_ih = getattr(lstm, f"weight_ih_l{layer}")
        w_hh = getattr(lstm, f"weight_hh_l{layer}")
        b_ih = getattr(lstm, f"bias_ih_l{layer}")
        b_hh = getattr(lstm, f"bias_hh_l{layer}")
        h = torch.zeros(batch, H, dtype=x.dtype)
        c = torch.zeros(batch, H, dtype=x.dtype)
        outputs = []
        for t in range(seq_len):
            x_t = layer_input[:, t, :]
            gates = x_t @ w_ih.t() + b_ih + h @ w_hh.t() + b_hh
            i, f, g, o = gates[:, 0:H], gates[:, H : 2 * H], gates[:, 2 * H : 3 * H], gates[:, 3 * H : 4 * H]
            i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h.unsqueeze(1))
        layer_input = torch.cat(outputs, dim=1)
    return layer_input


class _GruLstmExportWrapper(nn.Module):
    """Shared wrapper for GRU4Rec and LSTM -- both nets are
    embedding -> (nn.GRU | nn.LSTM) -> linear, structurally identical from the
    export wrapper's point of view. Runs the recurrence via a manual unroll
    (see _manual_gru_forward/_manual_lstm_forward) instead of calling
    self.rnn(emb) directly, for ONNX/TFLite-conversion robustness."""

    def __init__(self, embedding: nn.Embedding, rnn: nn.Module, output: nn.Linear) -> None:
        super().__init__()
        self.embedding = embedding
        self.rnn = rnn
        self.output = output
        self._is_lstm = isinstance(rnn, nn.LSTM)

    def forward(self, item_ids: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(item_ids)  # (1, T, H); input_dropout is a no-op in eval mode, omitted
        if self._is_lstm:
            rnn_out = _manual_lstm_forward(self.rnn, emb)
        else:
            rnn_out = _manual_gru_forward(self.rnn, emb)
        # batch is statically 1 (export/inference contract): a single-axis
        # index_select on the squeezed (T, H) tensor is equivalent to the
        # two-tensor rnn_out[batch_idx, last_index] gather but avoids emitting
        # the "aten::index" advanced-indexing op pattern (Reshape/Transpose/
        # Concat/Gather combo) that onnx2tf mis-converts into a ScatterND op
        # which then fails at TFLite-interpreter invoke time with "scatter_nd
        # index out of bounds", even though ONNX Runtime runs it correctly.
        last_hidden = torch.index_select(rnn_out.squeeze(0), dim=0, index=last_index)  # (1, H)
        return self.output(last_hidden)  # (1, n_items+1)


class _SasrecExportWrapper(nn.Module):
    def __init__(
        self,
        item_embedding: nn.Embedding,
        pos_embedding: nn.Embedding,
        blocks: nn.TransformerEncoder,
        output: nn.Linear,
    ) -> None:
        super().__init__()
        self.item_embedding = item_embedding
        self.pos_embedding = pos_embedding
        self.blocks = blocks
        self.output = output

    def forward(self, item_ids: torch.Tensor, last_index: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = item_ids.shape
        positions = torch.arange(seq_len, device=item_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        emb = self.item_embedding(item_ids) + self.pos_embedding(positions)  # input_dropout no-op in eval mode

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=item_ids.device), diagonal=1)
        key_padding_mask = item_ids == PAD_IDX

        hidden = self.blocks(emb, mask=causal_mask, src_key_padding_mask=key_padding_mask)  # (1, T, H)
        # see _GruLstmExportWrapper.forward for why a single-axis index_select
        # is used instead of the two-tensor hidden[batch_idx, last_index] gather
        last_hidden = torch.index_select(hidden.squeeze(0), dim=0, index=last_index)  # (1, H)
        return self.output(last_hidden)  # (1, n_items+1)


def build_export_wrapper(model_name: ModelName, net: nn.Module) -> nn.Module:
    if model_name == "gru4rec":
        return _GruLstmExportWrapper(net.embedding, net.gru, net.output)
    if model_name == "lstm":
        return _GruLstmExportWrapper(net.embedding, net.lstm, net.output)
    if model_name == "sasrec":
        return _SasrecExportWrapper(net.item_embedding, net.pos_embedding, net.blocks, net.output)
    raise ValueError(f"unknown model_name: {model_name}")


def export_to_onnx(wrapper: nn.Module, max_seq_len: int, out_path: str | Path, opset: int = 17) -> None:
    wrapper.eval()
    dummy_item_ids = torch.zeros((1, max_seq_len), dtype=torch.long)
    dummy_item_ids[0, 0] = 1  # at least one real (non-pad) token, matching real usage
    dummy_last_index = torch.tensor([0], dtype=torch.long)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_item_ids, dummy_last_index),
        str(out_path),
        input_names=["item_ids", "last_index"],
        output_names=["logits"],
        opset_version=opset,
        dynamo=False,  # static shapes throughout; classic exporter is more predictable here
    )


def check_onnx_matches_pytorch(
    wrapper: nn.Module,
    onnx_path: str | Path,
    sample_inputs: list[tuple[np.ndarray, np.ndarray]],
    atol: float = 1e-4,
) -> float:
    """Runs the same (item_ids, last_index) samples through the PyTorch wrapper
    and the exported ONNX graph; returns the max abs logit difference observed.
    Raises AssertionError if any sample exceeds atol -- this is the gate that
    must pass before trusting anything downstream (TFLite conversion)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    wrapper.eval()
    max_diff = 0.0
    with torch.no_grad():
        for item_ids, last_index in sample_inputs:
            torch_out = wrapper(torch.from_numpy(item_ids), torch.from_numpy(last_index)).numpy()
            onnx_out = session.run(["logits"], {"item_ids": item_ids, "last_index": last_index})[0]
            diff = float(np.abs(torch_out - onnx_out).max())
            max_diff = max(max_diff, diff)
            assert diff <= atol, f"ONNX output diverges from PyTorch by {diff} (atol={atol})"
    return max_diff
