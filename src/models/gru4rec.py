"""GRU4Rec (Hidasi et al., 2016): session-based sequential recommender.

First PyTorch model in this repo (POP/S-POP/Item-KNN/XGBoost are numpy/pandas/
sklearn-style). Trains a GRU over a local item vocabulary built from train
(same convention as ItemKnnModel: sorted(train["itemid"].unique())), with a
full-softmax output layer over that vocabulary (no sampled softmax/negative
sampling in the training loss -- catalog sizes here are small enough, ~46k
items for RetailRocket, that a dense output layer is feasible on GPU).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from src.models.pop import PopModel

PAD_IDX = 0  # embedding index 0 reserved for padding; real items start at 1


class _GRU4RecNet(nn.Module):
    def __init__(self, n_items: int, hidden_size: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_items + 1, hidden_size, padding_idx=PAD_IDX)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, n_items + 1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, T) padded item indices (0=pad). Returns (B, T, H) hidden states."""
        emb = self.input_dropout(self.embedding(x))
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        return self.output_dropout(out)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (..., H) -> (..., n_items+1) logits. Call only on valid
        (non-pad) positions -- never on the full padded (B,T,H) tensor, since
        most positions in a batch are padding (median session length is 2)."""
        return self.output(hidden)


def _build_sequences(train: pd.DataFrame, item_to_idx: dict[int, int], max_seq_len: int) -> list[np.ndarray]:
    """One array of embedding indices per session, ordered by session_event_index,
    truncated to the most recent (max_seq_len + 1) items (need input=seq[:-1],
    target=seq[1:])."""
    sequences = []
    ordered = train.sort_values(["session_id", "session_event_index"])
    for _, items in ordered.groupby("session_id", sort=False)["itemid"]:
        idx_seq = [item_to_idx[i] for i in items.tolist()]
        if len(idx_seq) < 2:
            continue
        if len(idx_seq) > max_seq_len + 1:
            idx_seq = idx_seq[-(max_seq_len + 1):]
        sequences.append(np.asarray(idx_seq, dtype=np.int64))
    return sequences


def _iter_batches(sequences: list[np.ndarray], batch_size: int, rng: np.random.Generator):
    order = rng.permutation(len(sequences))
    for start in range(0, len(order), batch_size):
        batch = [sequences[i] for i in order[start : start + batch_size]]
        inputs, targets = [s[:-1] for s in batch], [s[1:] for s in batch]
        lengths = np.array([len(s) for s in inputs], dtype=np.int64)
        max_len = int(lengths.max())
        x = np.zeros((len(batch), max_len), dtype=np.int64)
        y = np.zeros((len(batch), max_len), dtype=np.int64)  # 0 = pad = ignored via mask
        for i, (inp, tgt) in enumerate(zip(inputs, targets)):
            x[i, : len(inp)] = inp
            y[i, : len(tgt)] = tgt
        yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(lengths)


class Gru4RecModel:
    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_seq_len: int = 50,
        max_epochs: int = 50,
        patience: int = 5,
        seed: int = 42,
        device: str | None = None,
        checkpoint_dir: str | None = None,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._pop = PopModel()
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: np.ndarray = np.array([], dtype=np.int64)
        self._net: _GRU4RecNet | None = None
        self.history: list[dict] = []

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> None:
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self._pop.fit(train)

        items = np.sort(train["itemid"].unique())
        self.idx_to_item = items
        self.item_to_idx = {int(item): idx + 1 for idx, item in enumerate(items)}
        n_items = len(items)

        sequences = _build_sequences(train, self.item_to_idx, self.max_seq_len)

        self._net = _GRU4RecNet(n_items, self.hidden_size, self.num_layers, self.dropout).to(self.device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.learning_rate)

        start_epoch = 1
        best_state: dict | None = None
        best_val_ndcg = -float("inf")
        epochs_without_improvement = 0

        resumed = self._maybe_resume(optimizer)
        if resumed is not None:
            start_epoch = resumed["epoch"] + 1
            best_val_ndcg = resumed.get("best_val_ndcg", best_val_ndcg)
            epochs_without_improvement = resumed.get("epochs_without_improvement", 0)

        for epoch in range(start_epoch, self.max_epochs + 1):
            self._net.train()
            epoch_loss, n_batches = 0.0, 0
            for x, y, lengths in _iter_batches(sequences, self.batch_size, rng):
                x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                optimizer.zero_grad()
                hidden = self._net.encode(x, lengths)
                mask = y != PAD_IDX
                logits = self._net.project(hidden[mask])
                loss = F.cross_entropy(logits, y[mask])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            val_ndcg = self._fast_val_ndcg10(val) if val is not None else None
            self.history.append(
                {"epoch": epoch, "train_loss": epoch_loss / max(n_batches, 1), "val_ndcg@10": val_ndcg}
            )

            is_best = val_ndcg is not None and val_ndcg > best_val_ndcg
            if is_best:
                best_val_ndcg = val_ndcg
                epochs_without_improvement = 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            elif val_ndcg is not None:
                epochs_without_improvement += 1

            self._save_checkpoint(epoch, optimizer, best_val_ndcg, epochs_without_improvement, is_best)

            if val is not None and epochs_without_improvement >= self.patience:
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)

    def _fast_val_ndcg10(self, val: pd.DataFrame, k: int = 10) -> float:
        """Vectorized, last-item-only NDCG@10 proxy for early stopping.

        The real eval harness calls predict() once per session-prefix (tens
        of thousands of Python-level forward passes) -- too slow to run every
        epoch. This proxy scores every session's final transition in one
        batched forward pass. The metric actually reported after training
        uses the real harness (src/eval/harness.py), same as every other
        model in this repo.
        """
        self._net.eval()
        sequences = []
        for _, items in val.sort_values(["session_id", "session_event_index"]).groupby("session_id", sort=False)[
            "itemid"
        ]:
            known = [self.item_to_idx[i] for i in items.tolist() if i in self.item_to_idx]
            if len(known) < 2:
                continue
            known = known[-(self.max_seq_len + 1):]
            sequences.append(np.asarray(known, dtype=np.int64))
        if not sequences:
            return 0.0

        total_ndcg, n = 0.0, 0
        rng = np.random.default_rng(0)  # deterministic batching order for this proxy metric
        with torch.no_grad():
            for x, y, lengths in _iter_batches(sequences, 512, rng):
                x, y, lengths = x.to(self.device), y.to(self.device), lengths.to(self.device)
                hidden = self._net.encode(x, lengths)  # (B, T, H)
                last_hidden = hidden[torch.arange(len(lengths), device=self.device), lengths - 1]  # (B, H)
                logits = self._net.project(last_hidden)  # (B, n_items+1)
                true_idx = y[torch.arange(len(lengths), device=self.device), lengths - 1]  # (B,)
                topk_idx = logits.topk(k, dim=-1).indices  # (B, k)
                hit_pos = (topk_idx == true_idx.unsqueeze(1)).nonzero()  # (n_hits, 2)
                ndcg_per_row = torch.zeros(len(lengths), device=self.device)
                if hit_pos.numel():
                    rows, cols = hit_pos[:, 0], hit_pos[:, 1]
                    ndcg_per_row[rows] = 1.0 / torch.log2(cols.float() + 2.0)  # rank = col+1
                total_ndcg += ndcg_per_row.sum().item()
                n += len(lengths)
        return total_ndcg / n

    def predict(self, session: Sequence[int]) -> list[int]:
        known = [item for item in session if item in self.item_to_idx]
        if not known:
            return self._pop.predict(session)

        known = known[-self.max_seq_len :]
        idx_seq = torch.tensor([[self.item_to_idx[i] for i in known]], dtype=torch.long, device=self.device)
        length = torch.tensor([len(known)], dtype=torch.long, device=self.device)

        self._net.eval()
        with torch.no_grad():
            hidden = self._net.encode(idx_seq, length)  # (1, T, H)
            last_hidden = hidden[0, length[0] - 1].unsqueeze(0)  # (1, H)
            logits = self._net.project(last_hidden).squeeze(0)  # (n_items+1,)

        scores = logits[1:].cpu().numpy()  # drop pad index 0; remaining n_items align with idx_to_item
        order = np.lexsort((self.idx_to_item, -scores))  # tie-break: score desc, item id asc
        return self.idx_to_item[order].tolist()

    def _maybe_resume(self, optimizer: torch.optim.Optimizer) -> dict | None:
        if self.checkpoint_dir is None:
            return None
        latest = Path(self.checkpoint_dir) / f"gru4rec_seed{self.seed}_latest.pt"
        if not latest.exists():
            return None
        payload = torch.load(latest, map_location=self.device, weights_only=False)
        self._net.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        return payload

    def _save_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        best_val_ndcg: float,
        epochs_without_improvement: int,
        is_best: bool,
    ) -> None:
        if self.checkpoint_dir is None:
            return
        ckpt_dir = Path(self.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "seed": self.seed,
            "model_state_dict": self._net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item,
            "best_val_ndcg": best_val_ndcg,
            "epochs_without_improvement": epochs_without_improvement,
            "hyperparameters": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "max_seq_len": self.max_seq_len,
            },
        }
        torch.save(payload, ckpt_dir / f"gru4rec_seed{self.seed}_epoch{epoch:03d}.pt")
        torch.save(payload, ckpt_dir / f"gru4rec_seed{self.seed}_latest.pt")
        if is_best:
            torch.save(payload, ckpt_dir / f"gru4rec_seed{self.seed}_best.pt")
