"""On-device fizibilite, Aşama 2: Android benchmark uygulaması için sabit test
oturumu üretir (docs/00-karar-notu.md §12c).

RetailRocket train.parquet'ten session_id=3'ü alıp GRU4Rec checkpoint'indeki
item_to_idx ile aynı şekilde indeksler ve pad'ler (bkz. export_onnx.prepare_input) --
GRU4Rec ve LSTM aynı RetailRocket train verisinden `sorted(unique itemid)` ile
deterministik kurulduğu için tek bir item_to_idx her iki model için de geçerlidir.
Çıktı, Android uygulamasının assets/ altına koyacağı bir JSON dosyasıdır; ölçüm
sırasında her iki model/varyant için aynı sabit girdi kullanılır (adil karşılaştırma).

CLI usage:
  python -m src.ondevice.prepare_benchmark_session
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ondevice.export_onnx import get_vocab, load_checkpoint, prepare_input

SESSION_ID = 3  # RetailRocket train.parquet, 8 etkileşim -- bkz. plan
TRAIN_PATH = "data/processed/retailrocket/train.parquet"
CHECKPOINT_PATH = "checkpoints/gru4rec/seed_42/gru4rec_seed42_best.pt"
MAX_SEQ_LEN = 50  # configs/gru4rec.yaml / lstm.yaml training.max_seq_len ile eşleşmeli
OUT_PATH = "results/ondevice/benchmark_session.json"


def build_benchmark_session(
    train_path: str = TRAIN_PATH,
    checkpoint_path: str = CHECKPOINT_PATH,
    session_id: int = SESSION_ID,
    max_seq_len: int = MAX_SEQ_LEN,
) -> dict:
    train = pd.read_parquet(train_path)
    session = train[train["session_id"] == session_id].sort_values("session_event_index")
    if session.empty:
        raise ValueError(f"session_id={session_id} not found in {train_path}")
    raw_item_ids = session["itemid"].tolist()

    payload = load_checkpoint(checkpoint_path)
    item_to_idx, _ = get_vocab(payload, train_path=train_path)
    known_idx_seq = [item_to_idx[i] for i in raw_item_ids if i in item_to_idx]
    if len(known_idx_seq) != len(raw_item_ids):
        raise ValueError(f"session_id={session_id}: not all items are in {checkpoint_path}'s item_to_idx")

    item_ids, last_index = prepare_input(known_idx_seq, max_seq_len)
    return {
        "session_id": session_id,
        "raw_item_ids": raw_item_ids,
        "item_indices_padded": item_ids[0].tolist(),
        "last_index": int(last_index[0]),
        "seq_len": max_seq_len,
        "n_items": len(item_to_idx),
        "source": f"{train_path} session_id={session_id}",
    }


def main() -> None:
    benchmark_session = build_benchmark_session()
    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(benchmark_session, indent=2) + "\n")
    print(f"Yazıldı: {out_path}")


if __name__ == "__main__":
    main()
