"""Evaluation harness: slide over session prefixes and score a model.

For a session with items [i1, ..., iL] (L >= 2), yields (L-1) (context, true_next)
pairs: ([i1], i2), ([i1, i2], i3), ..., ([i1, ..., i_{L-1}], iL).
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from src.eval.metrics import average_at_k, average_multi, evaluate_pair


def iter_session_prefixes(split_df: pd.DataFrame) -> Iterator[dict]:
    """Yield one dict per (context prefix, true next item) within each session:
    {session_id, items, timestamps, true_item}. `items`/`timestamps` are
    parallel lists for the context so far (needed for temporal features that
    `iter_eval_pairs`'s plain item-id lists can't carry).

    Sessions with fewer than 2 surviving items are skipped (no next item to
    predict) — this also guards against any session left truncated to length
    1 by an upstream time-based split boundary.
    """
    for session_id, group in split_df.sort_values("timestamp").groupby("session_id", sort=False):
        items = group["itemid"].tolist()
        timestamps = group["timestamp"].tolist()
        if len(items) < 2:
            continue
        for t in range(1, len(items)):
            yield {
                "session_id": session_id,
                "items": items[:t],
                "timestamps": timestamps[:t],
                "true_item": items[t],
            }


def iter_eval_pairs(split_df: pd.DataFrame) -> Iterator[tuple[list[int], int]]:
    """Yield (context_items, true_next_item) pairs for every session in split_df."""
    for ctx in iter_session_prefixes(split_df):
        yield ctx["items"], ctx["true_item"]


def evaluate_model(model, split_df: pd.DataFrame, k: int = 10) -> dict[str, float]:
    """Run model.predict() over every eval pair in split_df; return averaged HR/NDCG/MRR@K."""

    def pairs():
        for context, true_item in iter_eval_pairs(split_df):
            yield model.predict(context), true_item

    return average_at_k(pairs(), k)


def evaluate_model_multi(model, split_df: pd.DataFrame, specs: list[tuple[str, int]]) -> dict[str, float]:
    """Like `evaluate_model`, but for an arbitrary (metric_name, k) spec list --
    e.g. LSApp's `[("hr", 1), ("hr", 3), ("mrr", 5)]` (karar-notu SS3 exception),
    where a single shared K across HR/NDCG/MRR doesn't apply."""

    def pairs():
        for context, true_item in iter_eval_pairs(split_df):
            yield model.predict(context), true_item

    return average_multi(pairs(), specs)


def evaluate_per_pair_and_session(
    model, split_df: pd.DataFrame, k: int = 10, batched: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(session, prefix) HR/NDCG/MRR@K, both ungrouped (one row per prefix --
    its unweighted mean equals average_at_k/evaluate_model's prefix-weighted metric,
    the number every run_*.py script logs to results/runs.jsonl) and grouped to
    per-session means (the input a session-level paired Wilcoxon signed-rank test
    needs, karar-notu SS6 -- unweighted mean of THIS differs from the prefix-weighted
    one whenever session length correlates with the metric, which is expected and
    not a reproducibility bug).

    batched=False calls model.predict(items) once per prefix (cheap deterministic
    models: Item-KNN, S-POP, POP). batched=True collects every prefix's context
    first and calls model.predict_many(contexts) once (GRU4Rec/LSTM/SASRec -- avoids
    tens of thousands of unbatched single-session GPU forward passes, same
    motivation as the run_*.py scripts' _evaluate_fast).
    """
    if batched:
        contexts, session_ids, true_items = [], [], []
        for ctx in iter_session_prefixes(split_df):
            contexts.append(ctx["items"])
            session_ids.append(ctx["session_id"])
            true_items.append(ctx["true_item"])
        ranked_lists = model.predict_many(contexts)
        rows = [
            {"session_id": sid, **evaluate_pair(ranked, true_item, k)}
            for sid, ranked, true_item in zip(session_ids, ranked_lists, true_items)
        ]
    else:
        rows = []
        for ctx in iter_session_prefixes(split_df):
            ranked = model.predict(ctx["items"])
            rows.append({"session_id": ctx["session_id"], **evaluate_pair(ranked, ctx["true_item"], k)})

    per_pair = pd.DataFrame(rows)
    per_session = per_pair.groupby("session_id").mean(numeric_only=True)
    return per_pair, per_session


def per_session_metrics(model, split_df: pd.DataFrame, k: int = 10, batched: bool = False) -> pd.DataFrame:
    """Per-session means only -- see evaluate_per_pair_and_session for details."""
    _, per_session = evaluate_per_pair_and_session(model, split_df, k=k, batched=batched)
    return per_session


def count_eval_pairs(split_df: pd.DataFrame) -> int:
    """Number of (context, true_next_item) pairs iter_eval_pairs would yield."""
    lengths = split_df.groupby("session_id", sort=False).size()
    return int((lengths[lengths >= 2] - 1).sum())
