"""Unit tests for src/eval/harness.py."""

import pandas as pd
import pytest

from src.eval.harness import count_eval_pairs, evaluate_per_pair_and_session, per_session_metrics


class _AlwaysHitModel:
    """Ranks the true next item first, regardless of context -- used only to
    pin down per_session_metrics' session_id grouping/averaging, not ranking quality."""

    def predict(self, items):
        return [items[-1] + 1] + [i for i in range(1, 100) if i != items[-1] + 1]

    def predict_many(self, contexts):
        return [self.predict(items) for items in contexts]


def _toy_split() -> pd.DataFrame:
    # session 1: items [1, 2, 3] -> prefixes ([1],2), ([1,2],3) -- both hits under _AlwaysHitModel
    # session 2: items [10, 11] -> prefix ([10],11) -- one hit
    return pd.DataFrame(
        {
            "session_id": [1, 1, 1, 2, 2],
            "itemid": [1, 2, 3, 10, 11],
            "timestamp": [1, 2, 3, 1, 2],
        }
    )


def test_per_session_metrics_groups_and_averages_unbatched():
    split_df = _toy_split()
    result = per_session_metrics(_AlwaysHitModel(), split_df, k=10, batched=False)
    assert set(result.index) == {1, 2}
    assert result.loc[1, "hr@10"] == pytest.approx(1.0)
    assert result.loc[2, "hr@10"] == pytest.approx(1.0)
    assert result.loc[1, "ndcg@10"] == pytest.approx(1.0)


def test_per_session_metrics_batched_matches_unbatched():
    split_df = _toy_split()
    model = _AlwaysHitModel()
    unbatched = per_session_metrics(model, split_df, k=10, batched=False)
    batched = per_session_metrics(model, split_df, k=10, batched=True)
    pd.testing.assert_frame_equal(unbatched.sort_index(), batched.sort_index())


def test_per_session_metrics_row_count_matches_eval_pairs():
    split_df = _toy_split()
    result = per_session_metrics(_AlwaysHitModel(), split_df, k=10, batched=False)
    # 2 sessions, but count_eval_pairs counts per-prefix pairs (3), not per-session rows (2)
    assert len(result) == 2
    assert count_eval_pairs(split_df) == 3


class _LastItemThresholdModel:
    """Hits iff the context's last item is >= 10 -- used to make the prefix-weighted
    mean (all pairs, equal weight) diverge from the per-session mean (one long,
    all-hit session vs. one short, all-miss session get equal weight each)."""

    def predict(self, items):
        true_item = items[-1] + 1
        if items[-1] >= 10:
            return [true_item] + [i for i in range(1, 100) if i != true_item]
        return [i for i in range(1, 100) if i != true_item]

    def predict_many(self, contexts):
        return [self.predict(items) for items in contexts]


def test_prefix_weighted_mean_differs_from_session_grouped_mean_by_session_length():
    # session 1: 1 prefix, always a miss (last item 1 < 10)
    # session 2: 3 prefixes, always a hit (last item >= 10)
    split_df = pd.DataFrame(
        {
            "session_id": [1, 1, 2, 2, 2, 2],
            "itemid": [1, 2, 10, 11, 12, 13],
            "timestamp": [1, 2, 1, 2, 3, 4],
        }
    )
    per_pair, per_session = evaluate_per_pair_and_session(_LastItemThresholdModel(), split_df, k=10, batched=False)

    assert per_pair["hr@10"].mean() == pytest.approx(0.75)  # (0 + 1 + 1 + 1) / 4 pairs
    assert per_session.loc[1, "hr@10"] == pytest.approx(0.0)
    assert per_session.loc[2, "hr@10"] == pytest.approx(1.0)
    assert per_session["hr@10"].mean() == pytest.approx(0.5)  # unweighted mean of session means
