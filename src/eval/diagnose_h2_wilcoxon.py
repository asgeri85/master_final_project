"""Diagnostic re-run of the H2 ablation to inspect the Wilcoxon test's zero-difference
handling and persist per-session metrics for further analysis (e.g. a difference histogram).

Reuses the Optuna-selected best hyperparameters from the original run (logged in
results/runs.jsonl) so it doesn't repeat the 20-trial search -- only rebuilds the
train candidate table (unavoidable, nothing was persisted) and refits both finals.

CLI usage: python -m src.eval.diagnose_h2_wilcoxon
"""

from __future__ import annotations

import time

import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.eval.harness import iter_session_prefixes
from src.eval.metrics import evaluate_pair
from src.features.build_candidate_table import build_candidate_table
from src.features.feature_builder import FeatureBuilder
from src.models.item_knn import ItemKnnModel
from src.models.xgboost_ranker import XGBoostRankerModel

BEST_PARAMS = {
    "raw": {"n_estimators": 479, "max_depth": 3, "learning_rate": 0.013488204720120515},
    "rich": {"n_estimators": 173, "max_depth": 5, "learning_rate": 0.05958389350068958},
}


def _per_session_metrics(model: XGBoostRankerModel, split_df: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for ctx in iter_session_prefixes(split_df):
        ranked = model.predict_from_context(ctx["items"], ctx["timestamps"])
        result = evaluate_pair(ranked, ctx["true_item"], k)
        rows.append({"session_id": ctx["session_id"], **result})
    per_pair = pd.DataFrame(rows)
    return per_pair.groupby("session_id").mean(numeric_only=True)


def main() -> None:
    config = yaml.safe_load(open("configs/xgboost_raw.yaml"))
    k = config["eval"]["k"]

    train = pd.read_parquet(config["data"]["train_path"])
    val = pd.read_parquet(config["data"]["val_path"])

    item_knn = ItemKnnModel(neighborhood_size=config["candidate_generation"]["item_knn_neighborhood_size"])
    item_knn.fit(train)
    feature_builder = FeatureBuilder()
    feature_builder.fit(train)

    print("Building train candidate table...")
    t0 = time.monotonic()
    train_table = build_candidate_table(
        item_knn, feature_builder, train,
        n_candidates=config["candidate_generation"]["n_candidates"],
        include_label=True,
        negatives_per_positive=config["training"]["negatives_per_positive"],
        random_seed=config["training"]["random_seed"],
    )
    print(f"  {len(train_table):,} rows ({time.monotonic() - t0:.1f}s)")

    per_session = {}
    for feature_set in ("raw", "rich"):
        print(f"Fitting XGBoost-{feature_set} with best_params={BEST_PARAMS[feature_set]}...")
        model = XGBoostRankerModel(
            item_knn=item_knn, feature_builder=feature_builder, feature_set=feature_set,
            xgb_params=BEST_PARAMS[feature_set],
            n_candidates=config["candidate_generation"]["n_candidates"],
            negatives_per_positive=config["training"]["negatives_per_positive"],
            random_seed=config["training"]["random_seed"],
        )
        model.fit_from_table(train_table)
        t0 = time.monotonic()
        per_session[feature_set] = _per_session_metrics(model, val, k)
        print(f"  full val eval done ({time.monotonic() - t0:.1f}s), "
              f"HR@{k}={per_session[feature_set][f'hr@{k}'].mean():.4f}")

    per_session["raw"].to_parquet("results/h2_per_session_raw.parquet")
    per_session["rich"].to_parquet("results/h2_per_session_rich.parquet")
    print("Saved results/h2_per_session_{raw,rich}.parquet")

    common = per_session["raw"].index.intersection(per_session["rich"].index)
    raw_a, rich_a = per_session["raw"].loc[common], per_session["rich"].loc[common]

    for metric in (f"hr@{k}", f"ndcg@{k}"):
        diff = rich_a[metric] - raw_a[metric]
        n_zero = int((diff == 0).sum())
        n_pos = int((diff > 0).sum())
        n_neg = int((diff < 0).sum())
        print(f"\n[{metric}] n_sessions={len(diff)} zero={n_zero} positive(rich>raw)={n_pos} negative(rich<raw)={n_neg}")

        for zero_method in ("wilcox", "pratt", "zsplit"):
            stat, p = wilcoxon(rich_a[metric], raw_a[metric], zero_method=zero_method)
            print(f"  zero_method={zero_method:8s} statistic={stat:.2f} p={p:.6e}")

    diffs = pd.DataFrame({
        f"hr@{k}_diff": rich_a[f"hr@{k}"] - raw_a[f"hr@{k}"],
        f"ndcg@{k}_diff": rich_a[f"ndcg@{k}"] - raw_a[f"ndcg@{k}"],
    })
    diffs.to_parquet("results/h2_session_diffs.parquet")
    print("Saved results/h2_session_diffs.parquet")


if __name__ == "__main__":
    main()
