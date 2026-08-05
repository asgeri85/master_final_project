"""H2 ablation: XGBoost-raw (Item-KNN score only) vs XGBoost-rich (12 A5 features).

Both re-rank the SAME Item-KNN top-100 candidate pool per session prefix
(karar-notu Katman-2 design), tuned with Optuna within the locked 20-trial/
model budget (SS12a A6). Compares HR@10/NDCG@10 with a session-level paired
Wilcoxon signed-rank test + Holm-Bonferroni correction (SS6, locked H2 plan).
Logs both final runs to results/runs.jsonl.

CLI usage: python -m src.eval.run_h2_ablation
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
import yaml
from scipy.stats import wilcoxon

from src.eval.harness import iter_session_prefixes
from src.eval.metrics import evaluate_pair
from src.features.build_candidate_table import build_candidate_table
from src.features.feature_builder import FEATURE_COLUMNS, RAW_FEATURE_COLUMNS, FeatureBuilder
from src.models.item_knn import ItemKnnModel
from src.models.xgboost_ranker import XGBoostRankerModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIG_PATHS = {"raw": "configs/xgboost_raw.yaml", "rich": "configs/xgboost_rich.yaml"}


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _feature_columns(feature_set: str) -> list[str]:
    return RAW_FEATURE_COLUMNS if feature_set == "raw" else FEATURE_COLUMNS


def _make_predict_fn(booster: xgb.Booster, item_knn: ItemKnnModel, feature_builder: FeatureBuilder, feature_set: str, n_candidates: int):
    feature_columns = _feature_columns(feature_set)

    def predict_fn(items: list[int], timestamps: list[int]) -> list[int]:
        candidates = item_knn.top_candidates(items, n=n_candidates)
        candidate_items = [c for c, _ in candidates]
        knn_scores = {c: s for c, s in candidates}
        feats = feature_builder.build_candidate_features(items, timestamps, candidate_items, knn_scores)
        dmat = xgb.DMatrix(feats[feature_columns].to_numpy(dtype=np.float32), feature_names=feature_columns)
        scores = booster.predict(dmat)
        candidate_arr = np.asarray(candidate_items, dtype=np.int64)
        order = np.lexsort((candidate_arr, -scores))
        return candidate_arr[order].tolist()

    return predict_fn


def _per_session_metrics(predict_fn, split_df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-(session, prefix) HR/NDCG@K, aggregated to per-session means for the paired test."""
    rows = []
    for ctx in iter_session_prefixes(split_df):
        ranked = predict_fn(ctx["items"], ctx["timestamps"])
        result = evaluate_pair(ranked, ctx["true_item"], k)
        rows.append({"session_id": ctx["session_id"], **result})
    per_pair = pd.DataFrame(rows)
    return per_pair.groupby("session_id").mean(numeric_only=True)


def _tune(
    feature_set: str,
    config: dict,
    item_knn: ItemKnnModel,
    feature_builder: FeatureBuilder,
    train_table: pd.DataFrame,
    val_search_df: pd.DataFrame,
    k: int,
) -> tuple[dict, float]:
    feature_columns = _feature_columns(feature_set)
    X = train_table[feature_columns].to_numpy(dtype=np.float32)
    y = train_table["label"].to_numpy(dtype=np.float32)
    space = config["optuna"]["search_space"]
    seed = config["training"]["random_seed"]

    def objective(trial: optuna.Trial) -> float:
        n_estimators = trial.suggest_int("n_estimators", space["n_estimators"]["low"], space["n_estimators"]["high"])
        params = {
            "max_depth": trial.suggest_int("max_depth", space["max_depth"]["low"], space["max_depth"]["high"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", space["learning_rate"]["low"], space["learning_rate"]["high"], log=True
            ),
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": seed,
        }
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_columns)
        booster = xgb.train(params, dtrain, num_boost_round=n_estimators)

        predict_fn = _make_predict_fn(
            booster, item_knn, feature_builder, feature_set, config["candidate_generation"]["n_candidates"]
        )
        per_session = _per_session_metrics(predict_fn, val_search_df, k)
        return float(per_session[f"hr@{k}"].mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=config["optuna"]["n_trials"])
    return study.best_params, study.best_value


def _holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        candidate = (m - rank) * pvalues[i]
        running_max = max(running_max, candidate)
        adjusted[i] = min(running_max, 1.0)
    return adjusted


def run_ablation() -> dict:
    configs = {name: yaml.safe_load(Path(path).read_text()) for name, path in CONFIG_PATHS.items()}
    base_config = configs["raw"]
    k = base_config["eval"]["k"]

    train = pd.read_parquet(base_config["data"]["train_path"])
    val = pd.read_parquet(base_config["data"]["val_path"])

    item_knn = ItemKnnModel(neighborhood_size=base_config["candidate_generation"]["item_knn_neighborhood_size"])
    item_knn.fit(train)

    feature_builder = FeatureBuilder()
    feature_builder.fit(train)

    print("Building train candidate table (shared across raw/rich, with negative subsampling)...")
    build_start = time.monotonic()
    train_table = build_candidate_table(
        item_knn,
        feature_builder,
        train,
        n_candidates=base_config["candidate_generation"]["n_candidates"],
        include_label=True,
        negatives_per_positive=base_config["training"]["negatives_per_positive"],
        random_seed=base_config["training"]["random_seed"],
    )
    print(f"  train candidate table: {len(train_table):,} rows ({time.monotonic() - build_start:.1f}s)")

    rng = np.random.default_rng(base_config["training"]["random_seed"])
    all_sessions = val["session_id"].unique()
    n_search = min(base_config["optuna"]["search_val_sample_sessions"], len(all_sessions))
    search_sessions = rng.choice(all_sessions, size=n_search, replace=False)
    val_search_df = val[val["session_id"].isin(search_sessions)]
    print(f"Optuna trial scoring will use {n_search:,}/{len(all_sessions):,} val sessions.")

    results: dict[str, dict] = {}
    for feature_set, config in (("raw", configs["raw"]), ("rich", configs["rich"])):
        print(f"\n=== Tuning XGBoost-{feature_set} ({config['optuna']['n_trials']} trials) ===")
        tune_start = time.monotonic()
        best_params, best_search_hr = _tune(
            feature_set, config, item_knn, feature_builder, train_table, val_search_df, k
        )
        tune_duration = time.monotonic() - tune_start
        print(f"  best params: {best_params}  (search HR@{k}={best_search_hr:.4f}, {tune_duration:.1f}s)")

        fit_start = time.monotonic()
        model = XGBoostRankerModel(
            item_knn=item_knn,
            feature_builder=feature_builder,
            feature_set=feature_set,
            xgb_params=best_params,
            n_candidates=config["candidate_generation"]["n_candidates"],
            negatives_per_positive=config["training"]["negatives_per_positive"],
            random_seed=config["training"]["random_seed"],
        )
        model.fit_from_table(train_table)
        fit_duration = time.monotonic() - fit_start

        eval_start = time.monotonic()
        predict_fn = lambda items, ts, m=model: m.predict_from_context(items, ts)
        per_session = _per_session_metrics(predict_fn, val, k)
        eval_duration = time.monotonic() - eval_start
        print(
            f"  full val: HR@{k}={per_session[f'hr@{k}'].mean():.4f} "
            f"NDCG@{k}={per_session[f'ndcg@{k}'].mean():.4f} "
            f"MRR@{k}={per_session[f'mrr@{k}'].mean():.4f}  ({eval_duration:.1f}s)"
        )

        results[feature_set] = {
            "config": config,
            "best_params": best_params,
            "per_session_metrics": per_session,
            "metrics": {
                f"hr@{k}": float(per_session[f"hr@{k}"].mean()),
                f"ndcg@{k}": float(per_session[f"ndcg@{k}"].mean()),
                f"mrr@{k}": float(per_session[f"mrr@{k}"].mean()),
            },
            "tune_duration_seconds": round(tune_duration, 3),
            "fit_duration_seconds": round(fit_duration, 3),
            "eval_duration_seconds": round(eval_duration, 3),
        }

    # session-level paired Wilcoxon signed-rank test (H2, SS6), HR@10 and NDCG@10
    raw_sessions = results["raw"]["per_session_metrics"]
    rich_sessions = results["rich"]["per_session_metrics"]
    common_sessions = raw_sessions.index.intersection(rich_sessions.index)
    raw_aligned = raw_sessions.loc[common_sessions]
    rich_aligned = rich_sessions.loc[common_sessions]

    hr_stat, hr_p = wilcoxon(rich_aligned[f"hr@{k}"], raw_aligned[f"hr@{k}"])
    ndcg_stat, ndcg_p = wilcoxon(rich_aligned[f"ndcg@{k}"], raw_aligned[f"ndcg@{k}"])
    adjusted = _holm_bonferroni([hr_p, ndcg_p])

    test_result = {
        "n_sessions": int(len(common_sessions)),
        f"hr@{k}": {
            "statistic": float(hr_stat),
            "p_value": float(hr_p),
            "p_value_holm": float(adjusted[0]),
            "mean_diff_rich_minus_raw": float((rich_aligned[f"hr@{k}"] - raw_aligned[f"hr@{k}"]).mean()),
        },
        f"ndcg@{k}": {
            "statistic": float(ndcg_stat),
            "p_value": float(ndcg_p),
            "p_value_holm": float(adjusted[1]),
            "mean_diff_rich_minus_raw": float((rich_aligned[f"ndcg@{k}"] - raw_aligned[f"ndcg@{k}"]).mean()),
        },
    }

    return {"results": results, "wilcoxon_holm": test_result, "k": k}


def _log_run(feature_set: str, payload: dict, k: int) -> dict:
    metrics = payload["metrics"]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": f"xgboost_{feature_set}",
        "dataset": "retailrocket",
        "split": "val",
        "seed": None,
        "config": {
            **{key: value for key, value in payload["config"].items() if key != "optuna"},
            "optuna_best_params": payload["best_params"],
        },
        "metrics": metrics,
        "num_eval_pairs": int(payload["per_session_metrics"].shape[0]),
        "fit_duration_seconds": payload["fit_duration_seconds"],
        "eval_duration_seconds": payload["eval_duration_seconds"],
        "tune_duration_seconds": payload["tune_duration_seconds"],
        "duration_seconds": round(
            payload["tune_duration_seconds"] + payload["fit_duration_seconds"] + payload["eval_duration_seconds"], 3
        ),
        "git_commit": _git_commit(),
    }


def main() -> None:
    outcome = run_ablation()
    k = outcome["k"]

    os.makedirs("results", exist_ok=True)
    with open("results/runs.jsonl", "a") as f:
        for feature_set in ("raw", "rich"):
            run = _log_run(feature_set, outcome["results"][feature_set], k)
            f.write(json.dumps(run) + "\n")

    print("\n=== H2 Wilcoxon signed-rank + Holm-Bonferroni (session-level, rich vs raw) ===")
    print(json.dumps(outcome["wilcoxon_holm"], indent=2))


if __name__ == "__main__":
    main()
