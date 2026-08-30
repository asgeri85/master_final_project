"""LSTM: Optuna-tuned (20 trials, A6 search space), final model trained and
evaluated with 5 seeds (karar-notu SS6, CLAUDE.md rule 9). Logs one row per
seed plus one mean±std summary row to results/runs.jsonl.

CLI usage:
  python -m src.eval.run_lstm
  python -m src.eval.run_lstm --config configs/lstm_lsapp.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml

from src.eval.harness import count_eval_pairs, iter_eval_pairs
from src.eval.metrics import average_multi
from src.models.lstm import LstmModel

optuna.logging.set_verbosity(optuna.logging.WARNING)
CONFIG_PATH = "configs/lstm.yaml"


def _eval_specs(config: dict) -> list[tuple[str, int]]:
    """(metric_name, k) pairs to compute. RetailRocket's `eval.k` expands to the
    default HR/NDCG/MRR@K trio; LSApp's `eval.metrics` (karar-notu SS3 exception,
    e.g. HR@1/HR@3/MRR@5) is used as-is."""
    if "metrics" in config["eval"]:
        return [(m["name"], m["k"]) for m in config["eval"]["metrics"]]
    k = config["eval"]["k"]
    return [("hr", k), ("ndcg", k), ("mrr", k)]


def _objective_metric_key(config: dict) -> str:
    """The metrics-dict key Optuna maximizes. Defaults to ndcg@{k} (RetailRocket);
    LSApp configs set optuna.objective_metric explicitly since they have no
    ndcg@{k} entry in eval.metrics."""
    objective = config.get("optuna", {}).get("objective_metric")
    if objective:
        return objective
    return f"ndcg@{config['eval']['k']}"


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _build_model(
    config: dict, params: dict, seed: int, checkpoint_dir: str | None, verbose: bool = False
) -> LstmModel:
    t = config["training"]
    return LstmModel(
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        batch_size=t["batch_size"],
        max_seq_len=t["max_seq_len"],
        max_epochs=t["max_epochs"],
        patience=t["patience"],
        seed=seed,
        device=t.get("device"),
        checkpoint_dir=checkpoint_dir,
        num_workers=t.get("num_workers", 0),
        verbose=verbose,
    )


def _evaluate_fast(model: LstmModel, split_df: pd.DataFrame, specs: list[tuple[str, int]]) -> dict[str, float]:
    """Same metrics as src.eval.harness.evaluate_model_multi (identical
    iter_eval_pairs/average_multi), but drives predict_many() instead of calling
    model.predict() once per session-prefix -- collapses tens of thousands of
    unbatched single-session GPU forward passes into a handful of batches."""
    contexts, true_items = zip(*iter_eval_pairs(split_df))
    ranked_lists = model.predict_many(list(contexts))
    return average_multi(zip(ranked_lists, true_items), specs)


def _sample_val_sessions(val: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_sessions = val["session_id"].unique()
    n = min(n, len(all_sessions))
    chosen = rng.choice(all_sessions, size=n, replace=False)
    return val[val["session_id"].isin(chosen)]


def _tune(
    config: dict, train: pd.DataFrame, val_search: pd.DataFrame, specs: list[tuple[str, int]], objective_key: str
) -> tuple[dict, float]:
    space = config["optuna"]["search_space"]
    seed = config["training"]["random_seed"]
    n_trials = config["optuna"]["n_trials"]
    n_jobs = config["optuna"].get("n_jobs", 1)

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.monotonic()
        params = {
            "hidden_size": trial.suggest_int("hidden_size", space["hidden_size"]["low"], space["hidden_size"]["high"]),
            "num_layers": trial.suggest_int("num_layers", space["num_layers"]["low"], space["num_layers"]["high"]),
            "dropout": trial.suggest_float("dropout", space["dropout"]["low"], space["dropout"]["high"]),
            "learning_rate": trial.suggest_float(
                "learning_rate", space["learning_rate"]["low"], space["learning_rate"]["high"], log=True
            ),
        }
        model = _build_model(config, params, seed, checkpoint_dir=None, verbose=False)
        model.fit(train, val=val_search)
        score = _evaluate_fast(model, val_search, specs)[objective_key]
        duration = time.monotonic() - trial_start
        print(f"Trial {trial.number + 1}/{n_trials} bitti, {objective_key}={score:.4f}, süre={duration:.1f}s", flush=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    return study.best_params, study.best_value


def _train_one_seed(
    config: dict, train: pd.DataFrame, val: pd.DataFrame, best_params: dict, seed: int, specs: list[tuple[str, int]]
) -> dict:
    checkpoint_dir = None
    if config.get("checkpoint_dir"):
        checkpoint_dir = str(Path(config["checkpoint_dir"]) / f"seed_{seed}")
    model = _build_model(config, best_params, seed, checkpoint_dir, verbose=True)

    fit_start = time.monotonic()
    model.fit(train, val=val)
    fit_duration = time.monotonic() - fit_start

    eval_start = time.monotonic()
    metrics = _evaluate_fast(model, val, specs)
    eval_duration = time.monotonic() - eval_start

    return {
        "seed": seed,
        "metrics": metrics,
        "fit_duration_seconds": round(fit_duration, 3),
        "eval_duration_seconds": round(eval_duration, 3),
        "n_epochs_trained": model.history[-1]["epoch"] if model.history else 0,
    }


def _aggregate(per_seed: list[dict], specs: list[tuple[str, int]]) -> dict:
    keys = [f"{name}@{k}" for name, k in specs]
    out = {}
    for key in keys:
        values = np.array([run["metrics"][key] for run in per_seed])
        out[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune/train/evaluate LSTM, log to results/runs.jsonl")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to model YAML config")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    train = pd.read_parquet(config["data"]["train_path"])
    val = pd.read_parquet(config["data"]["val_path"])
    specs = _eval_specs(config)
    objective_key = _objective_metric_key(config)

    val_search = _sample_val_sessions(
        val, config["optuna"]["search_val_sample_sessions"], config["training"]["random_seed"]
    )

    print(f"=== Tuning LSTM ({config['optuna']['n_trials']} trials) ===")
    tune_start = time.monotonic()
    best_params, best_search_score = _tune(config, train, val_search, specs, objective_key)
    tune_duration = round(time.monotonic() - tune_start, 3)
    print(f"  best params: {best_params}  (search {objective_key}={best_search_score:.4f}, {tune_duration:.1f}s)")

    per_seed = []
    for seed in config["training"]["seeds"]:
        print(f"=== Final training, seed={seed} ===")
        run = _train_one_seed(config, train, val, best_params, seed, specs)
        per_seed.append(run)
        print(f"  seed {seed}: {run['metrics']}  (epochs={run['n_epochs_trained']})")

    summary_metrics = _aggregate(per_seed, specs)
    print(f"=== Mean±std over {len(per_seed)} seeds ===")
    print(json.dumps(summary_metrics, indent=2))

    n_eval_pairs = count_eval_pairs(val)
    commit = _git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    base_config = {key: value for key, value in config.items() if key != "optuna"}
    base_config["optuna_best_params"] = best_params

    os.makedirs("results", exist_ok=True)
    with open("results/runs.jsonl", "a") as f:
        for run in per_seed:
            f.write(
                json.dumps(
                    {
                        "timestamp": timestamp,
                        "model": "lstm",
                        "dataset": config["dataset"],
                        "split": "val",
                        "run_type": "seed",
                        "seed": run["seed"],
                        "config": base_config,
                        "metrics": run["metrics"],
                        "num_eval_pairs": n_eval_pairs,
                        "fit_duration_seconds": run["fit_duration_seconds"],
                        "eval_duration_seconds": run["eval_duration_seconds"],
                        "tune_duration_seconds": tune_duration,
                        "duration_seconds": round(
                            tune_duration + run["fit_duration_seconds"] + run["eval_duration_seconds"], 3
                        ),
                        "git_commit": commit,
                    }
                )
                + "\n"
            )
        f.write(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "model": "lstm",
                    "dataset": config["dataset"],
                    "split": "val",
                    "run_type": "summary",
                    "seed": None,
                    "seeds": config["training"]["seeds"],
                    "config": base_config,
                    "metrics": summary_metrics,
                    "num_eval_pairs": n_eval_pairs,
                    "fit_duration_seconds": round(sum(r["fit_duration_seconds"] for r in per_seed), 3),
                    "eval_duration_seconds": round(sum(r["eval_duration_seconds"] for r in per_seed), 3),
                    "tune_duration_seconds": tune_duration,
                    "duration_seconds": round(
                        tune_duration + sum(r["fit_duration_seconds"] + r["eval_duration_seconds"] for r in per_seed),
                        3,
                    ),
                    "git_commit": commit,
                }
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
