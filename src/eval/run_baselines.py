"""Fit POP/S-POP/Item-KNN on train, evaluate on val, log to results/runs.jsonl.

CLI usage:
  python -m src.eval.run_baselines
  python -m src.eval.run_baselines --configs configs/pop_lsapp.yaml configs/s_pop_lsapp.yaml configs/item_knn_lsapp.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.eval.harness import count_eval_pairs, evaluate_model, evaluate_model_multi
from src.models.item_knn import ItemKnnModel
from src.models.pop import PopModel
from src.models.s_pop import SPopModel

MODEL_CONFIGS = ["configs/pop.yaml", "configs/s_pop.yaml", "configs/item_knn.yaml"]

MODEL_CLASSES = {
    "pop": PopModel,
    "s_pop": SPopModel,
    "item_knn": ItemKnnModel,
}


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


def _build_model(config: dict):
    name = config["model"]
    cls = MODEL_CLASSES[name]
    if name == "item_knn":
        return cls(neighborhood_size=config["hyperparameters"]["neighborhood_size"])
    return cls()


def run_one(config_path: str) -> dict:
    config = yaml.safe_load(Path(config_path).read_text())

    train = pd.read_parquet(config["data"]["train_path"])
    val = pd.read_parquet(config["data"]["val_path"])

    model = _build_model(config)

    fit_start = time.monotonic()
    model.fit(train)
    fit_duration = time.monotonic() - fit_start

    eval_start = time.monotonic()
    if "metrics" in config["eval"]:
        specs = [(spec["name"], spec["k"]) for spec in config["eval"]["metrics"]]
        metrics = evaluate_model_multi(model, val, specs)
    else:
        metrics = evaluate_model(model, val, k=config["eval"]["k"])
    eval_duration = time.monotonic() - eval_start

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config["model"],
        "dataset": config["dataset"],
        "split": "val",
        "seed": None,
        "config": config,
        "metrics": metrics,
        "num_eval_pairs": count_eval_pairs(val),
        "fit_duration_seconds": round(fit_duration, 3),
        "eval_duration_seconds": round(eval_duration, 3),
        "duration_seconds": round(fit_duration + eval_duration, 3),
        "git_commit": _git_commit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit/evaluate POP, S-POP, Item-KNN, log to results/runs.jsonl")
    parser.add_argument(
        "--configs", nargs="+", default=MODEL_CONFIGS, help="YAML config paths to run (default: RetailRocket's 3)"
    )
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)
    runs_path = Path("results/runs.jsonl")

    with runs_path.open("a") as f:
        for config_path in args.configs:
            print(f"Running {config_path} ...")
            run = run_one(config_path)
            print(f"  metrics: {run['metrics']}  (fit={run['fit_duration_seconds']}s, eval={run['eval_duration_seconds']}s)")
            f.write(json.dumps(run) + "\n")


if __name__ == "__main__":
    main()
