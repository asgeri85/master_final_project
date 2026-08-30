"""H1: sequential deep models (GRU4Rec, LSTM, SASRec) vs. classical baselines
(Item-KNN, S-POP), each at seed=42, karar-notu SS6 (locked H1 test plan).

For every (deep, classical) pair, tests HR@10 and NDCG@10 with a session-level
paired Wilcoxon signed-rank test (zero_method="wilcox" as the official statistic,
cross-checked against "pratt"/"zsplit" the way diagnose_h2_wilcoxon.py does for
H2), then applies Holm-Bonferroni across all 6 pairs x 2 metrics = 12 tests.

Deep models are refit at seed=42 using the Optuna-selected hyperparameters already
logged in results/runs.jsonl (no new search) purely to recover per-session metrics,
which no run_*.py script persists -- only averaged metrics are logged there.
Item-KNN/S-POP are deterministic and refit directly from their configs/*.yaml.

Also reports a seed-averaging sensitivity check (5-seed mean vs. seed=42, read
directly from the already-logged run_type=="summary" rows -- no retraining) so the
H1 test's reliance on a single representative seed can be sanity-checked without
turning it into "pick the best seed" p-hacking.

CLI usage: python -m src.eval.run_h1_hypothesis
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import wilcoxon

from src.eval.harness import evaluate_per_pair_and_session
from src.eval.stats import holm_bonferroni
from src.models.gru4rec import Gru4RecModel
from src.models.item_knn import ItemKnnModel
from src.models.lstm import LstmModel
from src.models.s_pop import SPopModel
from src.models.sasrec import SasrecModel

RUNS_PATH = Path("results/runs.jsonl")
DATASET = "retailrocket"
SEED = 42
K = 10

DEEP_MODELS = ["gru4rec", "lstm", "sasrec"]
CLASSICAL_CONFIGS = {"item_knn": "configs/item_knn.yaml", "s_pop": "configs/s_pop.yaml"}
PAIRS = [(deep, classical) for deep in DEEP_MODELS for classical in ("item_knn", "s_pop")]

DEEP_MODEL_CLASSES = {"gru4rec": Gru4RecModel, "lstm": LstmModel, "sasrec": SasrecModel}

REPRODUCIBILITY_TOLERANCE = 0.01  # absolute HR@10/NDCG@10 tolerance vs. the logged run


def _load_runs() -> list[dict]:
    with RUNS_PATH.open() as f:
        return [json.loads(line) for line in f]


def _find_run(runs: list[dict], model: str, *, run_type: str | None, seed: int | None) -> dict:
    matches = [
        r
        for r in runs
        if r["model"] == model and r.get("dataset") == DATASET and r.get("run_type") == run_type and r.get("seed") == seed
    ]
    if not matches:
        raise ValueError(f"No runs.jsonl entry found for model={model} run_type={run_type} seed={seed}")
    if len(matches) > 1:
        print(f"  [uyarı] {model} run_type={run_type} seed={seed} için birden fazla kayıt bulundu, sonuncusu kullanılıyor.")
    return matches[-1]


def _build_deep_model(model_name: str, config: dict):
    t = config["training"]
    params = config["optuna_best_params"]
    cls = DEEP_MODEL_CLASSES[model_name]
    kwargs = dict(
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        batch_size=t["batch_size"],
        max_seq_len=t["max_seq_len"],
        max_epochs=t["max_epochs"],
        patience=t["patience"],
        seed=SEED,
        device=t.get("device"),
        checkpoint_dir=None,  # diagnostic refit only -- don't touch/overwrite original checkpoints
        num_workers=t.get("num_workers", 0),
        verbose=True,
    )
    if model_name == "sasrec":
        kwargs["num_heads"] = params["num_heads"]
        kwargs["num_blocks"] = params["num_blocks"]
    else:
        kwargs["num_layers"] = params["num_layers"]
    return cls(**kwargs)


def _fit_deep_model(model_name: str, runs: list[dict], train: pd.DataFrame, val: pd.DataFrame) -> tuple:
    seed_run = _find_run(runs, model_name, run_type="seed", seed=SEED)
    model = _build_deep_model(model_name, seed_run["config"])
    model.fit(train, val=val)
    return model, seed_run


def _fit_classical_model(model_name: str):
    config = yaml.safe_load(Path(CLASSICAL_CONFIGS[model_name]).read_text())
    cls = ItemKnnModel if model_name == "item_knn" else SPopModel
    model = cls(neighborhood_size=config["hyperparameters"]["neighborhood_size"]) if model_name == "item_knn" else cls()
    train = pd.read_parquet(config["data"]["train_path"])
    model.fit(train)
    return model


def _check_reproducibility(model_name: str, per_pair: pd.DataFrame, logged_metrics: dict) -> dict:
    """Unweighted mean over every (session, prefix) row -- exactly how run_*.py's
    average_at_k/evaluate_model compute the metric that gets logged to runs.jsonl.
    A large deviation here (unlike the expected pair-vs-session weighting gap
    between this and the per-session-grouped mean used for Wilcoxon) means the
    refit didn't actually reproduce the original run -- config drift or nondeterminism."""
    prefix_weighted = {
        f"hr@{K}": float(per_pair[f"hr@{K}"].mean()),
        f"ndcg@{K}": float(per_pair[f"ndcg@{K}"].mean()),
    }
    deltas = {
        metric: abs(prefix_weighted[metric] - logged_metrics[metric]) for metric in (f"hr@{K}", f"ndcg@{K}")
    }
    flagged = {metric: delta for metric, delta in deltas.items() if delta > REPRODUCIBILITY_TOLERANCE}
    if flagged:
        print(f"  [UYARI] {model_name}: prefix-ağırlıklı ortalama, kayıtlı runs.jsonl metriğinden "
              f"tolerans dışı sapıyor: {flagged}. Bug şüphesi -- devam etmeden önce kontrol edin.")
    return {"prefix_weighted_mean": prefix_weighted, "logged": logged_metrics, "abs_delta": deltas}


def _wilcoxon_cross_check(deep_vals: pd.Series, classical_vals: pd.Series) -> dict:
    results = {}
    for zero_method in ("wilcox", "pratt", "zsplit"):
        stat, p = wilcoxon(deep_vals, classical_vals, zero_method=zero_method)
        results[zero_method] = {"statistic": float(stat), "p_value": float(p)}
    return results


def build_all_per_session_metrics() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    runs = _load_runs()
    base_config = yaml.safe_load(Path(CLASSICAL_CONFIGS["item_knn"]).read_text())
    train = pd.read_parquet(base_config["data"]["train_path"])
    val = pd.read_parquet(base_config["data"]["val_path"])

    per_session: dict[str, pd.DataFrame] = {}
    reproducibility: dict[str, dict] = {}

    for model_name in ("item_knn", "s_pop"):
        print(f"=== {model_name}: fit + oturum bazlı değerlendirme ===")
        model = _fit_classical_model(model_name)
        per_pair, ps = evaluate_per_pair_and_session(model, val, k=K, batched=False)
        logged = _find_run(runs, model_name, run_type=None, seed=None)["metrics"]
        reproducibility[model_name] = _check_reproducibility(model_name, per_pair, logged)
        per_session[model_name] = ps
        ps.to_parquet(f"results/h1_per_session_{model_name}.parquet")

    for model_name in DEEP_MODELS:
        print(f"=== {model_name}: seed={SEED} ile yeniden fit + oturum bazlı değerlendirme ===")
        model, seed_run = _fit_deep_model(model_name, runs, train, val)
        per_pair, ps = evaluate_per_pair_and_session(model, val, k=K, batched=True)
        reproducibility[model_name] = _check_reproducibility(model_name, per_pair, seed_run["metrics"])
        per_session[model_name] = ps
        ps.to_parquet(f"results/h1_per_session_{model_name}.parquet")

    return per_session, reproducibility


def run_pairwise_tests(per_session: dict[str, pd.DataFrame]) -> tuple[list[dict], pd.DataFrame]:
    metrics = [f"hr@{K}", f"ndcg@{K}"]
    test_rows = []
    diff_columns = {}

    for deep, classical in PAIRS:
        deep_sessions = per_session[deep]
        classical_sessions = per_session[classical]
        common = deep_sessions.index.intersection(classical_sessions.index)
        deep_a = deep_sessions.loc[common]
        classical_a = classical_sessions.loc[common]

        for metric in metrics:
            diff = deep_a[metric] - classical_a[metric]
            diff_columns[f"{deep}_vs_{classical}_{metric}_diff"] = diff

            cross_check = _wilcoxon_cross_check(deep_a[metric], classical_a[metric])
            official = cross_check["wilcox"]

            test_rows.append(
                {
                    "deep": deep,
                    "classical": classical,
                    "metric": metric,
                    "n_common_sessions": int(len(common)),
                    "n_zero": int((diff == 0).sum()),
                    "n_positive": int((diff > 0).sum()),
                    "n_negative": int((diff < 0).sum()),
                    "statistic": official["statistic"],
                    "p_value": official["p_value"],
                    "mean_diff_deep_minus_classical": float(diff.mean()),
                    "zero_method_cross_check": cross_check,
                }
            )

    diffs_df = pd.DataFrame(diff_columns)
    return test_rows, diffs_df


def apply_holm_bonferroni(test_rows: list[dict]) -> list[dict]:
    pvalues = [row["p_value"] for row in test_rows]
    adjusted = holm_bonferroni(pvalues)
    for row, p_holm in zip(test_rows, adjusted):
        row["p_value_holm"] = p_holm
        row["significant_alpha_0.05"] = p_holm < 0.05
    return test_rows


def sensitivity_analysis() -> list[dict]:
    """5-seed mean vs. seed=42, read from already-logged runs.jsonl rows -- no retraining."""
    runs = _load_runs()
    rows = []
    for model_name in DEEP_MODELS:
        seed_run = _find_run(runs, model_name, run_type="seed", seed=SEED)
        summary_run = _find_run(runs, model_name, run_type="summary", seed=None)
        for metric in (f"hr@{K}", f"ndcg@{K}"):
            seed42_value = seed_run["metrics"][metric]
            mean_5seed = summary_run["metrics"][metric]["mean"]
            std_5seed = summary_run["metrics"][metric]["std"]
            rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "seed42_value": seed42_value,
                    "mean_5seed": mean_5seed,
                    "std_5seed": std_5seed,
                    "abs_diff": abs(seed42_value - mean_5seed),
                    "relative_diff_pct": abs(seed42_value - mean_5seed) / mean_5seed * 100 if mean_5seed else float("nan"),
                }
            )
    return rows


def _fmt(x: float) -> str:
    return f"{x:.6g}"


def render_markdown_report(test_rows: list[dict], reproducibility: dict[str, dict], sensitivity_rows: list[dict]) -> str:
    lines = ["# H1 Hipotez Testi: Katman 3-4 (Derin) vs Katman 1-2 (Klasik)", ""]
    lines.append(
        f"Oturum bazlı eşleştirilmiş Wilcoxon signed-rank testi (zero_method=\"wilcox\"), "
        f"6 çift x 2 metrik = {len(test_rows)} test, Holm-Bonferroni düzeltmesi (RetailRocket val, seed={SEED})."
    )
    lines.append("")

    lines.append("## Ana sonuç tablosu")
    lines.append("")
    lines.append(
        "| Çift | Metrik | n_ortak_oturum | p-değeri | Holm-düzeltilmiş p | Anlamlı mı (α=0.05) | Fark yönü | Fark büyüklüğü (deep - classical) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    for row in test_rows:
        direction = "derin > klasik" if row["mean_diff_deep_minus_classical"] > 0 else "derin < klasik" if row["mean_diff_deep_minus_classical"] < 0 else "fark yok"
        lines.append(
            f"| {row['deep']} vs {row['classical']} | {row['metric']} | {row['n_common_sessions']} | "
            f"{_fmt(row['p_value'])} | {_fmt(row['p_value_holm'])} | {'Evet' if row['significant_alpha_0.05'] else 'Hayır'} | "
            f"{direction} | {_fmt(row['mean_diff_deep_minus_classical'])} |"
        )
    lines.append("")

    lines.append("## Sıfır-fark oranı ve asimetri")
    lines.append("")
    lines.append("| Çift | Metrik | n_ortak_oturum | n_sıfır | n_pozitif (deep>classical) | n_negatif (deep<classical) |")
    lines.append("|---|---|---|---|---|---|")
    for row in test_rows:
        lines.append(
            f"| {row['deep']} vs {row['classical']} | {row['metric']} | {row['n_common_sessions']} | "
            f"{row['n_zero']} | {row['n_positive']} | {row['n_negative']} |"
        )
    lines.append("")

    lines.append("## zero_method çapraz kontrolü (wilcox / pratt / zsplit)")
    lines.append("")
    lines.append("| Çift | Metrik | wilcox p | pratt p | zsplit p | Tutarlı mı |")
    lines.append("|---|---|---|---|---|---|")
    for row in test_rows:
        cc = row["zero_method_cross_check"]
        ps = [cc[m]["p_value"] for m in ("wilcox", "pratt", "zsplit")]
        consistent = (max(ps) < 0.05) or (min(ps) >= 0.05)
        lines.append(
            f"| {row['deep']} vs {row['classical']} | {row['metric']} | {_fmt(ps[0])} | {_fmt(ps[1])} | {_fmt(ps[2])} | "
            f"{'Evet' if consistent else 'HAYIR — anlamlılık kararı yönteme bağlı, dikkat'} |"
        )
    lines.append("")

    lines.append("## Yeniden üretim (reproducibility) kontrolü")
    lines.append("")
    lines.append(
        f"Prefix-ağırlıklı ortalama (orijinal `run_*.py` scriptlerinin `average_at_k` ile "
        f"hesapladığı ve `results/runs.jsonl`'e kaydettiği yöntemin birebir aynısı) ile kayıtlı "
        f"değer arasındaki mutlak fark; tolerans={REPRODUCIBILITY_TOLERANCE}. (Bu, Wilcoxon testinde "
        f"kullanılan session-gruplu ortalamadan farklıdır -- session uzunluğu metrikle ilişkiliyse "
        f"o ikisi arasında beklenen, hataya işaret etmeyen bir fark oluşur; ayrı olarak izlenmez.)"
    )
    lines.append("")
    lines.append("| Model | Metrik | Prefix-ağırlıklı ortalama (yeniden fit) | Kayıtlı (runs.jsonl) | Mutlak fark |")
    lines.append("|---|---|---|---|---|")
    for model_name, repro in reproducibility.items():
        for metric in (f"hr@{K}", f"ndcg@{K}"):
            lines.append(
                f"| {model_name} | {metric} | {_fmt(repro['prefix_weighted_mean'][metric])} | "
                f"{_fmt(repro['logged'][metric])} | {_fmt(repro['abs_delta'][metric])} |"
            )
    lines.append("")

    lines.append("## Duyarlılık analizi: 5-seed ortalaması vs seed=42 (H1 testi tekrarlanmadı)")
    lines.append("")
    lines.append("| Model | Metrik | seed=42 değeri | 5-seed ortalama±std | Mutlak fark | Oransal fark (%) |")
    lines.append("|---|---|---|---|---|---|")
    for row in sensitivity_rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | {_fmt(row['seed42_value'])} | "
            f"{_fmt(row['mean_5seed'])}±{_fmt(row['std_5seed'])} | {_fmt(row['abs_diff'])} | "
            f"{_fmt(row['relative_diff_pct'])} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    Path("results").mkdir(exist_ok=True)

    per_session, reproducibility = build_all_per_session_metrics()
    test_rows, diffs_df = run_pairwise_tests(per_session)
    test_rows = apply_holm_bonferroni(test_rows)
    diffs_df.to_parquet("results/h1_session_diffs.parquet")

    sensitivity_rows = sensitivity_analysis()

    report = render_markdown_report(test_rows, reproducibility, sensitivity_rows)
    Path("results/h1_hypothesis_report.md").write_text(report)

    print("\n" + report)
    print("\nKaydedildi: results/h1_per_session_*.parquet, results/h1_session_diffs.parquet, "
          "results/h1_hypothesis_report.md")


if __name__ == "__main__":
    main()
