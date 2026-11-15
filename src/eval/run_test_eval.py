"""A1: final held-out TEST-set evaluation (karar-notu SS11 locked protocol, last step).

No training of deep models and no hyperparameter search:
- POP / S-POP / Item-KNN: refit on train from configs/*.yaml (deterministic).
- XGBoost raw/rich (RetailRocket): refit on train with the Optuna-selected params
  already logged in results/runs.jsonl (deterministic, karar-notu H2 refit check).
- GRU4Rec / LSTM / SASRec: loaded from the saved best-on-val checkpoints
  ({ckpt_root}/{model}[_lsapp]/seed_{s}/{model}_seed{s}_best.pt). Each checkpoint
  is first re-scored on VAL and compared with its logged runs.jsonl row, so a wrong
  or stale checkpoint is caught before any test number is written.

Then repeats the H1 test (seed=42, 12 Wilcoxon tests, Holm) and the H2 test on
the test split (RetailRocket only), appends split="test" rows to results/runs.jsonl
and writes results/test_set_report.md (or test_set_report_lsapp.md).

CLI usage:
  python -m src.eval.run_test_eval --dataset retailrocket --ckpt-root checkpoints
  python -m src.eval.run_test_eval --dataset lsapp --ckpt-root /content/drive/MyDrive/thesis-checkpoints
  add --dry-run to print results without touching results/.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import wilcoxon

from src.eval.harness import count_eval_pairs, evaluate_model_multi, evaluate_per_pair_and_session, iter_eval_pairs
from src.eval.metrics import average_multi
from src.eval.run_baselines import _build_model as build_baseline, _git_commit
from src.eval.run_h1_hypothesis import apply_holm_bonferroni, run_pairwise_tests
from src.eval.run_h2_ablation import _per_session_metrics as xgb_per_session
from src.eval.stats import holm_bonferroni
from src.features.build_candidate_table import build_candidate_table
from src.features.feature_builder import FeatureBuilder
from src.models.item_knn import ItemKnnModel
from src.models.pop import PopModel
from src.models.xgboost_ranker import XGBoostRankerModel
from src.ondevice.export_onnx import build_net, get_vocab, load_checkpoint

RUNS_PATH = Path("results/runs.jsonl")
DEEP_MODELS = ["gru4rec", "lstm", "sasrec"]
SEEDS = [42, 43, 44, 45, 46]
H1_SEED = 42
VAL_TOLERANCE = 0.01  # same tolerance as run_h1_hypothesis.REPRODUCIBILITY_TOLERANCE

DATASETS = {
    "retailrocket": {"suffix": "", "specs": [("hr", 10), ("ndcg", 10), ("mrr", 10)], "report": "results/test_set_report.md"},
    "lsapp": {"suffix": "_lsapp", "specs": [("hr", 1), ("hr", 3), ("mrr", 5)], "report": "results/test_set_report_lsapp.md"},
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TopKRecommender:
    """Checkpoint net -> ranked top-K lists, batched on DEVICE.

    Same scoring as Gru4RecModel/LstmModel/SasrecModel.predict_many (same encode/project,
    cold-start fallback to POP, tie-break score desc / item id asc), but only the first K
    positions of the ranking are materialised. HR/NDCG/MRR@K for K <= self.k depend on
    nothing else, so metrics are unchanged; it avoids building a full-catalog Python list
    per prefix (the original path took ~350 s per model on an A100 and gigabytes of RAM).
    """

    def __init__(self, net, item_to_idx, idx_to_item, max_seq_len, pop, k):
        self.net = net.to(DEVICE).eval()
        self.item_to_idx = item_to_idx
        self.idx_to_item = np.asarray(idx_to_item)
        self.max_seq_len = max_seq_len
        self._pop = pop
        self.k = k

    def predict_many(self, sessions, eval_batch_size: int = 512):
        results = [None] * len(sessions)
        idx, known_all = [], []
        for i, session in enumerate(sessions):
            known = [it for it in session if it in self.item_to_idx]
            if not known:
                results[i] = self._pop.predict(session)[: self.k]
                continue
            idx.append(i)
            known_all.append(known[-self.max_seq_len:])
        with torch.no_grad():
            for start in range(0, len(known_all), eval_batch_size):
                chunk = known_all[start:start + eval_batch_size]
                lengths = torch.tensor([len(c) for c in chunk], dtype=torch.long)  # CPU (pack_padded_sequence)
                x = torch.zeros((len(chunk), int(lengths.max())), dtype=torch.long)
                for r, c in enumerate(chunk):
                    x[r, :len(c)] = torch.tensor([self.item_to_idx[it] for it in c], dtype=torch.long)
                x = x.to(DEVICE)
                hidden = self.net.encode(x, lengths)
                last = hidden[torch.arange(len(chunk), device=DEVICE), lengths.to(DEVICE) - 1]
                scores = self.net.project(last)[:, 1:]
                top_s, top_i = torch.topk(scores, self.k, dim=1)
                top_s, top_i = top_s.cpu().numpy(), top_i.cpu().numpy()
                for r, i in enumerate(idx[start:start + eval_batch_size]):
                    items = self.idx_to_item[top_i[r]]
                    results[i] = items[np.lexsort((items, -top_s[r]))].tolist()
        return results


def _load_runs() -> list[dict]:
    with RUNS_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _logged_val(runs, dataset, model, run_type=None, seed=None) -> dict:
    rows = [r for r in runs if r["dataset"] == dataset and r["model"] == model and r.get("split") == "val"
            and r.get("run_type") == run_type and r.get("seed") == seed]
    if not rows:
        raise ValueError(f"no logged val row: {dataset} {model} run_type={run_type} seed={seed}")
    return rows[-1]


def _metrics(per_pair: pd.DataFrame, specs) -> dict[str, float]:
    """Prefix-weighted mean, identical to average_at_k / what runs.jsonl stores."""
    return {f"{n}@{k}": float(per_pair[f"{n}@{k}"].mean()) for n, k in specs}


def _eval(model, split_df, specs, dataset, batched):
    """RetailRocket: per-pair + per-session (needed for H1). LSApp: averaged multi-K only."""
    if dataset == "retailrocket":
        per_pair, per_session = evaluate_per_pair_and_session(model, split_df, k=10, batched=batched)
        return _metrics(per_pair, specs), per_session
    if batched:
        contexts, true_items = zip(*iter_eval_pairs(split_df))
        return average_multi(zip(model.predict_many(list(contexts)), true_items), specs), None
    return evaluate_model_multi(model, split_df, specs), None


def _row(model, dataset, metrics, n_pairs, config, seed=None, run_type=None, extra=None) -> dict:
    row = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": model, "dataset": dataset,
           "split": "test", "seed": seed, "config": config, "metrics": metrics, "num_eval_pairs": n_pairs,
           "git_commit": _git_commit()}
    if run_type is not None:
        row["run_type"] = run_type
    if extra:
        row.update(extra)
    return row


def _recover_best_from_epochs(ckpt_dir: Path, name: str, seed: int):
    """best.pt is empty/unreadable (Drive write failure) -> rebuild it from the per-epoch files.

    _save_checkpoint stores `best_val_ndcg` (running best, updated BEFORE the save), so it is
    non-decreasing over epochs and the best epoch is the first one whose value equals the
    final maximum; that epoch's file holds exactly the best-on-val weights. Binary search
    over the epoch files; the caller still re-scores the result on val against runs.jsonl.
    """
    files = sorted(ckpt_dir.glob(f"{name}_seed{seed}_epoch*.pt"))
    files = [f for f in files if f.stat().st_size > 0]
    if not files:
        return None, None
    final_best = load_checkpoint(files[-1])["best_val_ndcg"]
    lo, hi = 0, len(files) - 1  # invariant: files[hi] reaches final_best
    while lo < hi:
        mid = (lo + hi) // 2
        if load_checkpoint(files[mid])["best_val_ndcg"] >= final_best:
            hi = mid
        else:
            lo = mid + 1
    payload = load_checkpoint(files[lo])
    print(f"  [kurtarma] {name} seed={seed}: best.pt boş; en iyi epoch dosyası {files[lo].name} "
          f"(best_val_ndcg={payload['best_val_ndcg']:.6f}) kullanılıyor", flush=True)
    return payload, files[lo].name


def run_classical(dataset, train, test, specs, runs):
    rows, per_session = [], {}
    for name in ("pop", "s_pop", "item_knn"):
        config = yaml.safe_load(Path(f"configs/{name}{DATASETS[dataset]['suffix']}.yaml").read_text())
        model = build_baseline(config)
        model.fit(train)
        metrics, ps = _eval(model, test, specs, dataset, batched=False)
        print(f"  {name}: {metrics}")
        rows.append(_row(name, dataset, metrics, count_eval_pairs(test), config))
        if ps is not None:
            per_session[name] = ps
    return rows, per_session


def run_deep(dataset, train, val, test, specs, runs, ckpt_root):
    pop = PopModel()
    pop.fit(train)
    rows, per_session, checks = [], {}, []
    for name in DEEP_MODELS:
        per_seed = []
        for seed in SEEDS:
            path = Path(ckpt_root) / f"{name}{DATASETS[dataset]['suffix']}" / f"seed_{seed}" / f"{name}_seed{seed}_best.pt"
            payload, source = None, "best_on_val"
            if path.exists() and path.stat().st_size > 0:
                try:
                    payload = load_checkpoint(path)
                except (EOFError, RuntimeError, OSError) as e:
                    print(f"  [uyarı] {path.name} okunamadı ({type(e).__name__})", flush=True)
            if payload is None:
                payload, epoch_file = _recover_best_from_epochs(path.parent, name, seed)
                if payload is None:
                    print(f"  [atlandı] {path} yok/boş ve epoch dosyası bulunamadı", flush=True)
                    continue
                source = f"recovered_from_{epoch_file}"
            t0 = time.monotonic()
            item_to_idx, idx_to_item = get_vocab(payload)
            net = build_net(name, payload, len(idx_to_item))
            model = _TopKRecommender(net, item_to_idx, idx_to_item, payload["hyperparameters"]["max_seq_len"], pop,
                                     k=max(k for _, k in specs))

            logged = _logged_val(runs, dataset, name, run_type="seed", seed=seed)["metrics"]
            val_metrics, _ = _eval(model, val, specs, dataset, batched=True)
            deltas = {m: abs(val_metrics[m] - logged[m]) for m in val_metrics}
            ok = max(deltas.values()) <= (VAL_TOLERANCE if source == "best_on_val" else 1e-3)
            checks.append({"model": name, "seed": seed, "max_abs_delta_vs_logged_val": max(deltas.values()), "ok": ok, "source": source})
            print(f"  {name} seed={seed}: val kontrolü max|Δ|={max(deltas.values()):.2e} {'OK' if ok else 'BAŞARISIZ'}", flush=True)
            if not ok:
                raise RuntimeError(f"{path}: checkpoint kayıtlı val sonucunu üretmiyor ({deltas}); test yazılmadı.")

            metrics, ps = _eval(model, test, specs, dataset, batched=True)
            print(f"  {name} seed={seed} test: {metrics}  ({time.monotonic() - t0:.0f}s)", flush=True)
            per_seed.append({"seed": seed, "metrics": metrics, "source": source})
            if seed == H1_SEED and ps is not None:
                per_session[name] = ps

        if not per_seed:
            continue
        config = _logged_val(runs, dataset, name, run_type="seed", seed=per_seed[0]["seed"])["config"]
        n_pairs = count_eval_pairs(test)
        for r in per_seed:
            rows.append(_row(name, dataset, r["metrics"], n_pairs, config, seed=r["seed"], run_type="seed",
                             extra={"checkpoint_source": r["source"]}))
        if len(per_seed) > 1:
            summary = {}
            for key in per_seed[0]["metrics"]:
                values = np.array([r["metrics"][key] for r in per_seed])
                summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
            rows.append(_row(name, dataset, summary, n_pairs, config, run_type="summary",
                             extra={"seeds": [r["seed"] for r in per_seed]}))
    return rows, per_session, checks


def run_xgboost(train, test, runs):
    """H2 setup on test: same candidate pool, same logged best params, session-level unit."""
    base = _logged_val(runs, "retailrocket", "xgboost_raw")["config"]
    item_knn = ItemKnnModel(neighborhood_size=base["candidate_generation"]["item_knn_neighborhood_size"])
    item_knn.fit(train)
    feature_builder = FeatureBuilder()
    feature_builder.fit(train)
    table = build_candidate_table(item_knn, feature_builder, train,
                                  n_candidates=base["candidate_generation"]["n_candidates"], include_label=True,
                                  negatives_per_positive=base["training"]["negatives_per_positive"],
                                  random_seed=base["training"]["random_seed"])
    rows, per_session = [], {}
    for fs in ("raw", "rich"):
        config = _logged_val(runs, "retailrocket", f"xgboost_{fs}")["config"]
        model = XGBoostRankerModel(item_knn=item_knn, feature_builder=feature_builder, feature_set=fs,
                                   xgb_params=config["optuna_best_params"],
                                   n_candidates=config["candidate_generation"]["n_candidates"],
                                   negatives_per_positive=config["training"]["negatives_per_positive"],
                                   random_seed=config["training"]["random_seed"])
        model.fit_from_table(table)
        ps = xgb_per_session(lambda items, ts, m=model: m.predict_from_context(items, ts), test, 10)
        metrics = {c: float(ps[c].mean()) for c in ("hr@10", "ndcg@10", "mrr@10")}
        print(f"  xgboost_{fs}: {metrics}")
        rows.append(_row(f"xgboost_{fs}", "retailrocket", metrics, int(ps.shape[0]), config))
        per_session[fs] = ps

    tests = []
    for metric in ("hr@10", "ndcg@10"):
        diff = per_session["rich"][metric] - per_session["raw"][metric]
        _, p = wilcoxon(per_session["rich"][metric], per_session["raw"][metric], zero_method="wilcox")
        tests.append({"metric": metric, "p_value": float(p), "n_sessions": int(len(diff)),
                      "n_zero": int((diff == 0).sum()), "n_rich_better": int((diff > 0).sum()),
                      "n_raw_better": int((diff < 0).sum()), "mean_diff_rich_minus_raw": float(diff.mean())})
    for t, p_holm in zip(tests, holm_bonferroni([t["p_value"] for t in tests])):
        t["p_value_holm"] = p_holm
    return rows, tests


def _fmt(x) -> str:
    return f"{x:.6g}" if isinstance(x, float) else str(x)


def render_report(dataset, rows, checks, h1_rows, h2_rows) -> str:
    lines = [f"# Test seti değerlendirmesi (A1) — {dataset}", "",
             "Yeni eğitim ve hiperparametre araması yok. Derin modeller val üzerinde seçilmiş en iyi checkpoint'lerden "
             "yüklendi; klasik modeller ve XGBoost train üzerinde deterministik olarak yeniden fit edildi.", "",
             "## Checkpoint doğrulaması (val yeniden puanlama vs runs.jsonl)", "",
             "| Model | Seed | Kaynak | max mutlak fark | Durum |", "|---|---|---|---|---|"]
    lines += [f"| {c['model']} | {c['seed']} | {c['source']} | {c['max_abs_delta_vs_logged_val']:.2e} | {'OK' if c['ok'] else 'HATA'} |" for c in checks]
    lines += ["", "## Test sonuçları", ""]
    keys = list(rows[0]["metrics"].keys())
    lines += ["| Model | Tür | Seed | n | " + " | ".join(keys) + " |", "|---|---|---|---|" + "---|" * len(keys)]
    for r in rows:
        cells = []
        for k in keys:
            v = r["metrics"].get(k)
            cells.append(f"{v['mean']:.4f} ± {v['std']:.4f}" if isinstance(v, dict) else f"{v:.4f}" if v is not None else "—")
        lines.append(f"| {r['model']} | {r.get('run_type', '-')} | {r.get('seed') or '-'} | {r['num_eval_pairs']} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Not: XGBoost satırlarında n = oturum sayısı (H2 birimi); diğer satırlarda n = önek (tahmin çifti) sayısı.")
    if h1_rows:
        lines += ["", f"## H1 test setinde (seed={H1_SEED}, Wilcoxon, Holm, {len(h1_rows)} test)", "",
                  "| Çift | Metrik | n | p | Holm p | Anlamlı | Fark (derin−klasik) | sıfır | derin> | klasik> |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for t in h1_rows:
            lines.append(f"| {t['deep']} vs {t['classical']} | {t['metric']} | {t['n_common_sessions']} | {_fmt(t['p_value'])} | "
                         f"{_fmt(t['p_value_holm'])} | {'Evet' if t['significant_alpha_0.05'] else 'Hayır'} | "
                         f"{_fmt(t['mean_diff_deep_minus_classical'])} | {t['n_zero']} | {t['n_positive']} | {t['n_negative']} |")
        lines.append("")
        lines.append("p = 0 görünen değerler float64 alt taşmasıdır; metinde p < 10^-300 yazılır.")
    if h2_rows:
        lines += ["", "## H2 test setinde (zengin vs ham, Wilcoxon, Holm)", "",
                  "| Metrik | n oturum | p | Holm p | Fark (zengin−ham) | sıfır | zengin> | ham> |", "|---|---|---|---|---|---|---|---|"]
        for t in h2_rows:
            lines.append(f"| {t['metric']} | {t['n_sessions']} | {_fmt(t['p_value'])} | {_fmt(t['p_value_holm'])} | "
                         f"{_fmt(t['mean_diff_rich_minus_raw'])} | {t['n_zero']} | {t['n_rich_better']} | {t['n_raw_better']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--ckpt-root", default="checkpoints")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print only; do not write results/")
    args = parser.parse_args()

    print(f"Cihaz: {DEVICE}", flush=True)
    runs = _load_runs()
    if any(r.get("split") == "test" and r["dataset"] == args.dataset for r in runs) and not args.dry_run:
        raise SystemExit(f"runs.jsonl zaten {args.dataset} için test satırı içeriyor; tekrar eklenmedi.")

    data_dir = Path("data/processed") / args.dataset
    train, val, test = (pd.read_parquet(data_dir / f"{s}.parquet") for s in ("train", "val", "test"))
    specs = DATASETS[args.dataset]["specs"]
    start = time.monotonic()

    print("=== Klasik modeller ===")
    rows, per_session = run_classical(args.dataset, train, test, specs, runs)
    print("=== Derin modeller (checkpoint) ===")
    deep_rows, deep_ps, checks = run_deep(args.dataset, train, val, test, specs, runs, args.ckpt_root)
    rows += deep_rows
    per_session.update(deep_ps)

    h1_rows, h2_rows = [], []
    if args.dataset == "retailrocket":
        if all(m in per_session for m in DEEP_MODELS + ["item_knn", "s_pop"]):
            print("=== H1 (test) ===")
            h1_rows, _ = run_pairwise_tests(per_session)
            h1_rows = apply_holm_bonferroni(h1_rows)
        else:
            print("  [uyarı] seed=42 checkpoint'lerinin hepsi yok; H1 test seti tekrarı atlandı.")
        if not args.skip_xgboost:
            print("=== XGBoost / H2 (test) ===")
            xgb_rows, h2_rows = run_xgboost(train, test, runs)
            rows += xgb_rows

    report = render_report(args.dataset, rows, checks, h1_rows, h2_rows)
    print("\n" + report + f"\nToplam süre: {time.monotonic() - start:.1f}s")
    if args.dry_run:
        return
    with RUNS_PATH.open("a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    Path(DATASETS[args.dataset]["report"]).write_text(report)
    if h1_rows:
        pd.DataFrame(h1_rows).drop(columns=["zero_method_cross_check"]).to_csv("results/h1_test_tests.csv", index=False)
    print(f"Yazıldı: {RUNS_PATH} (+{len(rows)} satır), {DATASETS[args.dataset]['report']}")


if __name__ == "__main__":
    main()
