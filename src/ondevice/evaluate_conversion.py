"""On-device fizibilite, Aşama 1: PyTorch checkpoint -> ONNX -> TFLite
(FP32 / dinamik INT8 / tam INT8), her varyant için dosya boyutu + NDCG@10
kuantizasyon kaybı. Sonuçlar results/ondevice_conversion_report.md ve
results/runs.jsonl'e yazılır (docs/00-karar-notu.md §8/§12c).

CLI usage:
  python -m src.ondevice.evaluate_conversion --config configs/ondevice_conversion.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.eval.harness import count_eval_pairs, evaluate_per_pair_and_session
from src.models.pop import PopModel
from src.ondevice.convert_to_tflite import (
    convert_all_variants,
    onnx_to_saved_model,
    sample_representative_sessions,
)
from src.ondevice.export_onnx import (
    build_export_wrapper,
    build_net,
    check_onnx_matches_pytorch,
    export_to_onnx,
    get_vocab,
    load_checkpoint,
    prepare_input,
)
from src.ondevice.tflite_model import TFLiteRecommender

CONFIG_PATH = "configs/ondevice_conversion.yaml"


class _PyTorchNetRecommender:
    """Wraps a raw _GRU4RecNet/_LSTMNet/_SASRecNet (encode()/project()) behind
    the RecommenderModel protocol, replicating Gru4RecModel/LstmModel/
    SasrecModel's predict()/predict_many() logic exactly -- this is the FP32
    baseline every on-device variant is compared against."""

    def __init__(self, net, item_to_idx: dict[int, int], idx_to_item: np.ndarray, max_seq_len: int, pop: PopModel):
        self.net = net
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.max_seq_len = max_seq_len
        self._pop = pop

    def fit(self, train: pd.DataFrame) -> None:
        pass  # already trained; PopModel fallback is fit by the caller

    def predict(self, session):
        known = [item for item in session if item in self.item_to_idx]
        if not known:
            return self._pop.predict(session)
        known = known[-self.max_seq_len :]
        idx_seq = torch.tensor([[self.item_to_idx[i] for i in known]], dtype=torch.long)
        length = torch.tensor([len(known)], dtype=torch.long)
        with torch.no_grad():
            hidden = self.net.encode(idx_seq, length)
            last_hidden = hidden[0, length[0] - 1].unsqueeze(0)
            logits = self.net.project(last_hidden).squeeze(0)
        scores = logits[1:].cpu().numpy()
        order = np.lexsort((self.idx_to_item, -scores))
        return self.idx_to_item[order].tolist()

    def predict_many(self, sessions, eval_batch_size: int = 512):
        results: list[list[int] | None] = [None] * len(sessions)
        net_indices: list[int] = []
        net_known: list[list[int]] = []
        for i, session in enumerate(sessions):
            known = [item for item in session if item in self.item_to_idx]
            if not known:
                results[i] = self._pop.predict(session)
                continue
            net_indices.append(i)
            net_known.append(known[-self.max_seq_len :])

        with torch.no_grad():
            for start in range(0, len(net_known), eval_batch_size):
                chunk_indices = net_indices[start : start + eval_batch_size]
                chunk_known = net_known[start : start + eval_batch_size]
                lengths = torch.tensor([len(k) for k in chunk_known], dtype=torch.long)
                max_len = int(lengths.max())
                x = torch.zeros((len(chunk_known), max_len), dtype=torch.long)
                for row, known in enumerate(chunk_known):
                    x[row, : len(known)] = torch.tensor([self.item_to_idx[i] for i in known], dtype=torch.long)
                hidden = self.net.encode(x, lengths)
                last_hidden = hidden[torch.arange(len(chunk_known)), lengths - 1]
                logits = self.net.project(last_hidden)
                scores = logits[:, 1:].cpu().numpy()
                for row, result_idx in enumerate(chunk_indices):
                    order = np.lexsort((self.idx_to_item, -scores[row]))
                    results[result_idx] = self.idx_to_item[order].tolist()
        return results  # type: ignore[return-value]


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _harness_metrics(model, val: pd.DataFrame, k: int) -> dict[str, float]:
    per_pair, _ = evaluate_per_pair_and_session(model, val, k=k, batched=True)
    return {
        f"hr@{k}": float(per_pair[f"hr@{k}"].mean()),
        f"ndcg@{k}": float(per_pair[f"ndcg@{k}"].mean()),
        f"mrr@{k}": float(per_pair[f"mrr@{k}"].mean()),
    }


def _sample_eval_inputs(val: pd.DataFrame, item_to_idx: dict[int, int], max_seq_len: int, n: int = 20):
    """A handful of real val sessions/prefixes for the ONNX-vs-PyTorch sanity
    check (§7 of the plan) -- more robust than a single dummy tensor."""
    samples = []
    ordered = val.sort_values(["session_id", "session_event_index"])
    for _, items in ordered.groupby("session_id", sort=False)["itemid"]:
        known = [item_to_idx[i] for i in items.tolist() if i in item_to_idx]
        if known:
            samples.append(prepare_input(known, max_seq_len))
        if len(samples) >= n:
            break
    return samples


def _known_baseline_ndcg10(model_name: str) -> float | None:
    """Reference NDCG@10 already logged in results/runs.jsonl for the seed=42
    RetailRocket run (see results/h1_hypothesis_report.md's reproducibility
    check) -- the gate the reconstructed PyTorch baseline must reproduce
    before anything downstream (ONNX/TFLite) is trusted."""
    return {"gru4rec": 0.334646, "lstm": 0.330475}.get(model_name)


def run_model(model_name: str, config: dict) -> dict:
    print(f"=== {model_name}: checkpoint yükleniyor ===", flush=True)
    checkpoint_path = config["checkpoints"][model_name]
    if not Path(checkpoint_path).exists():
        return {"model": model_name, "conversion_error": f"checkpoint bulunamadı: {checkpoint_path}", "variants": {}}

    payload = load_checkpoint(checkpoint_path)
    item_to_idx, idx_to_item = get_vocab(payload, train_path=config["data"]["train_path"])
    n_items = len(idx_to_item)
    net = build_net(model_name, payload, n_items)

    train = pd.read_parquet(config["data"]["train_path"])
    val = pd.read_parquet(config["data"]["val_path"])
    k = config["eval"]["k"]
    pop = PopModel()
    pop.fit(train)

    baseline_model = _PyTorchNetRecommender(net, item_to_idx, idx_to_item, config["export"]["max_seq_len"], pop)
    baseline_metrics = _harness_metrics(baseline_model, val, k)
    print(f"  PyTorch FP32 baseline: {baseline_metrics}", flush=True)

    reference = _known_baseline_ndcg10(model_name)
    if reference is not None:
        tol = config["baseline_reproduction_tolerance"]
        diff = abs(baseline_metrics[f"ndcg@{k}"] - reference)
        if diff > tol:
            raise RuntimeError(
                f"{model_name}: yeniden kurulan PyTorch baseline NDCG@{k}={baseline_metrics[f'ndcg@{k}']:.6f}, "
                f"results/runs.jsonl'deki seed=42 değeri {reference} ile uyuşmuyor (fark={diff:.6f} > tol={tol}). "
                "Bu, checkpoint/vocab yükleme adımında bir hataya işaret eder -- export/dönüşüm koduna geçilmeden "
                "önce durup incelenmeli (CLAUDE.md: beklenenden kötü sonuç -> dur, sor)."
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        max_seq_len = config["export"]["max_seq_len"]
        onnx_path = Path(tmpdir) / f"{model_name}.onnx"
        saved_model_dir = Path(tmpdir) / "saved_model"

        # PyTorch->ONNX->TF-SavedModel is best-effort per model (SASRec's
        # causal-mask ops are known to sometimes fail this bridge entirely --
        # docs/00-karar-notu.md §9/§12c: SASRec conversion is optional/
        # sacrificable). Any failure here is a reportable finding, not a
        # pipeline-stopping bug -- caught broadly so one model's failure
        # never blocks the others.
        try:
            wrapper = build_export_wrapper(model_name, net)
            export_to_onnx(wrapper, max_seq_len, onnx_path, opset=config["export"]["onnx_opset"])
            sample_inputs = _sample_eval_inputs(val, item_to_idx, max_seq_len)
            max_diff = check_onnx_matches_pytorch(wrapper, onnx_path, sample_inputs)
            print(f"  ONNX vs PyTorch max fark: {max_diff:.2e}", flush=True)
            onnx_to_saved_model(onnx_path, saved_model_dir, max_seq_len)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see comment above
            error = f"{type(exc).__name__}: {exc}"
            return {"model": model_name, "baseline_metrics": baseline_metrics, "conversion_error": error, "variants": {}}

        representative_samples = None
        if "full_int8" in config["variants"]:
            representative_samples = sample_representative_sessions(
                config["data"]["train_path"],
                item_to_idx,
                max_seq_len,
                config["representative_dataset"]["sample_size"],
                config["representative_dataset"]["sample_seed"],
            )

        out_dir = Path(config["output_dir"])
        variant_results = convert_all_variants(
            saved_model_dir, config["variants"], out_dir, model_name, representative_samples
        )

        for variant, info in variant_results.items():
            if info["conversion_error"] is not None:
                print(f"  {variant}: dönüşüm başarısız -- {info['conversion_error']}", flush=True)
                continue
            tflite_model = TFLiteRecommender(info["path"], item_to_idx, idx_to_item, max_seq_len)
            tflite_model.fit(train)
            info["metrics"] = _harness_metrics(tflite_model, val, k)
            print(f"  {variant}: size={info['size_mb']:.3f}MB, metrics={info['metrics']}", flush=True)

    return {
        "model": model_name,
        "baseline_metrics": baseline_metrics,
        "num_eval_pairs": count_eval_pairs(val),
        "conversion_error": None,
        "variants": variant_results,
    }


def _sanitize_note(text: str) -> str:
    """Collapses embedded newlines/tabs (some TF/onnx2tf exceptions span
    multiple lines) so a single error message can't break a markdown table row."""
    return " ".join(text.split())


def write_report(all_results: list[dict], report_path: str, k: int) -> None:
    lines = [
        "# On-device Dönüşüm Raporu (Aşama 1): GRU4Rec/LSTM/SASRec → LiteRT",
        "",
        "RetailRocket val seti üzerinde, PyTorch FP32 baseline'a karşı FP32/dinamik "
        "INT8/tam INT8 TFLite varyantlarının dosya boyutu ve NDCG@10 kaybı "
        "(docs/00-karar-notu.md §8/§12c).",
        "",
        f"| Model | Varyant | Dosya boyutu (MB) | NDCG@{k} (orijinal) | NDCG@{k} (kuantize) | "
        "Fark (mutlak) | Fark (%) | Not |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for result in all_results:
        model = result["model"]
        if result["conversion_error"]:
            baseline = result.get("baseline_metrics")
            baseline_str = f"{baseline[f'ndcg@{k}']:.6f}" if baseline else "-"
            lines.append(f"| {model} | - | - | {baseline_str} | - | - | - | {_sanitize_note(result['conversion_error'])} |")
            continue
        baseline_ndcg = result["baseline_metrics"][f"ndcg@{k}"]
        for variant, info in result["variants"].items():
            if info["conversion_error"] is not None:
                note = _sanitize_note(f"dönüşüm başarısız: {info['conversion_error']}")
                lines.append(f"| {model} | {variant} | - | {baseline_ndcg:.6f} | - | - | - | {note} |")
                continue
            variant_ndcg = info["metrics"][f"ndcg@{k}"]
            abs_diff = baseline_ndcg - variant_ndcg
            pct_diff = (abs_diff / baseline_ndcg * 100) if baseline_ndcg else float("nan")
            lines.append(
                f"| {model} | {variant} | {info['size_mb']:.3f} | {baseline_ndcg:.6f} | {variant_ndcg:.6f} | "
                f"{abs_diff:.6f} | {pct_diff:.2f} | |"
            )
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines) + "\n")


def append_runs_jsonl(all_results: list[dict], config: dict, k: int) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = _git_commit()
    Path("results").mkdir(exist_ok=True)
    with open("results/runs.jsonl", "a") as f:
        for result in all_results:
            model = result["model"]
            if result["conversion_error"]:
                continue
            rows = [
                {
                    "timestamp": timestamp,
                    "model": model,
                    "dataset": config["dataset"],
                    "split": "val",
                    "variant": "pytorch_fp32",
                    "config": {"seed": config["seed"], "max_seq_len": config["export"]["max_seq_len"]},
                    "metrics": result["baseline_metrics"],
                    "model_size_mb": None,
                    "num_eval_pairs": result["num_eval_pairs"],
                    "conversion_error": None,
                    "git_commit": commit,
                }
            ]
            for variant, info in result["variants"].items():
                rows.append(
                    {
                        "timestamp": timestamp,
                        "model": model,
                        "dataset": config["dataset"],
                        "split": "val",
                        "variant": f"tflite_{variant}",
                        "config": {"seed": config["seed"], "max_seq_len": config["export"]["max_seq_len"]},
                        "metrics": info.get("metrics"),
                        "model_size_mb": info["size_mb"],
                        "num_eval_pairs": result["num_eval_pairs"],
                        "conversion_error": info["conversion_error"],
                        "git_commit": commit,
                    }
                )
            for row in rows:
                f.write(json.dumps(row) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-device dönüşüm + değerlendirme (Aşama 1)")
    parser.add_argument("--config", default=CONFIG_PATH, help="Path to ondevice_conversion.yaml")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    k = config["eval"]["k"]

    all_results = []
    for model_name in config["models"]:
        try:
            all_results.append(run_model(model_name, config))
        except RuntimeError as exc:
            print(f"HATA: {exc}", flush=True)
            raise

    write_report(all_results, config["report_path"], k)
    append_runs_jsonl(all_results, config, k)
    print(f"\nRapor yazıldı: {config['report_path']}", flush=True)


if __name__ == "__main__":
    main()
