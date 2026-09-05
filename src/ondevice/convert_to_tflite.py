"""ONNX -> TFLite conversion: FP32 / dynamic-range INT8 / full-integer INT8.

Uses onnx2tf to bridge ONNX -> TF SavedModel (actively maintained, unlike the
older onnx-tf; built specifically to feed tf.lite.TFLiteConverter), then
tf.lite.TFLiteConverter for all three variants from the same SavedModel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.ondevice.export_onnx import prepare_input

Variant = Literal["fp32", "dynamic_int8", "full_int8"]


def onnx_to_saved_model(onnx_path: str | Path, saved_model_dir: str | Path, max_seq_len: int) -> None:
    import onnx2tf

    Path(saved_model_dir).mkdir(parents=True, exist_ok=True)
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(saved_model_dir),
        copy_onnx_input_output_names_to_tflite=True,
        # Forces a fully static input shape (batch=1, matching the export
        # wrapper's fixed contract) rather than onnx2tf's default dynamic
        # (-1) batch dimension in the SavedModel signature.
        overwrite_input_shape=[f"item_ids:1,{max_seq_len}", "last_index:1"],
        non_verbose=True,
    )


def sample_representative_sessions(
    train_path: str,
    item_to_idx: dict[int, int],
    max_seq_len: int,
    n_samples: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Samples real sessions from train (never val, to avoid calibration
    leakage into the accuracy comparison) for full-integer INT8 calibration."""
    train = pd.read_parquet(train_path)
    rng = np.random.default_rng(seed)
    sessions = []
    ordered = train.sort_values(["session_id", "session_event_index"])
    for _, items in ordered.groupby("session_id", sort=False)["itemid"]:
        known = [item_to_idx[i] for i in items.tolist() if i in item_to_idx]
        if known:
            sessions.append(known)
    chosen = rng.choice(len(sessions), size=min(n_samples, len(sessions)), replace=False)
    return [prepare_input(sessions[i], max_seq_len) for i in chosen]


def _representative_dataset_fn(samples: list[tuple[np.ndarray, np.ndarray]]):
    def gen():
        for item_ids, last_index in samples:
            yield [item_ids.astype(np.int64), last_index.astype(np.int64)]

    return gen


def convert_variant(
    saved_model_dir: str | Path,
    variant: Variant,
    out_path: str | Path,
    representative_samples: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if variant == "dynamic_int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif variant == "full_int8":
        if representative_samples is None:
            raise ValueError("full_int8 requires representative_samples")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset_fn(representative_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    elif variant != "fp32":
        raise ValueError(f"unknown variant: {variant}")

    tflite_model = converter.convert()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(tflite_model)


def model_size_mb(path: str | Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def convert_all_variants(
    saved_model_dir: str | Path,
    variants: list[Variant],
    out_dir: str | Path,
    model_name: str,
    representative_samples: list[tuple[np.ndarray, np.ndarray]] | None,
) -> dict[str, dict]:
    """Converts every requested variant independently, catching failures
    per-variant (full_int8 is expected to sometimes fail on embedding-heavy
    nets -- this is a reportable finding, not a bug; see docs/00-karar-notu.md
    §8's own open question about embedding-layer quantization behavior)."""
    results: dict[str, dict] = {}
    for variant in variants:
        out_path = Path(out_dir) / f"{model_name}_{variant}.tflite"
        try:
            convert_variant(saved_model_dir, variant, out_path, representative_samples)
            results[variant] = {"path": str(out_path), "size_mb": model_size_mb(out_path), "conversion_error": None}
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: any converter failure is a reportable finding
            results[variant] = {"path": None, "size_mb": None, "conversion_error": f"{type(exc).__name__}: {exc}"}
    return results
