"""TFLite Interpreter wrapped behind the RecommenderModel protocol
(src/models/base.py: fit/predict), so src/eval/harness.py can evaluate a
converted .tflite variant exactly like the PyTorch models -- no harness
changes needed."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.models.pop import PopModel
from src.ondevice.export_onnx import prepare_input

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:  # package naming has been migrating; fall back to the TF-bundled runtime
    from tensorflow.lite import Interpreter


class TFLiteRecommender:
    """fit() is a no-op for the converted network itself (already trained) --
    it only fits the PopModel fallback used for sessions with zero known
    items, mirroring Gru4RecModel/LstmModel/SasrecModel.predict()."""

    def __init__(
        self,
        tflite_path: str,
        item_to_idx: dict[int, int],
        idx_to_item: np.ndarray,
        max_seq_len: int,
    ) -> None:
        self.item_to_idx = item_to_idx
        self.idx_to_item = idx_to_item
        self.max_seq_len = max_seq_len
        self._pop = PopModel()

        self._interpreter = Interpreter(model_path=tflite_path)
        self._interpreter.allocate_tensors()
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()
        # onnx2tf preserves the ONNX input names ("item_ids"/"last_index") when
        # copy_onnx_input_output_names_to_tflite=True -- match by name, since
        # TFLiteConverter does not guarantee input order matches ONNX's.
        self._item_ids_detail = next(d for d in input_details if "item_ids" in d["name"])
        self._last_index_detail = next(d for d in input_details if "last_index" in d["name"])
        self._output_detail = output_details[0]

    def fit(self, train: pd.DataFrame) -> None:
        self._pop.fit(train)

    def predict(self, session: Sequence[int]) -> list[int]:
        known = [self.item_to_idx[i] for i in session if i in self.item_to_idx]
        if not known:
            return self._pop.predict(session)

        item_ids, last_index = prepare_input(known, self.max_seq_len)
        logits = self._invoke(item_ids, last_index)
        scores = logits[1:]  # drop PAD index 0; remaining n_items align with idx_to_item
        order = np.lexsort((self.idx_to_item, -scores))
        return self.idx_to_item[order].tolist()

    def predict_many(self, sessions: Sequence[Sequence[int]]) -> list[list[int]]:
        # Static batch=1 input shape (the whole point of the export design)
        # means there is no native batching path here -- loop per session.
        # Acceptable for this accuracy/size-only stage (CLAUDE.md rule 6
        # forbids reporting any timing anyway, so per-call overhead is moot).
        return [self.predict(session) for session in sessions]

    def _invoke(self, item_ids: np.ndarray, last_index: np.ndarray) -> np.ndarray:
        item_ids = self._to_input_dtype(item_ids, self._item_ids_detail)
        last_index = self._to_input_dtype(last_index, self._last_index_detail)
        self._interpreter.set_tensor(self._item_ids_detail["index"], item_ids)
        self._interpreter.set_tensor(self._last_index_detail["index"], last_index)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(self._output_detail["index"])[0]
        return self._from_output_dtype(raw, self._output_detail)

    @staticmethod
    def _to_input_dtype(value: np.ndarray, detail: dict) -> np.ndarray:
        """item_ids/last_index are categorical indices, not continuous signal --
        if a full-integer-quantized graph forces them into int8/uint8 (rather
        than keeping index inputs at their natural int64/int32 dtype, an
        adaptation noted as an open risk in the implementation plan), apply the
        interpreter's own scale/zero-point so the values still round-trip;
        otherwise just cast to whatever dtype the graph expects."""
        dtype = detail["dtype"]
        if np.issubdtype(dtype, np.integer) and dtype not in (np.int64, np.int32):
            scale, zero_point = detail["quantization"]
            if scale:
                value = np.round(value / scale + zero_point)
        return value.astype(dtype)

    @staticmethod
    def _from_output_dtype(value: np.ndarray, detail: dict) -> np.ndarray:
        scale, zero_point = detail.get("quantization", (0.0, 0))
        if scale:
            return (value.astype(np.float32) - zero_point) * scale
        return value.astype(np.float32)
