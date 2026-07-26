"""Per-column Boolean encoders: numeric/categorical raw features -> fixed-
width bit blocks. Each encoder fits its parameters (bin edges, categories,
...) on the TRAIN split only, then applies the same fitted parameters to
both train and test — avoiding test-set leakage.

Non-finite (NaN) values — which can reach here from a csv reader's
na_values handling — encode to all-zero bits rather than raising; this is a
documented default, not silent data loss (the booleanize report records how
many NaNs were seen per column).
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def fit_thermometer(train_col: np.ndarray, bits: int, value_range: "tuple[float, float] | None",
                     bins: "list[float] | None", quantile: bool) -> np.ndarray:
    if bins:
        thresholds = np.asarray(bins, dtype=float)
        if thresholds.shape[0] != bits:
            raise ValueError(f"thermometer: {len(bins)} explicit bins given but bits={bits}")
        return thresholds

    finite = train_col[np.isfinite(train_col)]
    if finite.size == 0:
        raise ValueError("thermometer: column has no finite values to fit thresholds from")

    if quantile:
        qs = np.linspace(0, 1, bits + 2)[1:-1]
        return np.quantile(finite, qs)

    lo, hi = value_range if value_range else (float(finite.min()), float(finite.max()))
    return np.linspace(lo, hi, bits + 2)[1:-1]


def encode_thermometer(col: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    valid = np.isfinite(col)
    out = np.zeros((col.shape[0], thresholds.shape[0]), dtype=np.uint8)
    for i, t in enumerate(thresholds):
        out[valid, i] = (col[valid] > t).astype(np.uint8)
    return out


def encode_threshold(col: np.ndarray, threshold: float) -> np.ndarray:
    valid = np.isfinite(col)
    out = np.zeros((col.shape[0], 1), dtype=np.uint8)
    out[valid, 0] = (col[valid] > threshold).astype(np.uint8)
    return out


def fit_onehot(train_col: np.ndarray, categories: "list[Any] | None") -> list:
    if categories is not None:
        return list(categories)
    finite = train_col[np.isfinite(train_col)] if np.issubdtype(train_col.dtype, np.floating) else train_col
    return sorted(set(finite.tolist()))


def encode_onehot(col: np.ndarray, categories: list) -> np.ndarray:
    out = np.zeros((col.shape[0], len(categories)), dtype=np.uint8)
    for i, cat in enumerate(categories):
        out[:, i] = (col == cat).astype(np.uint8)
    return out


def encode_passthrough(col: np.ndarray) -> np.ndarray:
    valid = np.isfinite(col)
    out = np.zeros((col.shape[0], 1), dtype=np.uint8)
    out[valid, 0] = (col[valid] != 0).astype(np.uint8)
    return out


# Each entry: (fit(train_col, spec) -> params, apply(col, params) -> uint8[n, bits])
ENCODERS: dict[str, tuple[Callable, Callable]] = {
    "thermometer": (
        lambda train_col, spec: fit_thermometer(train_col, spec.bits or 8, spec.range, spec.bins, spec.quantile),
        encode_thermometer,
    ),
    "threshold": (
        lambda train_col, spec: spec.threshold if spec.threshold is not None else 0.0,
        encode_threshold,
    ),
    "onehot": (
        lambda train_col, spec: fit_onehot(train_col, spec.categories),
        encode_onehot,
    ),
    "passthrough": (
        lambda train_col, spec: None,
        lambda col, _params: encode_passthrough(col),
    ),
}


def n_bits_for(encoder: str, params: Any) -> int:
    if encoder == "thermometer":
        return int(params.shape[0])
    if encoder == "onehot":
        return len(params)
    return 1
