"""Stage 2 orchestration: raw (x, y) arrays -> Boolean train/test text files
in the exact format matador.models.trainer.load_data() already parses
(space-separated uint, last column = label) — so `matador train` needs no
changes at all to consume this stage's output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from matador.preprocessing.encoders import ENCODERS, n_bits_for

if TYPE_CHECKING:
    from matador.config.schema import BooleanisationConfig, FeatureEncoderSpec


@dataclass
class BooleanizeReport:
    name: str
    n_train: int
    n_test: int
    n_features_raw: int
    n_features_bool: int
    per_column: list[dict] = field(default_factory=list)
    output_paths: dict[str, Path] = field(default_factory=dict)


def _load_raw(config: "BooleanisationConfig") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(config.raw_npz)
    if "x_train" in data.files:
        x_train, x_test = data["x_train"], data["x_test"]
        y_train, y_test = data.get("y_train"), data.get("y_test")
    elif "x" in data.files:
        from matador.preprocessing.sources import SplitSpec
        from matador.preprocessing.splitters import split_random

        X, y = data["x"], data.get("y")
        split_spec = SplitSpec(mode="random", test_size=config.test_size, seed=config.seed, stratify=config.stratify)
        x_train, x_test, y_train, y_test = split_random(X, y, split_spec)
    else:
        raise ValueError(f"{config.raw_npz}: expected 'x_train'/'x_test' or 'x'/'y' arrays, found {data.files}")

    if y_train is None or y_test is None:
        raise ValueError(f"{config.raw_npz}: booleanization requires labels (y_train/y_test or y)")
    return x_train, y_train, x_test, y_test


def _expand_column(column: "int | str", n_columns: int) -> list[int]:
    if isinstance(column, int):
        idx = column % n_columns
        return [idx]
    lo_s, _, hi_s = column.partition("-")
    lo, hi = int(lo_s), int(hi_s)
    return list(range(lo, hi + 1))


def _resolve_column_specs(
    features: "list[FeatureEncoderSpec]",
    default_encoder: "Optional[FeatureEncoderSpec]",
    n_columns: int,
) -> "dict[int, FeatureEncoderSpec]":
    resolved: "dict[int, FeatureEncoderSpec]" = {}
    for spec in features:
        for c in _expand_column(spec.column, n_columns):
            if c in resolved:
                raise ValueError(f"column {c} is covered by more than one features[] entry")
            resolved[c] = spec
    for c in range(n_columns):
        if c not in resolved:
            if default_encoder is None:
                raise ValueError(
                    f"column {c} is not covered by features[] and no default_encoder is set"
                )
            resolved[c] = default_encoder
    return resolved


def run_booleanize(config: "BooleanisationConfig") -> BooleanizeReport:
    x_train, y_train, x_test, y_test = _load_raw(config)
    n_columns = x_train.shape[1]
    resolved = _resolve_column_specs(config.features, config.default_encoder, n_columns)

    train_blocks, test_blocks, per_column = [], [], []
    for c in range(n_columns):
        spec = resolved[c]
        fit_fn, apply_fn = ENCODERS[spec.encoder]
        train_col, test_col = x_train[:, c], x_test[:, c]
        params = fit_fn(train_col, spec)
        train_blocks.append(apply_fn(train_col, params))
        test_blocks.append(apply_fn(test_col, params))
        n_nan = int(np.sum(~np.isfinite(train_col))) if np.issubdtype(train_col.dtype, np.floating) else 0
        per_column.append({
            "column": c, "encoder": spec.encoder,
            "bits": n_bits_for(spec.encoder, params), "n_nan_train": n_nan,
        })

    X_train_bool = np.concatenate(train_blocks, axis=1)
    X_test_bool = np.concatenate(test_blocks, axis=1)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"{config.name}_train.txt"
    test_path = output_dir / f"{config.name}_test.txt"
    np.savetxt(train_path, np.hstack([X_train_bool, y_train.reshape(-1, 1).astype(np.uint32)]), fmt="%d")
    np.savetxt(test_path, np.hstack([X_test_bool, y_test.reshape(-1, 1).astype(np.uint32)]), fmt="%d")

    report_path = output_dir / f"{config.name}_report.json"
    classes, counts = np.unique(np.concatenate([y_train, y_test]), return_counts=True)
    report_data = {
        "name": config.name,
        "n_train": int(X_train_bool.shape[0]),
        "n_test": int(X_test_bool.shape[0]),
        "n_features_raw": n_columns,
        "n_features_bool": int(X_train_bool.shape[1]),
        "per_column": per_column,
        "class_distribution": {int(k): int(v) for k, v in zip(classes, counts)},
    }
    report_path.write_text(json.dumps(report_data, indent=2))

    return BooleanizeReport(
        name=config.name,
        n_train=int(X_train_bool.shape[0]),
        n_test=int(X_test_bool.shape[0]),
        n_features_raw=n_columns,
        n_features_bool=int(X_train_bool.shape[1]),
        per_column=per_column,
        output_paths={"train": train_path, "test": test_path, "report": report_path},
    )
