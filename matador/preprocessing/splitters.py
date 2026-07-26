"""Turns a raw (X, y) pool into train/test arrays, either by a fresh random
split or by honoring a source's own predefined partition."""

from __future__ import annotations

import fnmatch
import random
from pathlib import Path
from typing import Optional

import numpy as np

from matador.preprocessing.sources import SplitSpec


def split_random(X: np.ndarray, y: "np.ndarray | None", spec: SplitSpec) -> tuple[np.ndarray, np.ndarray, "np.ndarray | None", "np.ndarray | None"]:
    """Seeded random train/test split, optionally class-stratified.

    Uses sklearn's train_test_split when available (it's what actually
    produced several of the catalog's own recorded split sizes, e.g.
    digits' 1437/360 -- its stratified rounding isn't independent
    per-class, so a from-scratch reimplementation can land a row or two
    off). Falls back to a plain numpy stratified split if sklearn isn't
    installed -- sklearn is not a hard matador dependency."""
    n = X.shape[0]

    if spec.stratify and y is not None:
        try:
            from sklearn.model_selection import train_test_split as _sk_split
        except ImportError:
            _sk_split = None
        if _sk_split is not None:
            idx = np.arange(n)
            train_idx, test_idx = _sk_split(idx, test_size=spec.test_size, random_state=spec.seed, stratify=y)
            test_idx = np.sort(test_idx)
        else:
            rng = np.random.default_rng(spec.seed)
            test_idx_list: list[int] = []
            for cls in np.unique(y):
                cls_idx = np.flatnonzero(y == cls)
                rng.shuffle(cls_idx)
                k = max(1, round(len(cls_idx) * spec.test_size))
                test_idx_list.extend(cls_idx[:k].tolist())
            test_idx = np.asarray(sorted(test_idx_list))
    else:
        rng = np.random.default_rng(spec.seed)
        n_test = max(1, round(n * spec.test_size))
        perm = rng.permutation(n)
        test_idx = np.sort(perm[:n_test])

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_idx] = True
    train_mask = ~test_mask

    y_train = y[train_mask] if y is not None else None
    y_test = y[test_mask] if y is not None else None
    return X[train_mask], X[test_mask], y_train, y_test


def _matches(path_rel: str, path_name: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path_rel, pattern) or fnmatch.fnmatch(path_name, pattern)


def resolve_predefined_group(
    patterns: list[str],
    extracted: list[Path],
    root_by_path: "dict[Path, Path]",
    role_by_path: "dict[Path, str]",
) -> list[Path]:
    """Resolve one of split.train/split.test's entries against the actually
    extracted files. Each entry is tried first as an exact role name
    (matches source.urls[].role, e.g. mnist's "train_X"), then as a glob
    pattern against the extracted path (relative to its own root — see
    root_by_path — and against the bare filename — covers both
    human_activity's "**/train/X_train.txt" and cifar2's "data_batch_*")."""
    out: list[Path] = []
    for pattern in patterns:
        role_matches = [p for p in extracted if role_by_path.get(p) == pattern]
        if role_matches:
            out.extend(role_matches)
            continue
        for p in extracted:
            rel = p.relative_to(root_by_path[p]).as_posix()
            if _matches(rel, p.name, pattern):
                out.append(p)
    return out


def resolve_spec_files_split(
    spec_files: list[str],
    extracted: list[Path],
    root_by_path: "dict[Path, Path]",
) -> tuple[list[Path], list[Path]]:
    """kws2-style: one or more line-list files enumerate relative paths held
    out for validation/test; everything else is train. Matador only needs a
    train/test split, so validation_list.txt + testing_list.txt are unioned
    into "test"."""
    held_out: set[str] = set()
    for spec_file_pattern in spec_files:
        spec_path = next((p for p in extracted if p.name == spec_file_pattern), None)
        if spec_path is None:
            raise FileNotFoundError(f"spec_files entry {spec_file_pattern!r} not found among extracted files")
        held_out.update(ln.strip() for ln in spec_path.read_text().splitlines() if ln.strip())

    train, test = [], []
    for p in extracted:
        if p.suffix == ".txt" and p.name in spec_files:
            continue
        rel = p.relative_to(root_by_path[p]).as_posix()
        (test if rel in held_out else train).append(p)
    return train, test
