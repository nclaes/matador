"""Raw-data readers: one function per parse.reader value, each turning a
group of extracted files into (X, y) arrays.

Two label conventions show up across the catalog and are dispatched on
here rather than needing a reader per combination:
  - row-level: parse.label_column picks the label out of each row.
  - file-level: parse.label_from_path derives one label per whole file
    (the file's full contents become one flattened sample) — used when a
    file IS a sample (e.g. one time-window segment per file).
"""

from __future__ import annotations

import os
import pickle
import re
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np

from matador.preprocessing.sources import ParseSpec

Reader = Callable[[list[Path], ParseSpec], tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]]


def _tokenize(line: str, delimiter: str) -> list[str]:
    if delimiter in ("whitespace", " ", "\t"):
        # collapse runs -- real whitespace/fixed-width data dumps commonly
        # have irregular spacing even when a single-space delimiter is
        # "declared"; comma-delimited files keep a strict split below since
        # empty CSV fields can be meaningful.
        return line.split()
    return [tok.strip() for tok in line.split(delimiter)]


def _read_delimited_rows(path: Path, spec: ParseSpec) -> tuple["list[str] | None", list[list[str]]]:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        return None, []
    header = _tokenize(lines[0], spec.delimiter) if spec.header else None
    data_lines = lines[1:] if spec.header else lines
    rows = [_tokenize(ln, spec.delimiter) for ln in data_lines]
    return header, rows


def _resolve_drop_columns(drop_columns: list, header: "list[str] | None", ncols: int) -> set[int]:
    out: set[int] = set()
    for d in drop_columns:
        if isinstance(d, int):
            out.add(d % ncols)
        else:
            if header is None:
                raise ValueError(f"drop_columns entry {d!r} is a name but this file has no header row")
            out.add(header.index(d))
    return out


def _rows_to_xy(
    rows: list[list[str]], header: "list[str] | None", spec: ParseSpec, source: "str | None" = None
) -> tuple[np.ndarray, "np.ndarray | None", list[str]]:
    if not rows:
        return np.empty((0, 0), dtype=np.dtype(spec.dtype)), None, []

    # ncols is taken from the first row and assumed uniform for the rest --
    # true for well-formed data, but real-world dumps sometimes have a
    # truncated/malformed line (e.g. a write cut short at EOF) with fewer
    # tokens than every other row. Rather than crash the whole ingest on one
    # bad line, skip it and surface the count as a warning.
    ncols = len(rows[0])
    drop_idx = _resolve_drop_columns(spec.drop_columns, header, ncols)
    label_idx = None
    if not spec.label_file and spec.label_column is not None:
        label_idx = spec.label_column % ncols
    feat_idx = [i for i in range(ncols) if i != label_idx and i not in drop_idx]

    X_rows: list[list[float]] = []
    y_vals: list[int] = []
    n_skipped = 0
    for row in rows:
        if len(row) != ncols:
            n_skipped += 1
            continue
        feats = [np.nan if row[i] in spec.na_values else float(row[i]) for i in feat_idx]
        X_rows.append(feats)
        if label_idx is not None:
            raw = row[label_idx]
            y_vals.append(spec.label_map[raw] if spec.label_map else int(float(raw)))

    X = np.asarray(X_rows, dtype=np.dtype(spec.dtype))
    y = np.asarray(y_vals, dtype=np.int64) if y_vals else None
    warnings: list[str] = []
    if n_skipped:
        where = f" in {source}" if source else ""
        warnings.append(f"skipped {n_skipped} malformed row(s){where}: expected {ncols} columns, got a different count")
    return X, y, warnings


def _sibling_label_path(x_path: Path) -> Path:
    """UCI-HAR-style convention: X_train.txt -> y_train.txt (label_file=True)."""
    for needle, repl in (("X_", "y_"), ("x_", "y_")):
        if needle in x_path.name:
            candidate = x_path.with_name(x_path.name.replace(needle, repl))
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"parse.label_file=True but no sibling label file found for {x_path} "
        "(expected an 'X_'->'y_' filename swap, e.g. X_train.txt -> y_train.txt)"
    )


def _relative_strings(paths: list[Path]) -> dict[Path, str]:
    """label_from_path patterns (e.g. an anchored "^(yes|no)/") are
    written against a path relative to the dataset's own extraction root,
    not the absolute filesystem path -- derive that root as the common
    ancestor of the files actually being read, so callers don't need to
    thread extraction-root bookkeeping through every reader."""
    if len(paths) == 1:
        return {paths[0]: paths[0].name}
    base = Path(os.path.commonpath([str(p) for p in paths]))
    return {p: p.relative_to(base).as_posix() for p in paths}


def _read_label_file(path: Path) -> np.ndarray:
    return np.asarray([int(tok) for ln in path.read_text().splitlines() if (tok := ln.strip())], dtype=np.int64)


def read_csv(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]:
    """Row-level delimited reader. Handles both a normal per-row label
    column and parse.label_file=True (label lives in a sibling file)."""
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    warnings: list[str] = []
    for path in paths:
        header, rows = _read_delimited_rows(path, spec)
        X, y, row_warnings = _rows_to_xy(rows, header, spec, source=path.name)
        warnings.extend(row_warnings)
        if spec.label_file:
            y = _read_label_file(_sibling_label_path(path))
        X_parts.append(X)
        if y is not None:
            y_parts.append(y)
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, 0))
    y = np.concatenate(y_parts, axis=0) if y_parts else None
    return X, y, {"n_files": len(paths), "warnings": warnings}


def _assign_path_labels(groups: list[str], label_map: dict[str, int]) -> list[int]:
    if label_map:
        return [label_map[g] for g in groups]
    if all(g.isdigit() for g in groups):
        return [int(g) for g in groups]
    uniq = sorted(set(groups))
    idx = {g: i for i, g in enumerate(uniq)}
    return [idx[g] for g in groups]


def read_dir_of_txt(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]:
    """Directory-of-files reader. Dispatches on which labeling knob is set:
    label_column -> row-level (identical semantics to read_csv, applied
    across all files); label_from_path -> file-level (each whole file
    flattens into one sample, labeled from its own path)."""
    if spec.label_from_path:
        return _read_file_level(paths, spec)
    return read_csv(paths, spec)


read_dir_of_csv = read_dir_of_txt  # same delimited-text semantics; different catalog naming only


def _read_file_level(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pattern = spec.label_from_path
    assert pattern is not None
    rels = _relative_strings(paths)
    vectors: list[np.ndarray] = []
    groups: list[str] = []
    for path in paths:
        m = re.search(pattern, rels[path])
        if not m:
            raise ValueError(f"label_from_path pattern {pattern!r} did not match {rels[path]}")
        groups.append(m.group(1))
        _, rows = _read_delimited_rows(path, spec)
        flat = np.asarray([float(tok) for row in rows for tok in row], dtype=np.dtype(spec.dtype))
        vectors.append(flat)

    lengths = {v.shape[0] for v in vectors}
    if len(lengths) > 1:
        raise ValueError(f"file-level samples have inconsistent flattened lengths: {sorted(lengths)}")

    X = np.stack(vectors, axis=0) if vectors else np.empty((0, 0))
    y = np.asarray(_assign_path_labels(groups, spec.label_map), dtype=np.int64)
    return X, y, {"n_files": len(paths), "label_groups": sorted(set(groups))}


# ---------------------------------------------------------------------------
# idx (MNIST ubyte format) — single-file loaders, paired by ingest.py's
# predefined-split resolution (images/labels pairing is filename-based:
# MNIST's real files contain the substrings "images"/"labels").
# ---------------------------------------------------------------------------

def _load_idx(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        dims = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(dims)


def read_idx(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]:
    images_path = next((p for p in paths if "images" in p.name), None)
    labels_path = next((p for p in paths if "labels" in p.name), None)
    if images_path is None:
        raise ValueError(f"read_idx: no file with 'images' in its name among {paths}")
    raw_images = _load_idx(images_path)
    image_shape = raw_images.shape[1:]
    images = raw_images.reshape(raw_images.shape[0], -1).astype(np.dtype(spec.dtype))
    y = _load_idx(labels_path).astype(np.int64) if labels_path else None
    return images, y, {"image_shape": image_shape}


# ---------------------------------------------------------------------------
# cifar_pickle — CIFAR-10 python-pickle batch format.
# ---------------------------------------------------------------------------

def read_cifar_pickle(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]:
    batches = [p for p in paths if "batches.meta" not in p.name]
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    for path in batches:
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        data = np.asarray(d[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)   # CIFAR-10 layout: R,G,B planes
        labels = np.asarray(d[b"labels"], dtype=np.int64)
        X_parts.append(data)
        y_parts.append(labels)
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, 3, 32, 32), dtype=np.uint8)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.int64)

    if spec.to_greyscale:
        # ITU-R BT.601 luma weights — the standard assumption when a
        # dataset's own greyscale conversion weights aren't recorded.
        X = np.tensordot(X.astype(np.float32), [0.299, 0.587, 0.114], axes=([1], [0]))
        X = X.reshape(X.shape[0], -1).astype(np.dtype(spec.dtype))
    else:
        X = X.reshape(X.shape[0], -1).astype(np.dtype(spec.dtype))

    if spec.relabel:
        old_to_new = {}
        for new_label, groups in spec.relabel.items():
            for _name, old_labels in groups.items():
                for old in old_labels:
                    old_to_new[old] = new_label
        y = np.asarray([old_to_new.get(int(v), int(v)) for v in y], dtype=np.int64)

    return X, y, {"image_shape": (32, 32) if spec.to_greyscale else (3, 32, 32)}


# ---------------------------------------------------------------------------
# libsvm — sparse "<class>;<aux> <idx>:<val> <idx>:<val> ..." lines (1-based
# feature indices). The token right after ';' and before the first idx:val
# pair is a per-instance auxiliary value some UCI dumps embed there (e.g.
# gas_sensor's concentration) — recorded per-row but never treated as a
# feature; there is no catalog knob to keep it, since no current dataset
# needs it as one. Feature indices must be dense/contiguous (1..n) within a
# row; a row missing indices is skipped (consistent with read_csv's
# malformed-row handling) rather than silently zero-filled, since a real
# gap here would mean a genuinely different feature count, not padding.
# ---------------------------------------------------------------------------

def read_libsvm(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]:
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    n_features: "int | None" = None
    n_skipped = 0
    for path in paths:
        rows_X: list[list[float]] = []
        rows_y: list[int] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls_str, rest = line.split(";", 1)
            pairs = [tok for tok in rest.split() if ":" in tok]
            idx_val = {}
            for tok in pairs:
                idx_s, val_s = tok.split(":", 1)
                idx_val[int(idx_s)] = float(val_s)
            n = max(idx_val) if idx_val else 0
            if n_features is None:
                n_features = n
            if n != n_features or sorted(idx_val) != list(range(1, n + 1)):
                n_skipped += 1
                continue
            rows_X.append([idx_val[i] for i in range(1, n_features + 1)])
            rows_y.append(spec.label_map[cls_str] if spec.label_map else int(float(cls_str)))
        if rows_X:
            X_parts.append(np.asarray(rows_X, dtype=np.dtype(spec.dtype)))
            y_parts.append(np.asarray(rows_y, dtype=np.int64))
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0, 0))
    y = np.concatenate(y_parts, axis=0) if y_parts else None
    warnings = [f"skipped {n_skipped} malformed/inconsistent row(s)"] if n_skipped else []
    return X, y, {"n_files": len(paths), "n_features": n_features, "warnings": warnings}


# ---------------------------------------------------------------------------
# wav_dir — fixed-length raw PCM sample reader (stdlib `wave` only; no DSP
# feature extraction — turning this into e.g. MFCCs is a booleanization-time
# concern, out of scope for raw ingestion; an MFCC-style bit encoding is
# generally unreproducible without the original feature pipeline).
# ---------------------------------------------------------------------------

def read_wav_dir(paths: list[Path], spec: ParseSpec) -> tuple[np.ndarray, "np.ndarray | None", dict[str, Any]]:
    pattern = spec.label_from_path
    if pattern is None:
        raise ValueError("wav_dir reader requires parse.label_from_path")

    rels = _relative_strings(paths)
    vectors: list[np.ndarray] = []
    groups: list[str] = []
    target_len: "int | None" = None
    for path in paths:
        m = re.search(pattern, rels[path])
        if not m:
            continue   # e.g. _background_noise_/*.wav, spec_files -- not a labeled clip
        with wave.open(str(path), "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if target_len is None:
            target_len = samples.shape[0]
        # pad/truncate to a common length so clips of a fixed nominal
        # duration (e.g. 1.0s @ 16kHz) can be stacked into one array
        if samples.shape[0] < target_len:
            samples = np.pad(samples, (0, target_len - samples.shape[0]))
        else:
            samples = samples[:target_len]
        vectors.append(samples)
        groups.append(m.group(1))

    X = np.stack(vectors, axis=0).astype(np.dtype(spec.dtype)) if vectors else np.empty((0, 0))
    y = np.asarray(_assign_path_labels(groups, spec.label_map), dtype=np.int64)
    return X, y, {"n_clips": len(vectors), "label_groups": sorted(set(groups))}


READERS: dict[str, Reader] = {
    "csv": read_csv,
    "dir_of_csv": read_dir_of_csv,
    "dir_of_txt": read_dir_of_txt,
    "idx": read_idx,
    "cifar_pickle": read_cifar_pickle,
    "wav_dir": read_wav_dir,
    "libsvm": read_libsvm,
}
