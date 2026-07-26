"""Stage 1 orchestration: fetch -> extract -> resolve split -> read -> export.

`run_ingest()` is the single entry point the `matador ingest` CLI command
calls; every other module in this package is a building block it composes.
"""

from __future__ import annotations

import csv
import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from matador.preprocessing.archives import extract_archive
from matador.preprocessing.fetch import fetch_source
from matador.preprocessing.readers import READERS
from matador.preprocessing.sources import CatalogDefaults, ExtractSpec, RawDataSourceSpec
from matador.preprocessing.splitters import (
    resolve_predefined_group,
    resolve_spec_files_split,
    split_random,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class IngestReport:
    key: str
    x_train: np.ndarray
    y_train: "Optional[np.ndarray]"
    x_test: np.ndarray
    y_test: "Optional[np.ndarray]"
    output_paths: dict[str, Path] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    label_offset: int = 0


def _resolve_local_files(path: Path, extract: ExtractSpec) -> list[Path]:
    if path.is_file():
        return [path]
    if not extract.members:
        return sorted(p for p in path.rglob("*") if p.is_file())
    out = []
    for p in path.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(path).as_posix()
        if any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(p.name, pat) for pat in extract.members):
            out.append(p)
    return sorted(out)


def _fetch_and_extract(spec: RawDataSourceSpec, cache_dir: Path, extract_dir: Path, defaults: CatalogDefaults) -> tuple[list[Path], dict[Path, str], dict[Path, Path]]:
    fetched = fetch_source(spec, cache_dir, defaults)
    extracted: list[Path] = []
    role_by_path: dict[Path, str] = {}
    root_by_path: dict[Path, Path] = {}

    for f in fetched:
        # extract.archive decides whether this path still needs unpacking --
        # independent of source.kind, which only decides whether it needed
        # downloading. A local zip still needs extracting; a local
        # already-materialized directory/file does not -- and in that case
        # relative-path resolution (for glob/spec_files split matching) must
        # be rooted at the local source dir itself, not the (unused) extract
        # dir, since nothing was copied there.
        if spec.extract.archive == "none":
            if f.path.is_dir():
                files = _resolve_local_files(f.path, spec.extract)
                root = f.path
            else:
                files = [f.path]
                root = f.path.parent
        else:
            files = extract_archive(f.path, spec.extract, extract_dir)
            root = extract_dir
        extracted.extend(files)
        for p in files:
            role_by_path[p] = f.role
            root_by_path[p] = root

    return extracted, role_by_path, root_by_path


def _check_expect(report: IngestReport, expect: dict) -> None:
    n_train, n_test = report.x_train.shape[0], report.x_test.shape[0]
    if "n_train" in expect and n_train != expect["n_train"]:
        report.warnings.append(f"expected n_train={expect['n_train']}, got {n_train}")
    if "n_test" in expect and n_test != expect["n_test"]:
        report.warnings.append(f"expected n_test={expect['n_test']}, got {n_test}")
    if "n_rows" in expect and (n_train + n_test) != expect["n_rows"]:
        report.warnings.append(f"expected n_rows={expect['n_rows']}, got {n_train + n_test}")
    if "n_classes" in expect and report.y_train is not None:
        n_classes = len(np.unique(np.concatenate([report.y_train, report.y_test])))
        if n_classes != expect["n_classes"]:
            report.warnings.append(f"expected n_classes={expect['n_classes']}, got {n_classes}")


def _normalize_labels(report: IngestReport) -> None:
    if report.y_train is None:
        return
    ys = [report.y_train]
    if report.y_test is not None:
        ys.append(report.y_test)
    offset = int(min(y.min() for y in ys if y.size))
    if offset != 0:
        report.y_train = report.y_train - offset
        if report.y_test is not None:
            report.y_test = report.y_test - offset
        report.label_offset = offset
        report.warnings.append(f"labels were not 0-indexed; shifted by -{offset} (raw min was {offset})")


def _export(report: IngestReport, spec: RawDataSourceSpec, output_dir: Path) -> None:
    dest = spec.export.path.format(output_dir=output_dir, key=spec.key)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if "npz" in spec.export.formats:
        npz_path = dest_path.with_suffix(".npz")
        save_kwargs = {"x_train": report.x_train, "x_test": report.x_test}
        if report.y_train is not None:
            save_kwargs["y_train"] = report.y_train
        if report.y_test is not None:
            save_kwargs["y_test"] = report.y_test
        np.savez_compressed(npz_path, **save_kwargs)
        report.output_paths["npz"] = npz_path

    if "csv" in spec.export.formats:
        for split_name, X, y in (("train", report.x_train, report.y_train), ("test", report.x_test, report.y_test)):
            csv_path = dest_path.parent / f"{dest_path.name}_{split_name}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                for i, row in enumerate(X):
                    out_row = list(row)
                    if y is not None:
                        out_row.append(int(y[i]))
                    w.writerow(out_row)
            report.output_paths[f"csv_{split_name}"] = csv_path

    # "keep_files" -- nothing to do; extracted per-file layout is never
    # deleted, so it's already "kept" alongside the cache/extract dirs.


def run_ingest(spec: RawDataSourceSpec, output_dir: Path, defaults: "Optional[CatalogDefaults]" = None, cache_dir: "Optional[Path]" = None) -> IngestReport:
    if not spec.enabled:
        raise ValueError(f"dataset {spec.key!r} is disabled (enabled: false)")

    defaults = defaults or CatalogDefaults()
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir else output_dir / "_cache" / spec.key
    extract_dir = output_dir / "_extracted" / spec.key

    extracted, role_by_path, root_by_path = _fetch_and_extract(spec, cache_dir, extract_dir, defaults)
    reader = READERS[spec.parse.reader]

    if spec.split.mode == "random":
        pool = extracted
        if spec.split.source_member:
            pool = [p for p in extracted if fnmatch.fnmatch(p.name, spec.split.source_member) or p.name == spec.split.source_member]
            if not pool:
                raise FileNotFoundError(f"split.source_member={spec.split.source_member!r} matched no extracted files")
        X, y, _meta = reader(pool, spec.parse)
        x_train, x_test, y_train, y_test = split_random(X, y, spec.split)
    else:
        if spec.split.spec_files:
            train_paths, test_paths = resolve_spec_files_split(spec.split.spec_files, extracted, root_by_path)
        else:
            train_paths = resolve_predefined_group(spec.split.train, extracted, root_by_path, role_by_path)
            test_paths = resolve_predefined_group(spec.split.test, extracted, root_by_path, role_by_path)
        x_train, y_train, _ = reader(train_paths, spec.parse)
        x_test, y_test, _ = reader(test_paths, spec.parse)

    report = IngestReport(key=spec.key, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    _normalize_labels(report)
    _check_expect(report, spec.expect)
    _export(report, spec, output_dir)

    for w in report.warnings:
        _LOGGER.warning("[%s] %s", spec.key, w)

    return report
