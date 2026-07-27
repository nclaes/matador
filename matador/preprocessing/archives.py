"""Stdlib-only archive extraction: zip / tar.gz / gzip, glob member
selection, and one level of nested-archive unpacking (extract.inner_archive
— needed for human_activity's zip-inside-zip)."""

from __future__ import annotations

import fnmatch
import gzip
import shutil
import tarfile
import zipfile
from pathlib import Path

from matador.preprocessing.sources import ExtractSpec


def _select_members(names: list[str], patterns: list[str]) -> list[str]:
    """Glob-match archive member names against extract.members patterns.
    An empty pattern list means "keep everything"."""
    if not patterns:
        return names
    selected: list[str] = []
    for name in names:
        if any(fnmatch.fnmatch(name, pat) for pat in patterns):
            selected.append(name)
    return selected


def extract_archive(archive_path: Path, spec: ExtractSpec, dest_dir: Path) -> list[Path]:
    """Extract archive_path into dest_dir per spec, returning the extracted
    file paths (post-member-filtering, post-inner_archive unpacking)."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    # spec.members always describes the FINAL files wanted, which live
    # inside inner_archive when one is set -- the outer archive only needs
    # to yield that one nested-archive member, not be filtered against
    # patterns meant for its contents.
    outer_patterns = [spec.inner_archive] if spec.inner_archive else spec.members

    if spec.archive == "zip":
        extracted = _extract_zip(archive_path, outer_patterns, dest_dir)
    elif spec.archive == "tar.gz":
        extracted = _extract_targz(archive_path, outer_patterns, dest_dir)
    elif spec.archive == "gzip":
        extracted = [_extract_gzip(archive_path, dest_dir)]
    else:
        raise ValueError(f"Unsupported archive type: {spec.archive!r}")

    if spec.inner_archive:
        inner_path = next((p for p in extracted if p.name == spec.inner_archive), None)
        if inner_path is None:
            raise FileNotFoundError(
                f"inner_archive {spec.inner_archive!r} not found among extracted members: "
                f"{[p.name for p in extracted]}"
            )
        inner_spec = ExtractSpec(archive="zip", members=spec.members)
        extracted = _extract_zip(inner_path, inner_spec.members, dest_dir)

    return extracted


def _extract_zip(archive_path: Path, patterns: list[str], dest_dir: Path) -> list[Path]:
    with zipfile.ZipFile(archive_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        keep = _select_members(names, patterns)
        out: list[Path] = []
        for name in keep:
            zf.extract(name, dest_dir)
            out.append(dest_dir / name)
        return out


def _extract_targz(archive_path: Path, patterns: list[str], dest_dir: Path) -> list[Path]:
    with tarfile.open(archive_path, "r:gz") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        names = [m.name for m in members]
        keep_names = set(_select_members(names, patterns))
        out: list[Path] = []
        for m in members:
            if m.name in keep_names:
                tf.extract(m, dest_dir)
                out.append(dest_dir / m.name)
        return out


def _extract_gzip(archive_path: Path, dest_dir: Path) -> Path:
    out_name = archive_path.name[:-3] if archive_path.name.endswith(".gz") else archive_path.stem
    out_path = dest_dir / out_name
    with gzip.open(archive_path, "rb") as src, open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path
