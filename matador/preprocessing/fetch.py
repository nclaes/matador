"""Downloads a RawDataSourceSpec's source.urls[] (or resolves a local path,
for source.kind == "local"), with retries, a mirrors[] fallback, and
checksum handling per the catalog's checksum_policy."""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from matador.preprocessing.sources import CatalogDefaults, RawDataSourceSpec, UrlSpec

_LOGGER = logging.getLogger(__name__)


@dataclass
class FetchedFile:
    path: Path
    role: str
    url: UrlSpec


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(url: str, dest: Path, defaults: CatalogDefaults) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": defaults.user_agent})
    last_exc: Exception | None = None
    for attempt in range(1, defaults.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=defaults.timeout_s) as resp, open(dest, "wb") as out:
                out.write(resp.read())
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            _LOGGER.warning("download attempt %d/%d failed for %s: %s", attempt, defaults.retries, url, exc)
            if attempt < defaults.retries:
                time.sleep(min(2 ** attempt, 10))
    raise ConnectionError(f"Failed to download {url} after {defaults.retries} attempt(s): {last_exc}")


def _download_with_mirrors(url_spec: UrlSpec, mirrors: list[str], dest: Path, defaults: CatalogDefaults) -> None:
    try:
        _download_one(url_spec.url, dest, defaults)
        return
    except ConnectionError as primary_exc:
        for mirror in mirrors:
            mirror_url = mirror.rstrip("/") + "/" + url_spec.filename
            try:
                _LOGGER.warning("primary URL failed, trying mirror: %s", mirror_url)
                _download_one(mirror_url, dest, defaults)
                return
            except ConnectionError:
                continue
        raise primary_exc


def _check_checksum(path: Path, url_spec: UrlSpec, policy: str) -> None:
    if url_spec.md5 is None:
        return
    actual = _md5(path)
    if actual == url_spec.md5:
        return
    msg = f"checksum mismatch for {path.name}: expected {url_spec.md5}, got {actual}"
    if policy == "enforce":
        raise ValueError(msg)
    elif policy == "warn":
        _LOGGER.warning(msg)
    # "record": silent — the value is treated as provisional per the catalog's own note


def fetch_source(spec: RawDataSourceSpec, cache_dir: Path, defaults: CatalogDefaults) -> list[FetchedFile]:
    """Materialize spec.source into local files under cache_dir.

    For source.kind == "local", no download happens — spec.source.path is
    resolved and returned directly.
    """
    if spec.source.kind == "local":
        path = Path(spec.source.path)
        if not path.exists():
            raise FileNotFoundError(f"source.kind == 'local' but path does not exist: {path}")
        return [FetchedFile(path=path, role="all", url=UrlSpec(url="", filename=path.name, role="all"))]

    cache_dir.mkdir(parents=True, exist_ok=True)
    out: list[FetchedFile] = []
    for url_spec in spec.source.urls:
        dest = cache_dir / url_spec.filename
        if not dest.exists():
            _download_with_mirrors(url_spec, spec.source.mirrors, dest, defaults)
        _check_checksum(dest, url_spec, defaults.checksum_policy)
        out.append(FetchedFile(path=dest, role=url_spec.role, url=url_spec))
    return out
