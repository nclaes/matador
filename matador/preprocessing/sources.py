"""Raw-data source specification — mirrors the schema already established by
data/Raw_Data_Bank.yaml (see that file's own "FIELD REFERENCE" header), plus
one addition: ``source.kind == "local"`` for raw data that is already on
disk, needing no fetch/extract step at all.

Two ways a user points Matador at raw data:

  1. A standalone single-dataset file with the same shape as one entry under
     a catalog's ``datasets:`` list (no wrapper) — this is the per-project
     ``data_source_config.yaml`` a user fills out by hand.
  2. A pointer into a shared multi-dataset catalog shaped like
     data/Raw_Data_Bank.yaml itself: ``{"catalog": "...", "key": "..."}``.

``resolve_source_config()`` accepts either.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Base(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class UrlSpec(_Base):
    url: str
    filename: str
    role: str = "all"
    md5: Optional[str] = None
    approx_size: Optional[str] = None


class SourceSpec(_Base):
    kind: Literal["http_zip", "http_tar", "http_files", "local"]
    homepage: Optional[str] = None
    urls: list[UrlSpec] = Field(default_factory=list)
    mirrors: list[str] = Field(default_factory=list)
    api: Optional[str] = None
    path: Optional[Path] = None   # required when kind == "local"

    @model_validator(mode="after")
    def _kind_requirements(self) -> "SourceSpec":
        if self.kind == "local":
            if self.path is None:
                raise ValueError("source.kind == 'local' requires source.path")
        elif not self.urls:
            raise ValueError(f"source.kind == {self.kind!r} requires at least one entry in source.urls")
        return self


class ExtractSpec(_Base):
    archive: Literal["none", "zip", "tar.gz", "gzip"] = "none"
    inner_archive: Optional[str] = None
    members: list[str] = Field(default_factory=list)


class ParseSpec(_Base):
    reader: Literal["csv", "dir_of_csv", "dir_of_txt", "idx", "cifar_pickle", "wav_dir"]
    delimiter: str = ","                       # "whitespace" means "any run of whitespace"
    header: bool = False
    na_values: list[str] = Field(default_factory=list)
    label_column: Optional[int] = None
    label_map: dict[str, int] = Field(default_factory=dict)
    label_from_path: Optional[str] = None       # regex with one capture group
    label_file: bool = False                    # y lives in a sibling *_y*/y_* file, not a column
    drop_columns: list[Union[int, str]] = Field(default_factory=list)
    column_names: list[str] = Field(default_factory=list)
    dtype: str = "float32"
    relabel: dict[int, dict[str, list[int]]] = Field(default_factory=dict)   # cifar_pickle superclassing
    to_greyscale: bool = False                  # cifar_pickle only
    sample_rate: Optional[int] = None            # wav_dir only
    bit_depth: Optional[int] = None
    channels: Optional[int] = None


class SplitSpec(_Base):
    mode: Literal["predefined", "random", "by_member"]
    train: list[str] = Field(default_factory=list)   # glob patterns OR source.urls[].role names
    test: list[str] = Field(default_factory=list)
    source_member: Optional[str] = None               # restrict the input pool to one extracted member
    spec_files: list[str] = Field(default_factory=list)  # line-lists of held-out relative paths
    test_size: float = 0.2
    seed: int = 0
    stratify: bool = False


class ExportSpec(_Base):
    formats: list[Literal["npz", "csv", "keep_files"]] = Field(default_factory=lambda: ["npz"])
    path: str = "{output_dir}/{key}"


class StatusSpec(_Base):
    downloadable: str = "yes"
    direct_url: bool = True
    checked: Optional[Union[str, date]] = None


class RawDataSourceSpec(_Base):
    key: str
    name: str = ""
    enabled: bool = True
    status: StatusSpec = Field(default_factory=StatusSpec)
    source: SourceSpec
    extract: ExtractSpec = Field(default_factory=ExtractSpec)
    parse: ParseSpec
    split: SplitSpec
    expect: dict[str, Any] = Field(default_factory=dict)
    export: ExportSpec = Field(default_factory=ExportSpec)
    booleanised: dict[str, Any] = Field(default_factory=dict)
    license: Optional[str] = None
    citation: Optional[str] = None
    notes: Optional[str] = None


class CatalogDefaults(_Base):
    cache_dir: str = "./_cache"
    output_dir: str = "./raw"
    output_format: str = "npz"
    checksum_policy: Literal["record", "warn", "enforce"] = "record"
    user_agent: str = "matador-fetch/1.0"
    retries: int = 3
    timeout_s: int = 120


class RawDataCatalog(_Base):
    schema_version: int = 1
    defaults: CatalogDefaults = Field(default_factory=CatalogDefaults)
    datasets: list[RawDataSourceSpec]

    def get(self, key: str) -> RawDataSourceSpec:
        for ds in self.datasets:
            if ds.key == key:
                return ds
        available = ", ".join(sorted(d.key for d in self.datasets))
        raise KeyError(f"No dataset with key {key!r} in this catalog. Available: {available}")


def resolve_source_config(path: Path) -> tuple[RawDataSourceSpec, CatalogDefaults]:
    """Load a data_source_config.yaml, returning (spec, defaults).

    Accepts either a standalone single-dataset file, or a {"catalog": ...,
    "key": ...} pointer into a shared catalog (e.g. data/Raw_Data_Bank.yaml).
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())

    if isinstance(raw, dict) and "catalog" in raw and "key" in raw:
        catalog_path = Path(raw["catalog"])
        if not catalog_path.is_absolute():
            catalog_path = (path.parent / catalog_path).resolve()
        catalog = RawDataCatalog.model_validate(yaml.safe_load(catalog_path.read_text()))
        return catalog.get(raw["key"]), catalog.defaults

    if isinstance(raw, dict) and "datasets" in raw:
        raise ValueError(
            f"{path} looks like a multi-dataset catalog (has a top-level 'datasets:' "
            "list). Point a data_source_config.yaml at it with {'catalog': ..., 'key': ...} "
            "instead of passing the catalog itself."
        )

    spec = RawDataSourceSpec.model_validate(raw)
    return spec, CatalogDefaults()
