"""Dataset registry — maps --dataset flag values to catalog entries.

Mirrors matador.backends.registry's name -> implementation lookup, except
here "implementation" is pure data (a RawDataSourceSpec), so this is a
thin name-based lookup over an already-parsed RawDataCatalog rather than a
lazy Python-module loader.

DEFAULT_CATALOG resolves relative to the installed matador package's own
location, not the current working directory — this works both in a local
dev checkout (matador/__init__.py's parent.parent is the repo root) and
inside the Docker container, where `make shell` bind-mounts the repo root
at /workspace via the exact same mechanism (matador/__init__.py lives at
/workspace/matador/__init__.py there too).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

import matador
from matador.preprocessing.sources import RawDataCatalog

if TYPE_CHECKING:
    from matador.config.schema import BooleanizationRecipe
    from matador.preprocessing.sources import RawDataSourceSpec

DEFAULT_CATALOG = Path(matador.__file__).resolve().parent.parent / "data" / "Raw_Data_Bank.yaml"


def _load_catalog(catalog_path: Path) -> RawDataCatalog:
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Dataset catalog not found: {catalog_path}\n"
            "  Expected data/Raw_Data_Bank.yaml relative to the matador package "
            "install location — pass catalog_path explicitly if yours lives elsewhere."
        )
    return RawDataCatalog.model_validate(yaml.safe_load(catalog_path.read_text()))


def list_datasets(catalog_path: Path = DEFAULT_CATALOG) -> list[str]:
    """Return sorted list of registered dataset keys."""
    return sorted(ds.key for ds in _load_catalog(catalog_path).datasets)


def get_dataset(name: str, catalog_path: Path = DEFAULT_CATALOG) -> "RawDataSourceSpec":
    """Return the RawDataSourceSpec for the given dataset key.

    Raises:
        KeyError: if the name is not registered (via RawDataCatalog.get()).
    """
    return _load_catalog(catalog_path).get(name)


def get_dataset_and_defaults(name: str, catalog_path: Path = DEFAULT_CATALOG):
    """Same as resolve_source_config()'s return shape (spec, defaults), for
    CLI code paths that need to treat --dataset and --config symmetrically."""
    catalog = _load_catalog(catalog_path)
    return catalog.get(name), catalog.defaults


def describe_dataset(name: str, catalog_path: Path = DEFAULT_CATALOG) -> str:
    """One-line description: display name + whether a VERIFIED default
    booleanization recipe is available. --show-recipe works either way (see
    get_recipe_for_inspection) — this only distinguishes whether it's safe
    to run --dataset NAME straight (without --show-recipe first)."""
    ds = get_dataset(name, catalog_path)
    label = ds.name or ds.key
    if ds.booleanization is not None:
        return f"{label} (has a verified default booleanization recipe)"
    return f"{label} (no verified default — inspect an unverified skeleton with --show-recipe)"


def get_default_booleanization(name: str, catalog_path: Path = DEFAULT_CATALOG) -> "Optional[BooleanizationRecipe]":
    """Return the dataset's VERIFIED default booleanization recipe, or None
    if it doesn't have one (several catalog entries deliberately don't —
    see their own notes/booleanised.encoding for why: a feature-engineering
    step this pipeline doesn't implement sits between raw ingest output
    and their committed Boolean shape). This is what `matador booleanize
    --dataset NAME` actually runs — it never silently applies an unverified
    guess; see get_recipe_for_inspection() for --show-recipe's looser
    always-return-something behavior."""
    return get_dataset(name, catalog_path).booleanization


def _generic_skeleton_recipe() -> "BooleanizationRecipe":
    """A generic, unvalidated starting point for datasets with no catalog-
    recorded default: an 8-bit quantile-fit thermometer applied uniformly
    (no assumed value range, since none is known) — reasonable for numeric
    raw features in general, but NOT checked to reproduce any particular
    documented Boolean shape the way a real default is."""
    from matador.config.schema import BooleanizationRecipe, FeatureEncoderSpec

    return BooleanizationRecipe(
        default_encoder=FeatureEncoderSpec(encoder="thermometer", bits=8, quantile=True),
    )


def get_recipe_for_inspection(name: str, catalog_path: Path = DEFAULT_CATALOG) -> "tuple[BooleanizationRecipe, bool]":
    """Return (recipe, is_verified) for --show-recipe: every registered
    dataset has SOMETHING to inspect and edit, never just a hard error.

    is_verified=True: the catalog's own recorded default (get_default_
    booleanization() — reproduces the documented Boolean shape exactly).
    is_verified=False: no verified default exists; a generic unvalidated
    skeleton is returned instead, meant purely as an editable starting
    point (see the dataset's own `notes` for why nothing verified exists).

    Raises:
        KeyError: if `name` is not a registered dataset at all.
    """
    ds = get_dataset(name, catalog_path)  # raises KeyError for an unknown name
    if ds.booleanization is not None:
        return ds.booleanization, True
    return _generic_skeleton_recipe(), False


def resolve_booleanised_files(name: str, catalog_path: Path = DEFAULT_CATALOG) -> "list[Path] | None":
    """Resolve a dataset's ALREADY-COMMITTED booleanised train/test files
    (booleanised.files — the pre-made copy shipped in the repo's data/ dir,
    e.g. from data.zip) to real paths, relative to the catalog file's own
    location (data/Raw_Data_Bank.yaml's parent is data/, whose parent is the
    repo root). Returns None if the catalog entry has no booleanised.files
    listed. The returned paths are NOT guaranteed to exist — several
    datasets' per-dataset archives (data/<Name>.zip) aren't pre-extracted in
    every checkout; check with Path.exists()."""
    ds = get_dataset(name, catalog_path)
    files = ds.booleanised.get("files")
    if not files:
        return None
    data_dir = catalog_path.resolve().parent
    return [data_dir / f for f in files]


def booleanised_files_exist(name: str, catalog_path: Path = DEFAULT_CATALOG) -> bool:
    """True if this dataset's pre-made booleanised files are already
    extracted on disk (not just described in the catalog)."""
    files = resolve_booleanised_files(name, catalog_path)
    return bool(files) and all(p.exists() for p in files)
