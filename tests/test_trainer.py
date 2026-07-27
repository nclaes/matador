"""Tests for matador.models.trainer — specifically export_tmir()'s per-model
output namespacing (<output_dir>/TMIR/<model_name>/), which fixes a real
collision: validation_config.yaml/ta_actions.npy/model_metadata.json/
rom_inference.py used to be fixed filenames, so every `matador train` run
overwrote the previous run's companion files regardless of dataset.

export_tmir() takes a raw TMU classifier and calls from_tmu() internally,
but the namespacing logic added here is orthogonal to TMU's internals — so
these tests stub from_tmu() with a pre-built, hand-constructed TMIR (the
same pattern tests/backends/test_gp_tiled.py already uses for RTL-side
tests) rather than requiring the real tmu C extension, which isn't always
buildable (e.g. non-Linux/non-x86 hosts, or before `make tmu-build`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datetime import datetime, timezone

from matador.config.schema import TrainingConfig
from matador.ir.tm_ir import Architecture, Hyperparameters, Provenance, Representation, TMIR
from matador.models import trainer

INCLUDE = 200
EXCLUDE = 50


def _make_fake_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4) -> TMIR:
    n_literals = 2 * n_features
    ta = np.full((n_classes, n_clauses_pc, n_literals), EXCLUDE, dtype=np.int32)
    ta[0, 0, 0] = INCLUDE
    ta[1, 0, 1] = INCLUDE
    return TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=n_literals,
            n_classes=n_classes, n_clauses_per_class=n_clauses_pc,
            n_clauses_total=n_classes * n_clauses_pc,
            clause_organization="per_class", threshold=threshold,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
        provenance=Provenance(
            framework="tmu", framework_version="0.0.0",
            trained_at=datetime.now(timezone.utc), epochs=1,
        ),
    )


def _write_boolean_data(path: Path, n_rows: int, n_features: int, n_classes: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_rows):
        feats = rng.integers(0, 2, size=n_features)
        label = rng.integers(0, n_classes)
        rows.append(" ".join(map(str, feats.tolist())) + f" {label}")
    path.write_text("\n".join(rows) + "\n")


def _make_config(tmp_path: Path, train_name: str, **overrides) -> TrainingConfig:
    n_features, n_classes = overrides.pop("n_features", 8), overrides.pop("n_classes", 2)
    train_data = tmp_path / "data" / train_name
    train_data.parent.mkdir(parents=True, exist_ok=True)
    _write_boolean_data(train_data, 20, n_features, n_classes, seed=0)
    test_data = train_data.with_name(train_name.replace("_train", "_test"))
    _write_boolean_data(test_data, 5, n_features, n_classes, seed=1)

    payload = dict(
        tm_type="vanilla", clauses=8, classes=n_classes, features=n_features,
        s=3.9, T=4, epochs=1, max_included_literals=n_features * 2,
        train_data=train_data, test_data=test_data, output_dir=tmp_path,
    )
    payload.update(overrides)
    return TrainingConfig.model_validate(payload)


def _export(monkeypatch, config: TrainingConfig, tmir: TMIR):
    monkeypatch.setattr(trainer, "from_tmu", lambda tm: tmir)
    return trainer.export_tmir(object(), config)


# ---------------------------------------------------------------------------
# The actual bug: two models with different datasets, same hyperparameters
# ---------------------------------------------------------------------------

def test_two_models_different_datasets_same_hyperparams_do_not_collide(tmp_path, monkeypatch):
    cfg_a = _make_config(tmp_path, "digits_train.txt")
    cfg_b = _make_config(tmp_path, "sports_train.txt")

    yaml_a, npz_a, val_a = _export(monkeypatch, cfg_a, _make_fake_tmir())
    yaml_b, npz_b, val_b = _export(monkeypatch, cfg_b, _make_fake_tmir())

    assert yaml_a.parent == tmp_path / "TMIR" / "digits"
    assert yaml_b.parent == tmp_path / "TMIR" / "sports"
    assert yaml_a != yaml_b
    assert val_a != val_b

    # Every companion file from both trainings must still exist after both
    # runs -- the second training must not have clobbered the first's.
    for d in (yaml_a.parent, yaml_b.parent):
        assert (d / "validation_config.yaml").exists()
        assert (d / "ta_actions.npy").exists()
        assert (d / "model_metadata.json").exists()
        assert (d / "rom_inference.py").exists()
    assert yaml_a.exists() and npz_a.exists()
    assert yaml_b.exists() and npz_b.exists()

    # And each validation_config.yaml must point at ITS OWN model, not the
    # other one's (the exact bug: a shared fixed filename used to mean the
    # second run's validation_config.yaml pointed at the second model even
    # when read via what looked like "the first model's" path).
    import yaml as _yaml
    val_a_data = _yaml.safe_load(val_a.read_text())
    val_b_data = _yaml.safe_load(val_b.read_text())
    assert val_a_data["model_path"] == str(yaml_a)
    assert val_b_data["model_path"] == str(yaml_b)


def test_model_name_derived_from_train_data_filename(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, "digits_train.txt")
    yaml_path, _, _ = _export(monkeypatch, cfg, _make_fake_tmir())
    assert yaml_path.parent.name == "digits"


@pytest.mark.parametrize("filename,expected", [
    ("digits_train.txt", "digits"),
    ("my-dataset-train.txt", "my-dataset"),
    ("train.txt", "train"),
])
def test_derive_model_name_handles_naming_variants(tmp_path, filename, expected):
    assert trainer._derive_model_name(Path(filename)) == expected


def test_explicit_model_name_overrides_derived_name(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, "digits_train.txt", model_name="my_custom_name")
    yaml_path, _, _ = _export(monkeypatch, cfg, _make_fake_tmir())
    assert yaml_path.parent.name == "my_custom_name"


def test_model_name_rejects_path_separators(tmp_path):
    with pytest.raises(Exception, match="path separators"):
        _make_config(tmp_path, "digits_train.txt", model_name="a/b")


def test_model_name_rejects_empty_string(tmp_path):
    with pytest.raises(Exception, match="non-empty"):
        _make_config(tmp_path, "digits_train.txt", model_name="   ")


# ---------------------------------------------------------------------------
# dataset_id provenance — previously a dead, never-populated schema field
# ---------------------------------------------------------------------------

def test_dataset_id_provenance_populated_from_model_name(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, "digits_train.txt")
    tmir = _make_fake_tmir()
    yaml_path, _, _ = _export(monkeypatch, cfg, tmir)

    reloaded = TMIR.from_yaml(yaml_path)
    assert reloaded.provenance.dataset_id == "digits"


def test_dataset_id_reflects_explicit_model_name_override(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, "digits_train.txt", model_name="custom")
    tmir = _make_fake_tmir()
    yaml_path, _, _ = _export(monkeypatch, cfg, tmir)

    reloaded = TMIR.from_yaml(yaml_path)
    assert reloaded.provenance.dataset_id == "custom"


# ---------------------------------------------------------------------------
# Intentional overwrite: same model_name (same dataset) retrained
# ---------------------------------------------------------------------------

def test_retraining_same_dataset_overwrites_in_place_not_an_error(tmp_path, monkeypatch):
    cfg = _make_config(tmp_path, "digits_train.txt")

    yaml_1, _, _ = _export(monkeypatch, cfg, _make_fake_tmir(threshold=4))
    yaml_2, _, _ = _export(monkeypatch, cfg, _make_fake_tmir(threshold=4))

    assert yaml_1 == yaml_2
    assert yaml_1.parent == tmp_path / "TMIR" / "digits"


# ---------------------------------------------------------------------------
# rom_inference.py's own companion files are usable standalone per model
# ---------------------------------------------------------------------------

def test_rom_inference_script_runs_standalone_per_model(tmp_path, monkeypatch):
    """The provenance script written alongside each model must work using
    ONLY that model's own directory -- proving the two models' companion
    files are genuinely independent, not accidentally sharing state."""
    import subprocess
    import sys

    cfg_a = _make_config(tmp_path, "digits_train.txt")
    cfg_b = _make_config(tmp_path, "sports_train.txt")
    _export(monkeypatch, cfg_a, _make_fake_tmir())
    yaml_b, _, _ = _export(monkeypatch, cfg_b, _make_fake_tmir())

    script_b = yaml_b.parent / "rom_inference.py"
    result = subprocess.run(
        [sys.executable, str(script_b), "--test-data", str(cfg_b.test_data)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
