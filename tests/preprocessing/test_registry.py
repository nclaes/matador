"""Tests for matador.preprocessing.registry — the dataset registry
(--dataset flag on `matador ingest`/`matador booleanize`), mirroring
matador.backends.registry's name -> implementation lookup.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from matador.cli import main
from matador.preprocessing import registry

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Registry module
# ---------------------------------------------------------------------------

def test_default_catalog_resolves_relative_to_package_install():
    assert registry.DEFAULT_CATALOG == REPO_ROOT / "data" / "Raw_Data_Bank.yaml"
    assert registry.DEFAULT_CATALOG.exists()


def test_list_datasets_returns_all_catalog_entries():
    names = registry.list_datasets()
    assert names == sorted(names)
    assert set(names) == {
        "digits", "sports", "statlog", "gesture_phase", "human_activity",
        "mammographic", "emg", "sensorless_drive", "gas_sensor", "mnist",
    }


def test_get_dataset_returns_matching_spec():
    spec = registry.get_dataset("digits")
    assert spec.key == "digits"
    assert spec.parse.reader == "csv"


def test_get_dataset_raises_key_error_with_available_list():
    with pytest.raises(KeyError, match="Available:"):
        registry.get_dataset("not_a_real_dataset")


def test_get_dataset_and_defaults_matches_resolve_source_config_shape(tmp_path):
    from matador.preprocessing.sources import resolve_source_config

    ptr = tmp_path / "data_source_config.yaml"
    ptr.write_text(yaml.safe_dump({"catalog": str(registry.DEFAULT_CATALOG), "key": "digits"}))
    via_config = resolve_source_config(ptr)
    via_registry = registry.get_dataset_and_defaults("digits")

    assert via_config[0].key == via_registry[0].key
    assert via_config[1] == via_registry[1]


@pytest.mark.parametrize("name,has_recipe", [
    ("digits", True), ("statlog", True), ("mammographic", True),
    ("sensorless_drive", True), ("mnist", True), ("gas_sensor", True),
    ("sports", False), ("gesture_phase", False), ("human_activity", False),
    ("emg", False),
])
def test_get_default_booleanization_matches_expected_availability(name, has_recipe):
    recipe = registry.get_default_booleanization(name)
    assert (recipe is not None) == has_recipe


def test_describe_dataset_mentions_recipe_availability():
    assert "verified default booleanization recipe" in registry.describe_dataset("digits")
    assert "no verified default" in registry.describe_dataset("sports")


@pytest.mark.parametrize("name,expected_verified", [
    ("digits", True), ("statlog", True), ("mammographic", True),
    ("sensorless_drive", True), ("mnist", True), ("gas_sensor", True),
    ("sports", False), ("gesture_phase", False), ("human_activity", False),
    ("emg", False),
])
def test_get_recipe_for_inspection_always_returns_something(name, expected_verified):
    """Every registered dataset must have SOMETHING to inspect via
    --show-recipe -- verified datasets return their real default; the rest
    fall back to a generic unvalidated skeleton, never None/an error."""
    recipe, is_verified = registry.get_recipe_for_inspection(name)
    assert is_verified == expected_verified
    assert recipe.default_encoder is not None or recipe.features


def test_get_recipe_for_inspection_unverified_skeleton_is_generic_and_consistent():
    """The unverified skeleton must be the SAME generic shape across every
    dataset that lacks a verified default -- it's not dataset-specific
    (that's exactly why it's unverified)."""
    skeletons = [registry.get_recipe_for_inspection(n)[0] for n in ("sports", "gesture_phase", "human_activity")]
    for s in skeletons:
        assert s.default_encoder.encoder == "thermometer"
        assert s.default_encoder.quantile is True
        assert s.default_encoder.range is None   # no assumed value range -- none is known


def test_get_recipe_for_inspection_raises_for_unknown_dataset():
    with pytest.raises(KeyError, match="Available:"):
        registry.get_recipe_for_inspection("not_a_real_dataset")


# ---------------------------------------------------------------------------
# Booleanization recipes actually produce the documented bit width
# (a real correctness check, not just "the field is set")
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,n_features,expected_bits", [
    ("digits", 64, 512), ("statlog", 18, 288), ("mammographic", 5, 40),
    ("sensorless_drive", 48, 480), ("mnist", 784, 784),
    ("gas_sensor", 128, 128),
])
def test_default_recipe_produces_documented_bit_width(tmp_path, name, n_features, expected_bits):
    from matador.config.schema import BooleanisationConfig
    from matador.preprocessing.booleanize import run_booleanize

    recipe = registry.get_default_booleanization(name)
    rng = np.random.default_rng(0)
    x_train = rng.integers(0, 17, size=(10, n_features)).astype(float)
    x_test = rng.integers(0, 17, size=(4, n_features)).astype(float)
    y_train, y_test = rng.integers(0, 2, size=10), rng.integers(0, 2, size=4)
    npz = tmp_path / f"{name}.npz"
    np.savez(npz, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    cfg = BooleanisationConfig(
        raw_npz=npz, name=name, output_dir=tmp_path / "out",
        features=recipe.features, default_encoder=recipe.default_encoder,
    )
    report = run_booleanize(cfg)
    assert report.n_features_bool == expected_bits


# ---------------------------------------------------------------------------
# CLI: --dataset flag
# ---------------------------------------------------------------------------

def test_cli_list_datasets_shows_every_entry():
    result = CliRunner().invoke(main, ["list-datasets"])
    assert result.exit_code == 0
    for name in registry.list_datasets():
        assert name in result.output


def test_cli_ingest_rejects_both_dataset_and_config(tmp_path):
    cfg = tmp_path / "data_source_config.yaml"
    cfg.write_text("key: x\n")
    result = CliRunner().invoke(main, ["ingest", "--dataset", "digits", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "either --dataset or --config" in result.output


def test_cli_ingest_unknown_dataset_reports_available():
    result = CliRunner().invoke(main, ["ingest", "--dataset", "not_a_real_dataset"])
    assert result.exit_code != 0
    assert "Available:" in result.output


def test_cli_booleanize_rejects_both_dataset_and_config(tmp_path):
    cfg = tmp_path / "booleanisation_config.yaml"
    cfg.write_text("name: x\n")
    result = CliRunner().invoke(main, ["booleanize", "--dataset", "digits", "--config", str(cfg)])
    assert result.exit_code != 0
    assert "either --dataset or --config" in result.output


def test_cli_booleanize_dataset_without_recipe_gives_clear_error():
    result = CliRunner().invoke(main, ["booleanize", "--dataset", "sports"])
    assert result.exit_code != 0
    assert "no VERIFIED default booleanization recipe" in result.output
    assert "--show-recipe" in result.output


def test_cli_booleanize_dataset_missing_raw_npz_tells_user_to_ingest_first(tmp_path):
    result = CliRunner().invoke(main, [
        "booleanize", "--dataset", "digits", "--raw-dir", str(tmp_path / "nonexistent_raw"),
    ])
    assert result.exit_code != 0
    assert "matador ingest --dataset digits" in result.output


def _do_booleanize(tmp_path, out_subdir="bool"):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    x_train = np.random.default_rng(0).integers(0, 17, size=(10, 5)).astype(float)
    x_test = np.random.default_rng(1).integers(0, 17, size=(4, 5)).astype(float)
    y_train, y_test = np.array([0] * 10), np.array([1] * 4)
    np.savez(raw_dir / "digits.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    return CliRunner().invoke(main, [
        "booleanize", "--dataset", "digits", "--raw-dir", str(raw_dir), "--output-dir", str(tmp_path / out_subdir),
    ])


def test_cli_booleanize_tells_user_to_create_training_config_when_missing(tmp_path, monkeypatch):
    """Regression: matador booleanize's own success message used to always
    say "point training_config.yaml at these files" and jump straight to
    `matador train --config /work/training_config.yaml`, silently assuming
    that file already existed -- a first-time user running ingest ->
    booleanize has no training_config.yaml yet, and was never told to
    `cp examples/training_config.yaml` first."""
    import matador.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", tmp_path / "training_config.yaml")

    result = _do_booleanize(tmp_path)
    assert result.exit_code == 0
    assert "cp examples/training_config.yaml" in result.output
    assert str(tmp_path / "training_config.yaml") in result.output


def test_cli_booleanize_skips_cp_hint_when_training_config_already_targets_this_dataset(tmp_path, monkeypatch):
    """Re-booleanizing a dataset whose training_config.yaml already points
    train_data at it (e.g. re-running after an upstream data fix) should
    offer to reuse that same config, not suggest a fresh copy."""
    import matador.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", tmp_path / "training_config.yaml")
    (tmp_path / "training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {tmp_path / 'bool' / 'digits_train.txt'}\n"
    )

    result = _do_booleanize(tmp_path)
    assert result.exit_code == 0
    assert "cp examples/training_config.yaml" not in result.output
    assert f"matador train --config {tmp_path / 'training_config.yaml'}" in result.output


def test_cli_booleanize_respects_explicit_model_name_override_not_just_train_data(tmp_path, monkeypatch):
    """A training_config.yaml with an explicit model_name: (e.g. a
    differently-sized second model trained from the SAME dataset) must be
    recognized by its model_name, not misread as targeting the plain
    dataset name just because train_data happens to point at the same
    <name>_train.txt -- otherwise a second, differently-configured model for
    the same dataset would look identical to the first and get its config
    silently suggested for reuse/overwrite."""
    import matador.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", tmp_path / "training_config.yaml")
    (tmp_path / "training_config.yaml").write_text(
        f"tm_type: vanilla\nmodel_name: digits_large\n"
        f"train_data: {tmp_path / 'bool' / 'digits_train.txt'}\nclauses: 2000\n"
    )

    result = _do_booleanize(tmp_path)   # booleanizes "digits" again
    assert result.exit_code == 0
    assert "already trains digits_large" in result.output
    assert f"cp examples/training_config.yaml {tmp_path / 'digits_training_config.yaml'}" in result.output


def test_cli_booleanize_suggests_separate_config_for_a_different_dataset(tmp_path, monkeypatch):
    """Regression: booleanizing a second dataset while training_config.yaml
    already exists (and already targets a DIFFERENT model) used to say
    "edit training_config.yaml" unconditionally -- silently inviting the
    user to repurpose/overwrite a config that already belongs to another
    model. It must instead suggest a separate, non-colliding filename."""
    import matador.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", tmp_path / "training_config.yaml")
    (tmp_path / "training_config.yaml").write_text(
        "tm_type: vanilla\ntrain_data: /work/booleanised/sports_train.txt\n"
    )

    result = _do_booleanize(tmp_path)   # _do_booleanize always booleanizes "digits"
    assert result.exit_code == 0
    assert "already trains sports" in result.output
    assert f"cp examples/training_config.yaml {tmp_path / 'digits_training_config.yaml'}" in result.output
    assert f"matador train --config {tmp_path / 'digits_training_config.yaml'}" in result.output
    # must NOT silently tell the user to overwrite the existing one
    assert f"Edit {tmp_path / 'training_config.yaml'} and set" not in result.output


def test_cli_booleanize_tells_user_the_actual_class_count_and_to_review_hyperparameters(tmp_path, monkeypatch):
    """The guidance must give the real, computed class count (not leave the
    user to work it out themselves) and make clear that copying the
    template isn't enough on its own -- the other hyperparameters still
    need reviewing for this dataset."""
    import matador.cli as cli_mod

    monkeypatch.setattr(cli_mod, "_DEFAULT_CONFIG", tmp_path / "training_config.yaml")

    result = _do_booleanize(tmp_path)   # digits fixture: y_train=[0]*10, y_test=[1]*4 -> 2 classes
    assert result.exit_code == 0
    assert "classes:    2" in result.output
    assert "review the remaining hyperparameters" in result.output.lower()


# ---------------------------------------------------------------------------
# Network-gated end-to-end: --dataset reaches the same real code paths as
# the --config form already verified in test_ingest.py / test_booleanize.py
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("MATADOR_NETWORK_TESTS") != "1",
    reason="set MATADOR_NETWORK_TESTS=1 to run real-network ingestion tests",
)
def test_cli_dataset_flag_end_to_end_matches_committed_digits_shape(tmp_path):
    runner = CliRunner()

    r1 = runner.invoke(main, ["ingest", "--dataset", "digits", "--output-dir", str(tmp_path / "raw")])
    assert r1.exit_code == 0, r1.output

    r2 = runner.invoke(main, [
        "booleanize", "--dataset", "digits",
        "--raw-dir", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "bool"),
    ])
    assert r2.exit_code == 0, r2.output

    mine_train = np.genfromtxt(tmp_path / "bool" / "digits_train.txt", dtype=np.uint32)
    mine_test = np.genfromtxt(tmp_path / "bool" / "digits_test.txt", dtype=np.uint32)
    committed_train = np.genfromtxt(REPO_ROOT / "data" / "Digits" / "Digits_train.txt", dtype=np.uint32)
    committed_test = np.genfromtxt(REPO_ROOT / "data" / "Digits" / "Digits_test.txt", dtype=np.uint32)

    assert mine_train.shape == committed_train.shape == (1437, 513)
    assert mine_test.shape == committed_test.shape == (360, 513)


# ---------------------------------------------------------------------------
# --show-recipe: inspecting a registered dataset's default booleanization
# recipe, and using it as a starting point for a custom booleanisation_config.yaml.
# ---------------------------------------------------------------------------

def test_cli_list_datasets_shows_recipe_details_inline():
    result = CliRunner().invoke(main, ["list-datasets"])
    assert result.exit_code == 0
    assert "thermometer(bits=8, range=[0.0, 16.0])" in result.output   # digits
    assert "--show-recipe" in result.output
    # a dataset without a verified recipe must not claim to have one, but
    # must still point at --show-recipe (it works for these too now)
    assert "no verified default" in result.output


def test_cli_show_recipe_requires_dataset_flag():
    result = CliRunner().invoke(main, ["booleanize", "--show-recipe"])
    assert result.exit_code != 0
    assert "--show-recipe requires --dataset" in result.output


def test_cli_show_recipe_prints_valid_yaml_without_running_booleanize(tmp_path):
    result = CliRunner().invoke(main, [
        "booleanize", "--dataset", "digits", "--show-recipe",
        "--raw-dir", str(tmp_path / "raw"), "--output-dir", str(tmp_path / "bool"),
    ])
    assert result.exit_code == 0
    # nothing was actually written -- this is inspect-only
    assert not (tmp_path / "bool").exists()

    parsed = yaml.safe_load(result.output)
    assert parsed["raw_npz"] == str(tmp_path / "raw" / "digits.npz")
    assert parsed["name"] == "digits"
    assert parsed["output_dir"] == str(tmp_path / "bool")
    assert parsed["default_encoder"] == {
        "encoder": "thermometer", "bits": 8, "range": [0.0, 16.0], "quantile": False,
    }


def test_cli_show_recipe_notes_when_raw_npz_not_yet_ingested(tmp_path):
    result = CliRunner().invoke(main, [
        "booleanize", "--dataset", "digits", "--show-recipe", "--raw-dir", str(tmp_path / "raw"),
    ])
    assert result.exit_code == 0
    assert "does not exist yet" in result.output
    assert "matador ingest --dataset digits" in result.output


def test_cli_show_recipe_works_for_dataset_without_verified_default():
    """--show-recipe must always produce something to inspect/edit, even
    for datasets with no catalog-recorded default -- it must NOT error the
    way running --dataset straight (without --show-recipe) still does."""
    result = CliRunner().invoke(main, ["booleanize", "--dataset", "sports", "--show-recipe"])
    assert result.exit_code == 0
    assert "UNVERIFIED starting skeleton" in result.output
    # the dataset's own catalog notes must be surfaced, not just a generic message
    assert "19 activities" in result.output   # sports' own notes text

    parsed = yaml.safe_load(result.output)
    assert parsed["default_encoder"]["encoder"] == "thermometer"
    assert parsed["default_encoder"]["quantile"] is True


def test_cli_booleanize_dataset_without_verified_default_refuses_to_run():
    """The SAFE run path (no --show-recipe) must still refuse for a dataset
    with no verified default -- --show-recipe existing doesn't mean matador
    silently applies the unverified guess."""
    result = CliRunner().invoke(main, ["booleanize", "--dataset", "sports"])
    assert result.exit_code != 0
    assert "no VERIFIED default booleanization recipe" in result.output
    assert "--show-recipe" in result.output   # points at the way to inspect/edit one


def test_unverified_skeleton_round_trips_into_a_working_config(tmp_path):
    """The unverified skeleton --show-recipe prints must be just as usable
    as a verified one -- save it (optionally edited), run it via --config,
    and it works."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    x_train = np.random.default_rng(0).integers(0, 100, size=(10, 5)).astype(float)
    x_test = np.random.default_rng(1).integers(0, 100, size=(4, 5)).astype(float)
    y_train, y_test = np.array([0] * 10), np.array([1] * 4)
    np.savez(raw_dir / "sports.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    runner = CliRunner()
    shown = runner.invoke(main, [
        "booleanize", "--dataset", "sports", "--show-recipe",
        "--raw-dir", str(raw_dir), "--output-dir", str(tmp_path / "bool"),
    ])
    assert shown.exit_code == 0

    cfg_path = tmp_path / "my_booleanisation_config.yaml"
    cfg_path.write_text(shown.output)

    ran = runner.invoke(main, ["booleanize", "--config", str(cfg_path)])
    assert ran.exit_code == 0, ran.output
    assert (tmp_path / "bool" / "sports_train.txt").exists()


def test_show_recipe_output_round_trips_as_a_working_config(tmp_path):
    """The whole point: --show-recipe's output can be saved verbatim and
    fed back in via --config to actually run booleanization."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    x_train = np.random.default_rng(0).integers(0, 17, size=(10, 64)).astype(float)
    x_test = np.random.default_rng(1).integers(0, 17, size=(4, 64)).astype(float)
    y_train, y_test = np.array([0] * 10), np.array([1] * 4)
    np.savez(raw_dir / "digits.npz", x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    runner = CliRunner()
    shown = runner.invoke(main, [
        "booleanize", "--dataset", "digits", "--show-recipe",
        "--raw-dir", str(raw_dir), "--output-dir", str(tmp_path / "bool"),
    ])
    assert shown.exit_code == 0

    cfg_path = tmp_path / "my_booleanisation_config.yaml"
    cfg_path.write_text(shown.output)

    ran = runner.invoke(main, ["booleanize", "--config", str(cfg_path)])
    assert ran.exit_code == 0, ran.output
    assert (tmp_path / "bool" / "digits_train.txt").exists()

    train_txt = np.genfromtxt(tmp_path / "bool" / "digits_train.txt", dtype=np.uint32)
    assert train_txt.shape == (10, 513)   # 64 features x 8 bits + label column
