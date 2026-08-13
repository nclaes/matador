from pathlib import Path

import pytest
from pydantic import ValidationError

from coal_tm.config import RTLConfig, TrainingConfig


def test_rtl_config_accepts_a_well_formed_tiny_model(tiny_model, tmp_path):
    config = RTLConfig(
        output_dir=tmp_path / "out",
        tas=tiny_model["tas"],
        weights=tiny_model["weights"],
        classes=tiny_model["classes"],
        clauses=tiny_model["clauses"],
        features=tiny_model["features"],
        bus_width=8,
    )
    assert config.clauses == 2


def test_rtl_config_rejects_wrong_sized_tas_file(tiny_model, tmp_path):
    with pytest.raises(ValidationError, match="expected"):
        RTLConfig(
            output_dir=tmp_path / "out",
            tas=tiny_model["tas"],
            weights=tiny_model["weights"],
            classes=tiny_model["classes"],
            clauses=tiny_model["clauses"] + 1,  # now mismatched
            features=tiny_model["features"],
            bus_width=8,
        )


def test_rtl_config_rejects_adder_stages_that_do_not_divide_clauses(tiny_model, tmp_path):
    with pytest.raises(ValidationError, match="divisible"):
        RTLConfig(
            output_dir=tmp_path / "out",
            tas=tiny_model["tas"],
            weights=tiny_model["weights"],
            classes=tiny_model["classes"],
            clauses=tiny_model["clauses"],
            features=tiny_model["features"],
            bus_width=8,
            adder_stages=3,  # clauses=2, not divisible by 3
        )


def test_rtl_config_rejects_bus_width_that_fits_all_features_in_one_packet(tiny_model, tmp_path):
    """A model whose features all fit in a single packet gives the argmax
    pipeline no natural gap to drain between back-to-back inferences --
    a known, real limitation (see coal_tm/config.py), not just a slow path."""
    with pytest.raises(ValidationError, match="single packet"):
        RTLConfig(
            output_dir=tmp_path / "out",
            tas=tiny_model["tas"],
            weights=tiny_model["weights"],
            classes=tiny_model["classes"],
            clauses=tiny_model["clauses"],
            features=tiny_model["features"],
            bus_width=tiny_model["features"],  # == features, not <
        )


def test_rtl_config_rejects_missing_files(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        RTLConfig(
            output_dir=tmp_path / "out",
            tas=tmp_path / "nope.txt",
            weights=tmp_path / "also_nope.txt",
            classes=2, clauses=2, features=3, bus_width=8,
        )


def test_training_config_rejects_identical_train_and_test_files(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("0 0 0 0\n1 1 1 1\n")
    with pytest.raises(ValidationError, match="same file"):
        TrainingConfig(
            clauses=2, classes=2, features=3, s=3.0, T=15, epochs=1,
            max_included_literals=6,
            training_data=data, test_data=data,
            output_dir=tmp_path / "out",
        )


def test_training_config_rejects_max_included_literals_exceeding_2x_features(tmp_path):
    train = tmp_path / "train.txt"
    test = tmp_path / "test.txt"
    train.write_text("0 0 0 0\n")
    test.write_text("1 1 1 1\n")
    with pytest.raises(ValidationError, match="max_included_literals"):
        TrainingConfig(
            clauses=2, classes=2, features=3, s=3.0, T=15, epochs=1,
            max_included_literals=7,  # > 2*3
            training_data=train, test_data=test,
            output_dir=tmp_path / "out",
        )
