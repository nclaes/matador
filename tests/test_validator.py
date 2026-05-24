"""Tests for matador.models.validator — end-to-end validation flow."""

from __future__ import annotations

import numpy as np
import pytest

from matador.config.schema import ValidationConfig
from matador.ir.tm_ir import (
    Architecture,
    EvalVector,
    Hyperparameters,
    Representation,
    TMIR,
    Verification,
)
from matador.models.validator import validate_model


# ---------------------------------------------------------------------------
# Shared fixture: a tiny but complete TMIR saved to disk
# ---------------------------------------------------------------------------

INCLUDE = 200
EXCLUDE = 50


def _make_tmir() -> TMIR:
    ta = np.full((2, 4, 4), EXCLUDE, dtype=np.int32)
    ta[0, 0, 0] = INCLUDE   # class-0 positive clause 0: includes f0
    ta[0, 1, 1] = INCLUDE   # class-0 positive clause 1: includes f1
    ta[0, 2, 2] = INCLUDE   # class-0 negative clause 2: includes NOT-f0
    ta[0, 3, 3] = INCLUDE   # class-0 negative clause 3: includes NOT-f1
    ta[1, 0, 0] = INCLUDE   # class-1 positive clause 0: includes f0
    ta[1, 0, 1] = INCLUDE   #   ...AND f1
    ta[1, 1, 2] = INCLUDE   # class-1 positive clause 1: includes NOT-f0
    ta[1, 1, 3] = INCLUDE   #   ...AND NOT-f1
    ta[1, 2, 0] = INCLUDE   # class-1 negative clause 2: includes f0
    ta[1, 3, 1] = INCLUDE   # class-1 negative clause 3: includes f1

    return TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=2, n_literals=4, n_classes=2,
            n_clauses_per_class=4, n_clauses_total=8,
            clause_organization="per_class", threshold=4,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
    )


@pytest.fixture()
def saved_yaml(tmp_path) -> tuple[TMIR, "ValidationConfig"]:
    tmir = _make_tmir()
    model_path = tmp_path / "model.yaml"
    tmir.to_yaml(model_path)
    cfg = ValidationConfig.model_validate({"model_path": str(model_path)})
    return tmir, cfg


@pytest.fixture()
def saved_npz(tmp_path) -> tuple[TMIR, "ValidationConfig"]:
    tmir = _make_tmir()
    model_path = tmp_path / "model.npz"
    tmir.to_npz(model_path)
    cfg = ValidationConfig.model_validate({"model_path": str(model_path)})
    return tmir, cfg


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def test_schema_passes_for_valid_model(saved_yaml):
    _, cfg = saved_yaml
    report = validate_model(cfg)
    assert report.schema_ok
    assert report.schema_error == ""
    assert report.all_passed


def test_schema_passes_from_npz(saved_npz):
    _, cfg = saved_npz
    report = validate_model(cfg)
    assert report.schema_ok
    assert report.all_passed


def test_schema_fails_for_bad_shape(tmp_path):
    tmir = _make_tmir()
    # Corrupt the ta_states shape to trigger validate_self() failure
    tmir.representation.ta_states = np.zeros((2, 4, 6), dtype=np.int32)
    model_path = tmp_path / "bad.yaml"
    # Bypass validate_self() by saving directly (the TMIR constructor doesn't
    # call validate_self — that is the caller's responsibility)
    tmir.to_yaml(model_path)
    cfg = ValidationConfig.model_validate({"model_path": str(model_path)})
    report = validate_model(cfg)
    assert not report.schema_ok
    assert "shape" in report.schema_error
    assert not report.all_passed


# ---------------------------------------------------------------------------
# Test-vector verification
# ---------------------------------------------------------------------------

def test_vectors_pass_when_correct(tmp_path):
    from matador.inference.reference import predict

    tmir = _make_tmir()
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    preds, scores = predict(tmir, X)

    tmir.verification = Verification(
        test_vectors=[
            EvalVector(
                input=list(x),
                expected_class=int(preds[i]),
                expected_scores=[int(scores[i, 0]), int(scores[i, 1])],
            )
            for i, x in enumerate(X.tolist())
        ]
    )

    model_path = tmp_path / "model.yaml"
    tmir.to_yaml(model_path)
    cfg = ValidationConfig.model_validate({"model_path": str(model_path)})
    report = validate_model(cfg)

    assert report.vectors_ran
    assert report.vectors_total == 4
    assert report.vectors_passed == 4
    assert report.all_passed


def test_vectors_fail_when_wrong(tmp_path):
    tmir = _make_tmir()
    tmir.verification = Verification(
        test_vectors=[
            EvalVector(input=[0, 0], expected_class=99, expected_scores=[0, 0])
        ]
    )
    model_path = tmp_path / "model.yaml"
    tmir.to_yaml(model_path)
    cfg = ValidationConfig.model_validate({"model_path": str(model_path)})
    report = validate_model(cfg)

    assert report.vectors_ran
    assert report.vectors_passed == 0
    assert not report.all_passed


def test_no_vectors_not_run(saved_yaml):
    _, cfg = saved_yaml
    report = validate_model(cfg)
    assert not report.vectors_ran


# ---------------------------------------------------------------------------
# Held-out test-set accuracy
# ---------------------------------------------------------------------------

def test_accuracy_reported_when_test_data_provided(tmp_path):
    tmir = _make_tmir()
    model_path = tmp_path / "model.yaml"
    tmir.to_yaml(model_path)

    # Write a small test dataset: 4 samples, 2 features, last col = label
    # We don't care about exact accuracy — just that the field is populated.
    data_path = tmp_path / "test.txt"
    data_path.write_text("0 0 0\n0 1 0\n1 0 0\n1 1 1\n")

    cfg = ValidationConfig.model_validate(
        {"model_path": str(model_path), "test_data": str(data_path)}
    )
    report = validate_model(cfg)

    assert report.accuracy_pct is not None
    assert report.accuracy_correct is not None
    assert report.accuracy_total == 4
    assert 0.0 <= report.accuracy_pct <= 100.0


def test_accuracy_not_reported_without_test_data(saved_yaml):
    _, cfg = saved_yaml
    report = validate_model(cfg)
    assert report.accuracy_pct is None
