"""Tests for matador.ir.tm_ir — schema, validation, serialisation round-trips."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from matador.ir.tm_ir import (
    Architecture,
    Hyperparameters,
    Representation,
    TMIR,
    Weights,
    WeightRange,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_vanilla(
    n_features=4,
    n_classes=2,
    n_clauses_per_class=4,
) -> TMIR:
    """Build a minimal vanilla TMIR with random TA states."""
    rng = np.random.default_rng(0)
    n_literals = 2 * n_features
    ta = rng.integers(1, 257, size=(n_classes, n_clauses_per_class, n_literals)).astype(
        np.int32
    )
    return TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features,
            n_literals=n_literals,
            n_classes=n_classes,
            n_clauses_per_class=n_clauses_per_class,
            n_clauses_total=n_classes * n_clauses_per_class,
            clause_organization="per_class",
            threshold=10,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
    )


def _minimal_coalesced(
    n_features=4,
    n_classes=2,
    n_clauses=4,
) -> TMIR:
    """Build a minimal coalesced TMIR with random TA states and signed weights."""
    rng = np.random.default_rng(1)
    n_literals = 2 * n_features
    ta = rng.integers(1, 257, size=(n_clauses, n_literals)).astype(np.int32)
    w_vals = rng.integers(-3, 4, size=(n_classes, n_clauses)).astype(np.int32)
    abs_max = int(np.abs(w_vals).max()) or 1
    weights = Weights(
        values=w_vals,
        signed=True,
        bit_width=max(1, int(np.ceil(np.log2(abs_max + 1)))) + 1,
        range=WeightRange(min=int(w_vals.min()), max=int(w_vals.max()), abs_max=abs_max),
    )
    return TMIR(
        variant="coalesced",
        architecture=Architecture(
            n_features=n_features,
            n_literals=n_literals,
            n_classes=n_classes,
            n_clauses_per_class=None,
            n_clauses_total=n_clauses,
            clause_organization="coalesced",
            threshold=10,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
        weights=weights,
    )


# ---------------------------------------------------------------------------
# Schema construction
# ---------------------------------------------------------------------------

def test_vanilla_construction():
    tmir = _minimal_vanilla()
    assert tmir.variant == "vanilla"
    assert tmir.architecture.n_literals == 8
    assert tmir.architecture.clause_organization == "per_class"
    assert tmir.representation.ta_states.shape == (2, 4, 8)


def test_coalesced_construction():
    tmir = _minimal_coalesced()
    assert tmir.variant == "coalesced"
    assert tmir.architecture.clause_organization == "coalesced"
    assert tmir.representation.ta_states.shape == (4, 8)
    assert tmir.weights is not None
    assert tmir.weights.signed is True


def test_n_literals_must_equal_2x_features():
    with pytest.raises(ValidationError, match="n_literals"):
        Architecture(
            n_features=4,
            n_literals=7,   # wrong — should be 8
            n_classes=2,
            n_clauses_per_class=4,
            n_clauses_total=8,
            clause_organization="per_class",
            threshold=10,
        )


def test_representation_requires_at_least_one():
    with pytest.raises(ValidationError):
        Representation()


def test_derived_computed_automatically():
    tmir = _minimal_vanilla()
    assert tmir.derived is not None
    assert tmir.derived.avg_clause_length >= 0
    assert tmir.derived.empty_clause_count >= 0


def test_derived_not_recomputed_when_provided():
    from matador.ir.tm_ir import Derived
    tmir = _minimal_vanilla()
    fake_derived = Derived(
        score_accumulator_width=99,
        avg_clause_length=0.0,
        max_clause_length=0,
        min_clause_length=0,
        clause_length_distribution={},
        empty_clause_count=0,
    )
    tmir2 = TMIR(
        variant="vanilla",
        architecture=tmir.architecture,
        hyperparameters=tmir.hyperparameters,
        representation=tmir.representation,
        derived=fake_derived,
    )
    assert tmir2.derived.score_accumulator_width == 99


# ---------------------------------------------------------------------------
# validate_self()
# ---------------------------------------------------------------------------

def test_validate_self_vanilla_passes():
    tmir = _minimal_vanilla()
    tmir.validate_self()  # must not raise


def test_validate_self_coalesced_passes():
    tmir = _minimal_coalesced()
    tmir.validate_self()


def test_validate_self_coalesced_no_weights_fails():
    tmir = _minimal_coalesced()
    tmir.weights = None
    with pytest.raises(ValueError, match="weights are required"):
        tmir.validate_self()


def test_validate_self_wrong_ta_shape():
    tmir = _minimal_vanilla()
    rng = np.random.default_rng(5)
    # Bad shape: wrong n_literals
    tmir.representation.ta_states = rng.integers(1, 257, size=(2, 4, 6)).astype(np.int32)
    with pytest.raises(ValueError, match="shape"):
        tmir.validate_self()


# ---------------------------------------------------------------------------
# to_includes()
# ---------------------------------------------------------------------------

def test_to_includes_derives_from_ta_states():
    tmir = _minimal_vanilla()
    tmir.to_includes()
    assert tmir.representation.includes is not None
    threshold = tmir.hyperparameters.n_states // 2
    expected = tmir.representation.ta_states > threshold
    np.testing.assert_array_equal(tmir.representation.includes, expected)


def test_to_includes_idempotent():
    tmir = _minimal_vanilla()
    tmir.to_includes()
    first = tmir.representation.includes.copy()
    tmir.to_includes()
    np.testing.assert_array_equal(tmir.representation.includes, first)


def test_to_includes_requires_ta_states():
    tmir = _minimal_vanilla()
    tmir.to_includes()
    tmir.representation.ta_states = None
    # Now only includes is set — calling to_includes() should be idempotent
    tmir.to_includes()


# ---------------------------------------------------------------------------
# fingerprint()
# ---------------------------------------------------------------------------

def test_fingerprint_is_stable():
    tmir = _minimal_vanilla()
    fp1 = tmir.fingerprint()
    fp2 = tmir.fingerprint()
    assert fp1 == fp2
    assert fp1.startswith("sha256:")


def test_fingerprint_changes_on_different_ta_states():
    tmir1 = _minimal_vanilla()
    tmir2 = _minimal_vanilla()
    rng = np.random.default_rng(999)
    tmir2.representation.ta_states = rng.integers(
        1, 257, size=tmir2.representation.ta_states.shape
    ).astype(np.int32)
    assert tmir1.fingerprint() != tmir2.fingerprint()


# ---------------------------------------------------------------------------
# Serialisation round-trips
# ---------------------------------------------------------------------------

def test_to_dict_from_dict_roundtrip():
    tmir = _minimal_vanilla()
    d = tmir.to_dict()
    assert isinstance(d, dict)
    restored = TMIR.from_dict(d)
    np.testing.assert_array_equal(
        tmir.representation.ta_states, restored.representation.ta_states
    )
    assert restored.variant == tmir.variant


def test_coalesced_to_dict_from_dict_roundtrip():
    tmir = _minimal_coalesced()
    d = tmir.to_dict()
    restored = TMIR.from_dict(d)
    np.testing.assert_array_equal(tmir.weights.values, restored.weights.values)


def test_to_yaml_from_yaml_roundtrip(tmp_path):
    tmir = _minimal_vanilla()
    path = tmp_path / "model.yaml"
    tmir.to_yaml(path)
    restored = TMIR.from_yaml(path)
    np.testing.assert_array_equal(
        tmir.representation.ta_states, restored.representation.ta_states
    )
    assert restored.variant == tmir.variant
    assert restored.architecture.n_features == tmir.architecture.n_features


def test_coalesced_to_yaml_from_yaml_roundtrip(tmp_path):
    tmir = _minimal_coalesced()
    path = tmp_path / "model.yaml"
    tmir.to_yaml(path)
    restored = TMIR.from_yaml(path)
    np.testing.assert_array_equal(tmir.weights.values, restored.weights.values)
    assert restored.weights.signed is True


def test_to_npz_from_npz_roundtrip(tmp_path):
    tmir = _minimal_vanilla()
    path = tmp_path / "model"
    tmir.to_npz(path)
    assert (tmp_path / "model.npz").exists()
    restored = TMIR.from_npz(path)
    np.testing.assert_array_equal(
        tmir.representation.ta_states, restored.representation.ta_states
    )
    assert restored.variant == tmir.variant


def test_coalesced_to_npz_from_npz_roundtrip(tmp_path):
    tmir = _minimal_coalesced()
    path = tmp_path / "model"
    tmir.to_npz(path)
    restored = TMIR.from_npz(path)
    np.testing.assert_array_equal(tmir.weights.values, restored.weights.values)
    assert restored.architecture.clause_organization == "coalesced"


# ---------------------------------------------------------------------------
# Includes-only representation
# ---------------------------------------------------------------------------

def test_includes_only_representation():
    """TMIR with only includes (no ta_states) should work for inference."""
    tmir = _minimal_vanilla()
    tmir.to_includes()
    inc_copy = tmir.representation.includes.copy()
    tmir.representation.ta_states = None

    # Recreate to test construction with includes-only
    restored = TMIR(
        variant="vanilla",
        architecture=tmir.architecture,
        hyperparameters=tmir.hyperparameters,
        representation=Representation(includes=inc_copy),
    )
    assert restored.representation.ta_states is None
    assert restored.representation.includes is not None


# ---------------------------------------------------------------------------
# JSON encoding of NumpyArray
# ---------------------------------------------------------------------------

def test_numpy_array_json_encoding():
    tmir = _minimal_vanilla()
    json_str = tmir.model_dump_json()
    data = json.loads(json_str)
    # ta_states should be serialised as {dtype, shape, data_b64}
    ts = data["representation"]["ta_states"]
    assert "dtype" in ts
    assert "shape" in ts
    assert "data_b64" in ts


def test_numpy_array_model_dump_python_mode():
    tmir = _minimal_vanilla()
    d = tmir.model_dump()
    # In Python mode, ta_states should be a raw numpy array
    assert isinstance(d["representation"]["ta_states"], np.ndarray)
