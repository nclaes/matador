"""Tests for matador.models.tmu_adapter.

Requires tmu to be installed; skips gracefully otherwise.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest

# importorskip("tmu") only checks that the Python package exists; the C
# extension (tmulib) is a separate shared library that may be absent on
# non-Linux / non-x86 hosts.  Skip the whole module if it can't be loaded.
try:
    import tmu
    from tmu.tmulib import ffi, lib  # noqa: F401 — triggers the real check
except (ImportError, ModuleNotFoundError):
    pytest.skip("tmu C extension not available", allow_module_level=True)

from matador.models.tmu_adapter import from_tmu, to_tmu
from matador.ir.tm_ir import TMIR


# ---------------------------------------------------------------------------
# Noisy-XOR dataset helpers
# ---------------------------------------------------------------------------

N_FEATURES = 10
N_CLAUSES = 20
T = 5
S = 3.9
SEED = 7
EPOCHS = 20


def _make_noisy_xor(n_samples=200, n_features=N_FEATURES, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    X = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.uint32)
    y = (X[:, 0] ^ X[:, 1]).astype(np.uint32)
    return X, y


def _train_vanilla():
    from tmu.models.classification.vanilla_classifier import TMClassifier
    X, y = _make_noisy_xor()
    tm = TMClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S, weighted_clauses=False, seed=SEED
    )
    for _ in range(EPOCHS):
        tm.fit(X, y)
    return tm


def _train_weighted():
    from tmu.models.classification.vanilla_classifier import TMClassifier
    X, y = _make_noisy_xor()
    tm = TMClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S, weighted_clauses=True, seed=SEED
    )
    for _ in range(EPOCHS):
        tm.fit(X, y)
    return tm


def _train_coalesced():
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    X, y = _make_noisy_xor()
    tm = TMCoalescedClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S, seed=SEED
    )
    for _ in range(EPOCHS):
        tm.fit(X, y)
    return tm


# ---------------------------------------------------------------------------
# from_tmu tests
# ---------------------------------------------------------------------------

def test_vanilla_from_tmu():
    tm = _train_vanilla()
    tmir = from_tmu(tm)
    assert tmir.variant == "vanilla"
    assert tmir.architecture.clause_organization == "per_class"
    assert tmir.representation.ta_states is not None
    tmir.validate_self()


def test_weighted_from_tmu():
    tm = _train_weighted()
    tmir = from_tmu(tm)
    assert tmir.variant == "weighted"
    assert tmir.weights is not None
    assert not tmir.weights.signed
    tmir.validate_self()


def test_coalesced_from_tmu():
    tm = _train_coalesced()
    tmir = from_tmu(tm)
    assert tmir.variant == "coalesced"
    assert tmir.architecture.clause_organization == "coalesced"
    assert tmir.weights is not None
    assert tmir.weights.signed
    tmir.validate_self()


# ---------------------------------------------------------------------------
# YAML round-trip for each variant
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vanilla_tmir():
    return from_tmu(_train_vanilla())


@pytest.fixture(scope="module")
def weighted_tmir():
    return from_tmu(_train_weighted())


@pytest.fixture(scope="module")
def coalesced_tmir():
    return from_tmu(_train_coalesced())


def test_vanilla_yaml_roundtrip(vanilla_tmir, tmp_path):
    path = tmp_path / "vanilla.yaml"
    vanilla_tmir.to_yaml(path)
    restored = TMIR.from_yaml(path)
    restored.validate_self()
    np.testing.assert_array_equal(
        vanilla_tmir.representation.ta_states,
        restored.representation.ta_states,
    )


def test_weighted_yaml_roundtrip(weighted_tmir, tmp_path):
    path = tmp_path / "weighted.yaml"
    weighted_tmir.to_yaml(path)
    restored = TMIR.from_yaml(path)
    restored.validate_self()
    np.testing.assert_array_equal(
        weighted_tmir.weights.values,
        restored.weights.values,
    )


def test_coalesced_yaml_roundtrip(coalesced_tmir, tmp_path):
    path = tmp_path / "coalesced.yaml"
    coalesced_tmir.to_yaml(path)
    restored = TMIR.from_yaml(path)
    restored.validate_self()
    np.testing.assert_array_equal(
        coalesced_tmir.weights.values,
        restored.weights.values,
    )


# ---------------------------------------------------------------------------
# to_tmu raises NotImplementedError for v0.2
# ---------------------------------------------------------------------------

def test_to_tmu_not_implemented():
    tmir = from_tmu(_train_vanilla())
    with pytest.raises(NotImplementedError):
        to_tmu(tmir)


# ---------------------------------------------------------------------------
# Source-tree grep: only tmu_adapter.py may import tmu
# ---------------------------------------------------------------------------

def test_only_tmu_adapter_imports_tmu():
    """Walk matador/ package directory and verify no file besides tmu_adapter.py
    contains a bare 'import tmu' or 'from tmu' statement.
    """
    matador_root = Path(__file__).parent.parent / "matador"
    adapter_path = matador_root / "models" / "tmu_adapter.py"

    violations = []
    pattern = re.compile(r"^\s*(import tmu\b|from tmu\b)", re.MULTILINE)

    for py_file in matador_root.rglob("*.py"):
        if py_file.resolve() == adapter_path.resolve():
            continue
        text = py_file.read_text(encoding="utf-8")
        if pattern.search(text):
            violations.append(str(py_file.relative_to(matador_root)))

    assert violations == [], (
        "The following files in matador/ import tmu directly but only "
        "tmu_adapter.py is allowed to do so:\n  "
        + "\n  ".join(violations)
    )
