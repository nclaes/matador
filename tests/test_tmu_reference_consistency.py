"""Integration test: TMIR + reference inference matches TMU predictions.

Requires tmu to be installed; skips gracefully otherwise.

This is the proof that TMIR + reference inference is faithful to TMU:
  1. Train a tiny TMU model.
  2. Convert with from_tmu().
  3. For 200 random Boolean inputs, compare TMU predict() vs reference predict().
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import tmu
    from tmu.tmulib import ffi, lib  # noqa: F401
except (ImportError, ModuleNotFoundError):
    pytest.skip("tmu C extension not available", allow_module_level=True)

from matador.models.tmu_adapter import from_tmu
from matador.inference.reference import predict as ref_predict

# ---------------------------------------------------------------------------
# Dataset and hyperparameters
# ---------------------------------------------------------------------------

N_FEATURES = 10
N_CLAUSES = 20
T = 5
S = 3.9
SEED = 99
EPOCHS = 20
N_TEST = 200


def _make_noisy_xor(n_samples, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    X = rng.integers(0, 2, size=(n_samples, N_FEATURES)).astype(np.uint32)
    y = (X[:, 0] ^ X[:, 1]).astype(np.uint32)
    return X, y


def _make_test_inputs():
    rng = np.random.default_rng(SEED + 1)
    return rng.integers(0, 2, size=(N_TEST, N_FEATURES)).astype(np.uint32)


# ---------------------------------------------------------------------------
# Vanilla
# ---------------------------------------------------------------------------

def test_vanilla_reference_matches_tmu():
    from tmu.models.classification.vanilla_classifier import TMClassifier
    X_train, y_train = _make_noisy_xor(500)
    tm = TMClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S, weighted_clauses=False, seed=SEED
    )
    for _ in range(EPOCHS):
        tm.fit(X_train, y_train)

    X_test = _make_test_inputs()
    tmu_pred = tm.predict(X_test)

    tmir = from_tmu(tm)
    ref_pred, _ = ref_predict(tmir, X_test)

    n_agree = int(np.array_equal(tmu_pred, ref_pred))
    mismatches = np.where(tmu_pred != ref_pred)[0]
    assert len(mismatches) == 0, (
        f"Vanilla: {len(mismatches)}/{N_TEST} prediction mismatches.\n"
        f"First mismatch at index {mismatches[0]}: "
        f"tmu={tmu_pred[mismatches[0]]}, ref={ref_pred[mismatches[0]]}"
    )


# ---------------------------------------------------------------------------
# Weighted
# ---------------------------------------------------------------------------

def test_weighted_reference_matches_tmu():
    from tmu.models.classification.vanilla_classifier import TMClassifier
    X_train, y_train = _make_noisy_xor(500)
    tm = TMClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S, weighted_clauses=True, seed=SEED
    )
    for _ in range(EPOCHS):
        tm.fit(X_train, y_train)

    X_test = _make_test_inputs()
    tmu_pred = tm.predict(X_test)

    tmir = from_tmu(tm)
    ref_pred, _ = ref_predict(tmir, X_test)

    mismatches = np.where(tmu_pred != ref_pred)[0]
    assert len(mismatches) == 0, (
        f"Weighted: {len(mismatches)}/{N_TEST} prediction mismatches.\n"
        f"First mismatch at index {mismatches[0]}: "
        f"tmu={tmu_pred[mismatches[0]]}, ref={ref_pred[mismatches[0]]}"
    )


# ---------------------------------------------------------------------------
# Coalesced
# ---------------------------------------------------------------------------

def test_coalesced_reference_matches_tmu():
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
    X_train, y_train = _make_noisy_xor(500)
    tm = TMCoalescedClassifier(
        number_of_clauses=N_CLAUSES, T=T, s=S, seed=SEED
    )
    for _ in range(EPOCHS):
        tm.fit(X_train, y_train)

    X_test = _make_test_inputs()
    tmu_pred = tm.predict(X_test)

    tmir = from_tmu(tm)
    ref_pred, _ = ref_predict(tmir, X_test)

    mismatches = np.where(tmu_pred != ref_pred)[0]
    assert len(mismatches) == 0, (
        f"Coalesced: {len(mismatches)}/{N_TEST} prediction mismatches.\n"
        f"First mismatch at index {mismatches[0]}: "
        f"tmu={tmu_pred[mismatches[0]]}, ref={ref_pred[mismatches[0]]}"
    )
