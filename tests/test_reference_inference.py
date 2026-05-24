"""Tests for matador.inference.reference — pure-NumPy inference golden model.

No tmu dependency; do NOT add pytest.importorskip("tmu") here.
"""

from __future__ import annotations

import numpy as np
import pytest

from matador.ir.tm_ir import (
    Architecture,
    EvalVector,
    Hyperparameters,
    Representation,
    TMIR,
    Verification,
    Weights,
    WeightRange,
)
from matador.inference.reference import predict, clause_outputs, verify_against_vectors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vanilla_2f_2c(ta_states: np.ndarray) -> TMIR:
    """2 features, 2 classes, 4 clauses per class — vanilla."""
    # ta_states shape: (2, 4, 4)
    return TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=2,
            n_literals=4,
            n_classes=2,
            n_clauses_per_class=4,
            n_clauses_total=8,
            clause_organization="per_class",
            threshold=4,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta_states),
    )


def _make_weighted_2f_2c(ta_states: np.ndarray, weights: np.ndarray) -> TMIR:
    """Same as vanilla but with explicit non-negative integer weights."""
    abs_max = int(np.abs(weights).max()) or 1
    return TMIR(
        variant="weighted",
        architecture=Architecture(
            n_features=2,
            n_literals=4,
            n_classes=2,
            n_clauses_per_class=4,
            n_clauses_total=8,
            clause_organization="per_class",
            threshold=4,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta_states),
        weights=Weights(
            values=weights,
            signed=False,
            bit_width=max(1, int(np.ceil(np.log2(abs_max + 1)))),
            range=WeightRange(
                min=int(weights.min()),
                max=int(weights.max()),
                abs_max=abs_max,
            ),
        ),
    )


def _make_coalesced_2f_2c(ta_states: np.ndarray, weights: np.ndarray) -> TMIR:
    """2 features, 2 classes, 4 shared clauses — coalesced (signed weights)."""
    abs_max = int(np.abs(weights).max()) or 1
    return TMIR(
        variant="coalesced",
        architecture=Architecture(
            n_features=2,
            n_literals=4,
            n_classes=2,
            n_clauses_per_class=None,
            n_clauses_total=4,
            clause_organization="coalesced",
            threshold=4,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta_states),
        weights=Weights(
            values=weights,
            signed=True,
            bit_width=max(1, int(np.ceil(np.log2(abs_max + 1)))) + 1,
            range=WeightRange(
                min=int(weights.min()),
                max=int(weights.max()),
                abs_max=abs_max,
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Reference TA states for a known 2-feature, 2-class TM
#
# Features: f0, f1    Literals: [f0, f1, NOT-f0, NOT-f1]
#
# We craft clauses with known semantics:
#   Class-0 positive clauses (k=0,1): include specific literals
#   Class-0 negative clauses (k=2,3): include specific literals
#   Class-1 positive clauses (k=0,1): mirror
#   Class-1 negative clauses (k=2,3): mirror
#
# TA state > 128 → include literal.  We use 200 = include, 50 = exclude.
# ---------------------------------------------------------------------------

INCLUDE = 200  # > 128
EXCLUDE = 50   # <= 128

def _craft_ta(n_features=2):
    """
    Craft a deterministic TA state array for a 2-feature, 2-class TM.

    n_literals = 4: [f0, f1, NOT-f0, NOT-f1]

    Per-class clauses (4 per class, half positive / half negative):
      Class 0:
        clause 0 (positive): includes f0 only         → fires on {f0=1}
        clause 1 (positive): includes f1 only         → fires on {f1=1}
        clause 2 (negative): includes NOT-f0 only     → fires on {f0=0}
        clause 3 (negative): includes NOT-f1 only     → fires on {f1=0}
      Class 1:
        clause 0 (positive): includes f0 AND f1       → fires on {f0=1, f1=1}
        clause 1 (positive): includes NOT-f0 AND NOT-f1 → fires on {f0=0,f1=0}
        clause 2 (negative): includes f0 only         → fires on {f0=1}
        clause 3 (negative): includes f1 only         → fires on {f1=1}
    """
    ta = np.full((2, 4, 4), EXCLUDE, dtype=np.int32)
    # Class 0, clause 0 (positive): include f0
    ta[0, 0, 0] = INCLUDE
    # Class 0, clause 1 (positive): include f1
    ta[0, 1, 1] = INCLUDE
    # Class 0, clause 2 (negative): include NOT-f0
    ta[0, 2, 2] = INCLUDE
    # Class 0, clause 3 (negative): include NOT-f1
    ta[0, 3, 3] = INCLUDE

    # Class 1, clause 0 (positive): include f0 AND f1
    ta[1, 0, 0] = INCLUDE
    ta[1, 0, 1] = INCLUDE
    # Class 1, clause 1 (positive): include NOT-f0 AND NOT-f1
    ta[1, 1, 2] = INCLUDE
    ta[1, 1, 3] = INCLUDE
    # Class 1, clause 2 (negative): include f0
    ta[1, 2, 0] = INCLUDE
    # Class 1, clause 3 (negative): include f1
    ta[1, 3, 1] = INCLUDE

    return ta


_TA = _craft_ta()


def _expected_score_c0(x):
    """Hand-computed score for class 0 given 2-bit input x = (f0, f1)."""
    f0, f1 = x
    # positive clauses (+1 each if fired)
    c0 = int(f0 == 1)          # clause 0: includes f0
    c1 = int(f1 == 1)          # clause 1: includes f1
    # negative clauses (-1 each if fired)
    c2 = int(f0 == 0)          # clause 2: includes NOT-f0
    c3 = int(f1 == 0)          # clause 3: includes NOT-f1
    return c0 + c1 - c2 - c3


def _expected_score_c1(x):
    """Hand-computed score for class 1."""
    f0, f1 = x
    # positive clauses
    c0 = int(f0 == 1 and f1 == 1)   # clause 0: f0 AND f1
    c1 = int(f0 == 0 and f1 == 0)   # clause 1: NOT-f0 AND NOT-f1
    # negative clauses
    c2 = int(f0 == 1)               # clause 2: f0
    c3 = int(f1 == 1)               # clause 3: f1
    return c0 + c1 - c2 - c3


ALL_INPUTS = [(f0, f1) for f0 in (0, 1) for f1 in (0, 1)]
# (0,0), (0,1), (1,0), (1,1)


# ---------------------------------------------------------------------------
# Vanilla tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", ALL_INPUTS)
def test_vanilla_known_scores(x):
    tmir = _make_vanilla_2f_2c(_TA.copy())
    X = np.array([list(x)], dtype=np.uint8)
    preds, scores = predict(tmir, X)
    expected_s0 = _expected_score_c0(x)
    expected_s1 = _expected_score_c1(x)
    assert int(scores[0, 0]) == expected_s0, f"x={x}: c0 score mismatch"
    assert int(scores[0, 1]) == expected_s1, f"x={x}: c1 score mismatch"
    expected_pred = 0 if expected_s0 >= expected_s1 else 1
    assert int(preds[0]) == expected_pred, f"x={x}: prediction mismatch"


def test_vanilla_all_inputs_batch():
    tmir = _make_vanilla_2f_2c(_TA.copy())
    X = np.array(ALL_INPUTS, dtype=np.uint8)
    preds, scores = predict(tmir, X)
    assert preds.shape == (4,)
    assert scores.shape == (4, 2)
    for i, x in enumerate(ALL_INPUTS):
        assert int(scores[i, 0]) == _expected_score_c0(x)
        assert int(scores[i, 1]) == _expected_score_c1(x)


# ---------------------------------------------------------------------------
# Weighted tests (unit weights → same predictions as vanilla)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("x", ALL_INPUTS)
def test_weighted_unit_weights_matches_vanilla(x):
    unit_w = np.ones((2, 4), dtype=np.int32)
    tmir_w = _make_weighted_2f_2c(_TA.copy(), unit_w)
    tmir_v = _make_vanilla_2f_2c(_TA.copy())
    X = np.array([list(x)], dtype=np.uint8)
    _, scores_w = predict(tmir_w, X)
    _, scores_v = predict(tmir_v, X)
    np.testing.assert_array_equal(scores_w, scores_v)


def test_weighted_non_unit_weights():
    # Give all positive clauses weight=2, negative weight=1
    # Class 0 scores should double-count positive contributions.
    w = np.array([[2, 2, 1, 1], [2, 2, 1, 1]], dtype=np.int32)
    tmir = _make_weighted_2f_2c(_TA.copy(), w)
    X = np.array([[1, 1]], dtype=np.uint8)  # f0=1, f1=1
    _, scores = predict(tmir, X)
    # Class 0: +2*(f0=1) + 2*(f1=1) - 1*(f0=0→0) - 1*(f1=0→0) = 4
    assert int(scores[0, 0]) == 4


# ---------------------------------------------------------------------------
# Coalesced tests
# ---------------------------------------------------------------------------

def _craft_coalesced_ta():
    """4 shared clauses for 2-feature problem."""
    ta = np.full((4, 4), EXCLUDE, dtype=np.int32)
    # clause 0: includes f0
    ta[0, 0] = INCLUDE
    # clause 1: includes f1
    ta[1, 1] = INCLUDE
    # clause 2: includes NOT-f0
    ta[2, 2] = INCLUDE
    # clause 3: includes NOT-f1
    ta[3, 3] = INCLUDE
    return ta


@pytest.mark.parametrize("x", ALL_INPUTS)
def test_coalesced_known_scores(x):
    """Coalesced with signed weights: verify against hand-computed result."""
    ta = _craft_coalesced_ta()
    # Class 0: +1 on clauses 0 and 1, -1 on clauses 2 and 3
    # Class 1: -1 on clauses 0 and 1, +1 on clauses 2 and 3
    w = np.array([[1, 1, -1, -1], [-1, -1, 1, 1]], dtype=np.int32)
    tmir = _make_coalesced_2f_2c(ta, w)
    X = np.array([list(x)], dtype=np.uint8)
    preds, scores = predict(tmir, X)

    f0, f1 = x
    co = [int(f0), int(f1), int(1 - f0), int(1 - f1)]

    expected_c0 = co[0] * 1 + co[1] * 1 + co[2] * (-1) + co[3] * (-1)
    expected_c1 = co[0] * (-1) + co[1] * (-1) + co[2] * 1 + co[3] * 1

    assert int(scores[0, 0]) == expected_c0, f"x={x}: coalesced c0 score"
    assert int(scores[0, 1]) == expected_c1, f"x={x}: coalesced c1 score"


# ---------------------------------------------------------------------------
# includes-only vs ta_states TMIR — must produce same predictions
# ---------------------------------------------------------------------------

def test_includes_equals_ta_states_inference():
    ta = _TA.copy()
    tmir_ta = _make_vanilla_2f_2c(ta)
    X = np.array(ALL_INPUTS, dtype=np.uint8)

    # Derive includes and make an includes-only TMIR
    threshold = tmir_ta.hyperparameters.n_states // 2
    inc = (ta > threshold)
    tmir_inc = TMIR(
        variant="vanilla",
        architecture=tmir_ta.architecture,
        hyperparameters=tmir_ta.hyperparameters,
        representation=Representation(includes=inc),
    )

    preds_ta, scores_ta = predict(tmir_ta, X)
    preds_inc, scores_inc = predict(tmir_inc, X)
    np.testing.assert_array_equal(preds_ta, preds_inc)
    np.testing.assert_array_equal(scores_ta, scores_inc)


# ---------------------------------------------------------------------------
# Empty clause convention (inference: 0, not 1)
# ---------------------------------------------------------------------------

def test_empty_clause_outputs_zero():
    """A clause with no included literals must output 0 at inference."""
    # All TA states below threshold → no literals included
    ta = np.full((2, 4, 4), EXCLUDE, dtype=np.int32)
    tmir = _make_vanilla_2f_2c(ta)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    co = clause_outputs(tmir, X)
    # All clauses empty → all outputs must be 0
    assert co.max() == 0, "Empty clauses must output 0 at inference"


# ---------------------------------------------------------------------------
# clause_outputs() shape
# ---------------------------------------------------------------------------

def test_clause_outputs_per_class_shape():
    tmir = _make_vanilla_2f_2c(_TA.copy())
    X = np.array(ALL_INPUTS, dtype=np.uint8)
    co = clause_outputs(tmir, X)
    # per_class: (n_samples, n_classes * n_clauses_per_class)
    assert co.shape == (4, 8)


def test_clause_outputs_coalesced_shape():
    ta = _craft_coalesced_ta()
    w = np.ones((2, 4), dtype=np.int32)
    tmir = _make_coalesced_2f_2c(ta, w)
    X = np.array(ALL_INPUTS, dtype=np.uint8)
    co = clause_outputs(tmir, X)
    assert co.shape == (4, 4)


# ---------------------------------------------------------------------------
# verify_against_vectors()
# ---------------------------------------------------------------------------

def test_verify_against_vectors_all_pass():
    ta = _TA.copy()
    tmir = _make_vanilla_2f_2c(ta)
    X = np.array(ALL_INPUTS, dtype=np.uint8)
    _, scores = predict(tmir, X)
    preds = np.argmax(scores, axis=1)

    test_vectors = [
        EvalVector(
            input=list(x),
            expected_class=int(preds[i]),
            expected_scores=[int(scores[i, 0]), int(scores[i, 1])],
        )
        for i, x in enumerate(ALL_INPUTS)
    ]
    tmir.verification = Verification(test_vectors=test_vectors)
    result = verify_against_vectors(tmir)
    assert result.total == 4
    assert result.passed == 4
    assert result.failed == 0


def test_verify_against_vectors_detects_failure():
    ta = _TA.copy()
    tmir = _make_vanilla_2f_2c(ta)
    # Intentionally wrong expected_class
    tmir.verification = Verification(
        test_vectors=[
            EvalVector(
                input=[0, 0],
                expected_class=99,  # wrong
                expected_scores=[0, 0],  # also wrong
            )
        ]
    )
    result = verify_against_vectors(tmir)
    assert result.failed == 1
    assert result.per_vector[0].passed is False


def test_verify_no_vectors_returns_empty():
    tmir = _make_vanilla_2f_2c(_TA.copy())
    result = verify_against_vectors(tmir)
    assert result.total == 0
