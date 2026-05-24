"""Reference Tsetlin Machine inference — pure NumPy, zero tmu dependency.

This module is the golden model for TMIR inference:
  - Correct-first design: the algorithm is written to be readable as a spec.
  - No tmu, no torch, no sklearn — only numpy.
  - Suitable as a deployable inference path for small/medium models.

Important inference convention (differs from TMU training behaviour):
  A clause with ZERO included literals outputs 0 at inference time.
  During TMU training, empty clauses output 1 so the regulator can shrink
  them; at inference this convention must NOT be applied.  Violating this
  causes over-optimistic scores for untrained clauses and is a common
  source of bugs when porting TM inference code.

See tmir_spec.md §"Canonical Conventions" for the full spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from matador.ir.tm_ir import TMIR


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict(
    tmir: TMIR, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (predictions, scores).

    Args:
        tmir: Loaded TMIR model.
        X:    Boolean input features, shape (n_samples, n_features).
              Dtype must be uint8 or bool; values must be 0/1.

    Returns:
        predictions: shape (n_samples,)           — argmax class per sample.
        scores:      shape (n_samples, n_classes)  — raw class scores.

    Algorithm:
        1. Build literal matrix L of shape (n_samples, 2*n_features):
             L[:, :F]  = X          (positive literals)
             L[:, F:]  = 1 - X      (negated literals)
        2. Get include mask from representation (bool, same shape as ta_states).
        3. For each clause k: clause_out = AND over included literals.
             If clause k has ZERO included literals: clause_out = 0
             (inference convention — see module docstring).
        4. Accumulate scores per clause organisation.
        5. argmax for predictions.
    """
    X = np.asarray(X, dtype=np.uint8)
    n_samples, n_features = X.shape
    F = n_features

    # Step 1: build literal matrix
    L = np.empty((n_samples, 2 * F), dtype=np.uint8)
    L[:, :F] = X
    L[:, F:] = 1 - X

    # Step 2: resolve include mask
    inc = _resolve_includes(tmir)  # bool array, shape depends on organisation

    arch = tmir.architecture
    org = arch.clause_organization
    n_classes = arch.n_classes

    # Step 3+4: compute scores
    scores = np.zeros((n_samples, n_classes), dtype=np.int64)

    if org == "per_class":
        n_clauses_per_class = arch.n_clauses_per_class
        half = n_clauses_per_class // 2

        # inc shape: (n_classes, n_clauses_per_class, n_literals)
        co = _clause_outputs_per_class(L, inc)  # (n_samples, n_classes, n_clauses_per_class)

        polarity = np.where(np.arange(n_clauses_per_class) < half, 1, -1)
        # polarity: (n_clauses_per_class,)

        if tmir.weights is None:
            # weight=1 for all clauses
            # scores: (n_samples, n_classes) = sum over clauses of polarity * co
            scores = (co * polarity[np.newaxis, np.newaxis, :]).sum(axis=2)
        else:
            # weights.values shape: (n_classes, n_clauses_per_class)
            w = tmir.weights.values.astype(np.int64)  # (n_classes, n_clauses_per_class)
            effective = polarity[np.newaxis, :] * w   # (n_classes, n_clauses_per_class)
            # co: (n_samples, n_classes, n_clauses_per_class)
            scores = (co * effective[np.newaxis, :, :]).sum(axis=2)

    else:  # coalesced
        # inc shape: (n_clauses_total, n_literals)
        co = _clause_outputs_flat(L, inc)  # (n_samples, n_clauses_total)

        # weights.values shape: (n_classes, n_clauses_total), signed
        w = tmir.weights.values.astype(np.int64)  # (n_classes, n_clauses_total)
        # scores: (n_samples, n_classes) = co @ w.T
        scores = co.astype(np.int64) @ w.T

    predictions = np.argmax(scores, axis=1).astype(np.int64)
    return predictions, scores.astype(np.int64)


def clause_outputs(tmir: TMIR, X: np.ndarray) -> np.ndarray:
    """Return per-sample per-clause boolean outputs.

    Shape: (n_samples, n_clauses_total) for both organisations.

    Useful for:
      - Hardware verification (compare clause-by-clause against RTL).
      - Interpretability: which clauses fire on which inputs.
      - Telemetry: collecting clause activation frequency.
    """
    X = np.asarray(X, dtype=np.uint8)
    n_samples, n_features = X.shape
    F = n_features

    L = np.empty((n_samples, 2 * F), dtype=np.uint8)
    L[:, :F] = X
    L[:, F:] = 1 - X

    inc = _resolve_includes(tmir)
    arch = tmir.architecture

    if arch.clause_organization == "per_class":
        # inc: (n_classes, n_clauses_per_class, n_literals)
        co = _clause_outputs_per_class(L, inc)  # (n_samples, n_classes, n_clauses_per_class)
        n_classes = arch.n_classes
        n_cpc = arch.n_clauses_per_class
        return co.reshape(n_samples, n_classes * n_cpc).astype(np.uint8)
    else:
        return _clause_outputs_flat(L, inc).astype(np.uint8)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@dataclass
class VectorResult:
    input: list[int]
    expected_class: int
    expected_scores: list[int]
    predicted_class: int
    predicted_scores: list[int]
    passed: bool


@dataclass
class VerificationResult:
    total: int = 0
    passed: int = 0
    failed: int = 0
    per_vector: list[VectorResult] = field(default_factory=list)


def verify_against_vectors(tmir: TMIR) -> VerificationResult:
    """Run stored test vectors through predict() and check results.

    If ``tmir.verification`` is absent or has no test vectors, returns an
    empty VerificationResult.  This is the function CI runs to validate that
    the reference implementation is consistent with whatever produced the IR.
    """
    result = VerificationResult()

    if tmir.verification is None or not tmir.verification.test_vectors:
        return result

    for tv in tmir.verification.test_vectors:
        x = np.array([tv.input], dtype=np.uint8)
        preds, scores = predict(tmir, x)
        pred_class = int(preds[0])
        pred_scores = scores[0].tolist()

        passed = (
            pred_class == tv.expected_class
            and pred_scores == tv.expected_scores
        )
        result.per_vector.append(
            VectorResult(
                input=tv.input,
                expected_class=tv.expected_class,
                expected_scores=tv.expected_scores,
                predicted_class=pred_class,
                predicted_scores=pred_scores,
                passed=passed,
            )
        )
        result.total += 1
        if passed:
            result.passed += 1
        else:
            result.failed += 1

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_includes(tmir: TMIR) -> np.ndarray:
    rep = tmir.representation
    if rep.ta_states is not None:
        threshold = tmir.hyperparameters.n_states // 2
        return rep.ta_states > threshold
    if rep.includes is not None:
        return rep.includes.astype(bool)
    raise NotImplementedError(
        "Compressed representation is not yet supported by the reference "
        "inference engine (TMIR v0.2).  Decompress first, then set "
        "representation.includes or representation.ta_states."
    )


def _clause_outputs_flat(
    L: np.ndarray, inc: np.ndarray
) -> np.ndarray:
    """Compute clause outputs for a flat (coalesced) include mask.

    Args:
        L:   literal matrix, shape (n_samples, 2*F), uint8 0/1
        inc: include mask,    shape (n_clauses, 2*F), bool

    Returns:
        outputs: shape (n_samples, n_clauses), uint8 0/1
    """
    n_samples = L.shape[0]
    n_clauses = inc.shape[0]
    out = np.zeros((n_samples, n_clauses), dtype=np.uint8)

    for k in range(n_clauses):
        mask = inc[k]  # (2F,) bool
        if not mask.any():
            # Empty clause → output 0 at inference (see module docstring)
            continue
        # All included literals must be 1
        out[:, k] = L[:, mask].all(axis=1).astype(np.uint8)

    return out


def _clause_outputs_per_class(
    L: np.ndarray, inc: np.ndarray
) -> np.ndarray:
    """Compute clause outputs for per-class include mask.

    Args:
        L:   literal matrix, shape (n_samples, 2*F), uint8 0/1
        inc: include mask,    shape (n_classes, n_clauses_per_class, 2*F), bool

    Returns:
        outputs: shape (n_samples, n_classes, n_clauses_per_class), uint8
    """
    n_classes, n_clauses_per_class, _ = inc.shape
    # Reshape include to (n_classes * n_clauses_per_class, 2F) and compute flat
    flat_inc = inc.reshape(-1, inc.shape[-1])  # (n_classes*n_cpc, 2F)
    flat_out = _clause_outputs_flat(L, flat_inc)  # (n_samples, n_classes*n_cpc)
    return flat_out.reshape(flat_out.shape[0], n_classes, n_clauses_per_class)
