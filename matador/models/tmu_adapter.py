"""TMU ↔ TMIR adapter.

THIS IS THE ONLY MODULE IN THE ENTIRE matador PACKAGE THAT MAY IMPORT TMU.
An integration test in tests/test_tmu_adapter.py enforces this invariant
by grepping the source tree.

Two entry points:
  from_tmu(tm)         -> TMIR
  to_tmu(tmir, ...)    -> tmu model  (NotImplementedError for v0.2)

Also re-exports helpers used by the training pipeline so that
matador/models/trainer.py never has to import tmu directly.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matador.ir.tm_ir import TMIR


# ---------------------------------------------------------------------------
# Training-pipeline helpers (re-exported so trainer.py stays tmu-free)
# ---------------------------------------------------------------------------

def import_tmu_classifier():
    """Return (TMClassifier, BenchmarkTimer) from tmu, raising clearly on failure."""
    import traceback
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
        from tmu.tools import BenchmarkTimer
        return TMClassifier, BenchmarkTimer
    except Exception as exc:
        traceback.print_exc()
        raise SystemExit(
            f"\nImport failed: {exc}\n\n"
            "If the tmu C extension is missing, compile it:\n"
            "  python3 tmu/lib/tmulib_extension_build.py"
        ) from exc


# ---------------------------------------------------------------------------
# from_tmu
# ---------------------------------------------------------------------------

def from_tmu(tm) -> "TMIR":
    """Import a trained TMU classifier into TMIR.

    Supports tmu.models.classification.{vanilla, weighted, coalesced}
    classifiers.  Detects which variant by inspecting attributes on the
    tm object.

    TMU's internal layout has evolved across versions.  We access state
    through the public get_ta_state() method (or fall back to reading
    .clause_bank directly for older releases) and do not hard-code memory
    offsets.

    If this function raises an unexpected AttributeError or TypeError,
    your tmu version may have changed its internal layout.  Please open
    an issue with the output of ``import tmu; print(tmu.__version__)``.
    """
    import tmu  # noqa: PLC0415 — intentional, this module owns tmu imports

    from matador.ir.tm_ir import (
        Architecture,
        Derived,
        Hyperparameters,
        Provenance,
        Representation,
        TMIR,
        WeightRange,
        Weights,
        _compute_derived,
    )

    variant, org = _detect_variant(tm)

    if org == "per_class":
        ta_states, weights_arr = _extract_per_class(tm, variant)
        n_classes = len(list(tm.clause_banks.classes()))
        n_clauses_per_class = tm.number_of_clauses
        n_clauses_total = n_classes * n_clauses_per_class
        n_features = ta_states.shape[-1] // 2
    else:
        ta_states, weights_arr = _extract_coalesced(tm)
        n_classes = len(list(tm.weight_banks.classes()))
        n_clauses_per_class = None
        n_clauses_total = tm.number_of_clauses
        n_features = ta_states.shape[-1] // 2

    n_literals = 2 * n_features
    n_states = 2 ** tm.number_of_state_bits_ta

    arch = Architecture(
        n_features=n_features,
        n_literals=n_literals,
        n_classes=n_classes,
        n_clauses_per_class=n_clauses_per_class,
        n_clauses_total=n_clauses_total,
        clause_organization=org,
        threshold=int(tm.T),
    )

    hyper = Hyperparameters(
        s=float(tm.s),
        n_states=n_states,
        seed=tm.seed if hasattr(tm, "seed") else None,
    )

    rep = Representation(ta_states=ta_states.astype(np.int32))

    weights: Weights | None = None
    if weights_arr is not None:
        signed = (org == "coalesced")
        abs_max = int(np.abs(weights_arr).max())
        if abs_max == 0:
            abs_max = 1
        bw = max(1, math.ceil(math.log2(abs_max + 1))) + (1 if signed else 0)
        weights = Weights(
            values=weights_arr.astype(np.int32),
            signed=signed,
            bit_width=bw,
            range=WeightRange(
                min=int(weights_arr.min()),
                max=int(weights_arr.max()),
                abs_max=abs_max,
            ),
        )

    prov = Provenance(
        framework="tmu",
        framework_version=str(tmu.__version__),
        trained_at=datetime.now(timezone.utc),
        epochs=0,
        hyperparameter_log={
            k: v
            for k, v in vars(tm).items()
            if isinstance(v, (int, float, str, bool)) and not k.startswith("_")
        },
    )

    tmir = TMIR(
        variant=variant,
        architecture=arch,
        hyperparameters=hyper,
        representation=rep,
        weights=weights,
        provenance=prov,
    )
    return tmir


def to_tmu(tmir: "TMIR", tm_class=None):
    """Reconstruct a TMU model from TMIR.

    Not implemented for TMIR v0.2.  This would require re-initialising a
    TMU model, encoding the TMIR ta_states back into TMU's bit-packed
    clause bank format, and restoring weight banks — a non-trivial
    undertaking.  Contributions welcome.
    """
    raise NotImplementedError(
        "to_tmu() is not implemented for TMIR v0.2.  "
        "Re-training from TMIR is not yet supported."
    )


# ---------------------------------------------------------------------------
# Internal extraction helpers
# ---------------------------------------------------------------------------

def _detect_variant(tm) -> tuple[str, str]:
    """Return (variant_name, clause_organization) for a tmu model."""
    try:
        from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
        if isinstance(tm, TMCoalescedClassifier):
            return "coalesced", "coalesced"
    except ImportError:
        pass

    # Vanilla or weighted: both use TMClassifier with per-class clause banks.
    # The weighted flag lives on the model object.
    has_multi_banks = hasattr(tm, "clause_banks")
    if has_multi_banks:
        weighted = getattr(tm, "weighted_clauses", False)
        variant = "weighted" if weighted else "vanilla"
        return variant, "per_class"

    raise TypeError(
        f"Unrecognised tmu model type: {type(tm).__qualname__}.  "
        "Only vanilla, weighted, and coalesced classifiers are supported.  "
        "Please open an issue with your tmu version."
    )


def _bank_get_ta_states_vectorised(bank) -> np.ndarray:
    """Extract all TA states from a clause bank into shape (n_clauses, n_literals).

    Uses a vectorised decode of the raw bit-packed clause_bank array, which is
    orders of magnitude faster than calling bank.get_ta_state() in a loop.
    Falls back to the loop-based approach if the raw array is inaccessible
    (e.g., future tmu versions that change the internal layout).
    """
    n_clauses = int(bank.number_of_clauses)
    n_literals = int(bank.number_of_literals)

    try:
        raw = bank.clause_bank  # shape: 1-D uint32 array
        n_ta_chunks = int(bank.number_of_ta_chunks)
        n_state_bits = int(bank.number_of_state_bits_ta)

        cb = raw.reshape(n_clauses, n_ta_chunks, n_state_bits).astype(np.int64)

        ta_chunks = np.arange(n_literals) // 32
        chunk_positions = np.arange(n_literals) % 32

        # cb[:, ta_chunks, :] → shape (n_clauses, n_literals, n_state_bits)
        gathered = cb[:, ta_chunks, :]
        shifted = (gathered >> chunk_positions[np.newaxis, :, np.newaxis]) & 1
        powers = (1 << np.arange(n_state_bits, dtype=np.int64))
        states = (shifted * powers[np.newaxis, np.newaxis, :]).sum(axis=2)
        return states.astype(np.int32)
    except (AttributeError, TypeError, ValueError):
        # Fallback: public API loop (slow but safe)
        states = np.zeros((n_clauses, n_literals), dtype=np.int32)
        for c in range(n_clauses):
            for l in range(n_literals):
                states[c, l] = bank.get_ta_state(c, l)
        return states


def _extract_per_class(tm, variant: str):
    """Extract ta_states and (optional) weights for per_class organisation.

    Returns:
        ta_states:  (n_classes, n_clauses_per_class, n_literals)  int32
        weights:    (n_classes, n_clauses_per_class) int32, or None for vanilla
    """
    classes = sorted(tm.clause_banks.classes())
    n_classes = len(classes)
    sample_bank = tm.clause_banks[classes[0]]
    n_clauses = int(sample_bank.number_of_clauses)
    n_literals = int(sample_bank.number_of_literals)

    ta_states = np.zeros((n_classes, n_clauses, n_literals), dtype=np.int32)
    for i, cls in enumerate(classes):
        ta_states[i] = _bank_get_ta_states_vectorised(tm.clause_banks[cls])

    if variant == "vanilla":
        return ta_states, None

    # Weighted: store absolute values (sign is carried by clause index).
    # TMU initialises first-half weights positive and second-half negative;
    # the TMIR per_class convention keeps weights non-negative.
    weight_vals = np.zeros((n_classes, n_clauses), dtype=np.int32)
    for i, cls in enumerate(classes):
        w = tm.weight_banks[cls].get_weights().copy()
        weight_vals[i] = np.abs(w).astype(np.int32)

    return ta_states, weight_vals


def _extract_coalesced(tm):
    """Extract ta_states and weights for coalesced organisation.

    Returns:
        ta_states:  (n_clauses, n_literals)        int32
        weights:    (n_classes, n_clauses)          int32  (signed)
    """
    ta_states = _bank_get_ta_states_vectorised(tm.clause_bank)

    classes = sorted(tm.weight_banks.classes())
    n_classes = len(classes)
    n_clauses = int(tm.clause_bank.number_of_clauses)

    weight_vals = np.zeros((n_classes, n_clauses), dtype=np.int32)
    for i, cls in enumerate(classes):
        weight_vals[i] = tm.weight_banks[cls].get_weights().copy().astype(np.int32)

    return ta_states, weight_vals
