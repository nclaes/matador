"""Cross-reference verification for the TM accelerator.

compare_emulator_to_reference  — emulator vs. mathematical reference inference
run_regression                 — runs emulator on all embedded TMIR test vectors
                                 and compares to expected classes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VectorResult:
    index: int
    input_features: list
    expected_class: int
    emulator_class: int
    reference_class: int
    emulator_scores: list
    reference_scores: list
    emulator_matches_reference: bool
    emulator_matches_expected: bool

    @property
    def passed(self) -> bool:
        return self.emulator_matches_reference


@dataclass
class ComparisonReport:
    n_vectors: int
    n_passed: int
    n_failed: int
    results: list   # list[VectorResult]

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0

    @property
    def pass_rate(self) -> float:
        return 100.0 * self.n_passed / self.n_vectors if self.n_vectors > 0 else 0.0

    def first_failure(self) -> Optional[VectorResult]:
        return next((r for r in self.results if not r.passed), None)


# ---------------------------------------------------------------------------
# Core comparison functions
# ---------------------------------------------------------------------------

def compare_emulator_to_reference(tmir, cfg) -> ComparisonReport:
    """Run the emulator and reference inference on all embedded test vectors.

    Checks that TMAcceleratorEmulator agrees with the mathematical reference
    inference (matador.inference.reference.predict) on every vector.

    Args:
        tmir: TMIR model instance.
        cfg:  TMAcceleratorConfig instance.

    Returns:
        ComparisonReport with per-vector outcomes.
    """
    from matador.emulator.accelerator import TMAcceleratorEmulator
    from matador.inference.reference import predict

    emul = TMAcceleratorEmulator(tmir, cfg)

    if not (tmir.verification and tmir.verification.test_vectors):
        _LOGGER.warning("No embedded test vectors in TMIR — nothing to compare")
        return ComparisonReport(n_vectors=0, n_passed=0, n_failed=0, results=[])

    vectors = tmir.verification.test_vectors
    results = []
    n_passed = 0
    n_failed = 0

    for idx, vec in enumerate(vectors):
        features = list(vec.input)
        X = np.array([features], dtype=np.uint8)

        # Emulator
        trace = emul.run(features)
        emul_class = trace.predicted_class
        emul_scores = trace.argmax_event.scores if trace.argmax_event else []

        # Reference
        ref_preds, ref_scores_arr = predict(tmir, X)
        ref_class  = int(ref_preds[0])
        ref_scores = ref_scores_arr[0].tolist()

        match_ref = (emul_class == ref_class)
        match_exp = (emul_class == vec.expected_class)

        if not match_ref:
            n_failed += 1
            _LOGGER.debug(
                "Vector %d: emulator=%d reference=%d (scores emul=%s ref=%s)",
                idx, emul_class, ref_class, emul_scores, ref_scores,
            )
        else:
            n_passed += 1

        results.append(VectorResult(
            index=idx,
            input_features=features,
            expected_class=vec.expected_class,
            emulator_class=emul_class,
            reference_class=ref_class,
            emulator_scores=emul_scores,
            reference_scores=ref_scores,
            emulator_matches_reference=match_ref,
            emulator_matches_expected=match_exp,
        ))

    return ComparisonReport(
        n_vectors=len(vectors),
        n_passed=n_passed,
        n_failed=n_failed,
        results=results,
    )


def run_regression(tmir, cfg, extra_vectors: Optional[list] = None) -> ComparisonReport:
    """Run the emulator against reference on embedded vectors + optional extras.

    Args:
        tmir:          TMIR model.
        cfg:           TMAcceleratorConfig.
        extra_vectors: Additional (features_list, expected_class) tuples to test.

    Returns:
        ComparisonReport with per-vector outcomes.
    """
    from matador.emulator.accelerator import TMAcceleratorEmulator
    from matador.inference.reference import predict
    from matador.ir.tm_ir import EvalVector, Verification

    # Collect all vectors: embedded + extras
    all_vecs = []
    if tmir.verification and tmir.verification.test_vectors:
        all_vecs.extend(tmir.verification.test_vectors)
    if extra_vectors:
        for features, exp_cls in extra_vectors:
            all_vecs.append(EvalVector(
                input=features,
                expected_class=exp_cls if exp_cls is not None else 0,
                expected_scores=[],
            ))

    if not all_vecs:
        _LOGGER.warning("No test vectors provided to run_regression")
        return ComparisonReport(n_vectors=0, n_passed=0, n_failed=0, results=[])

    # Build a temporary TMIR with all vectors for compare_emulator_to_reference
    import copy
    tmir_copy = copy.deepcopy(tmir)
    tmir_copy.verification = Verification(test_vectors=all_vecs)
    return compare_emulator_to_reference(tmir_copy, cfg)
