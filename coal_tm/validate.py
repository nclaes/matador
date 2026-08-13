"""Validate a TAs.txt/weights.txt pair against real labeled test vectors,
using the emulator -- before spending time on RTL generation.

This is a model-quality/sanity check, not a hardware-equivalence check
(that's what `coal_tm testbench` + `coal_tm sim` are for, once RTL exists).
Catches a wrong/corrupted export, a config geometry mismatch (wrong
classes/clauses/features), or a genuinely bad model early, with a plain
accuracy number against ground truth -- instead of only discovering a
problem after generate + testbench + sim, or worse, not noticing at all
because a bad model can still "pass" a testbench (which only checks the
RTL against the *emulator*, not against real labels).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coal_tm.config import RTLConfig
from coal_tm.emulator import CoalescedEmulator


@dataclass
class ValidationResult:
    n_vectors: int
    correct: int
    accuracy: float
    predictions: np.ndarray
    labels: np.ndarray


def validate(config: RTLConfig, n_vectors: int, test_data: Path | None = None) -> ValidationResult:
    test_data = test_data or config.test_data
    if test_data is None:
        raise ValueError("no test_data given (neither the --test-data flag nor config.test_data)")

    raw = np.loadtxt(test_data, dtype=int, ndmin=2)
    X, Y = raw[:, :-1], raw[:, -1]
    if X.shape[1] != config.features:
        raise ValueError(f"{test_data} has {X.shape[1]} feature columns, expected {config.features}")
    if X.shape[0] < n_vectors:
        raise ValueError(f"{test_data} has only {X.shape[0]} rows, need {n_vectors}")
    X, Y = X[:n_vectors], Y[:n_vectors]

    emu = CoalescedEmulator(config.tas, config.weights, config.classes, config.clauses, config.features)
    predictions, _ = emu.predict_batch(X)

    correct = int((predictions == Y).sum())
    return ValidationResult(
        n_vectors=n_vectors,
        correct=correct,
        accuracy=100.0 * correct / n_vectors,
        predictions=predictions,
        labels=Y,
    )
