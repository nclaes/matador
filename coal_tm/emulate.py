"""Run the standalone Python reference model against real vectors.

Two modes: a single hand-supplied vector (--input, for quick ad hoc
checks), or rows read from a boolean text file -- defaulting to the
RTLConfig's own test_data, matching `validate`/`testbench`'s convention.
Typing out a `features`-long comma-separated vector by hand isn't
practical for anything but a toy model, so reading real data is the
normal path, not a special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coal_tm.config import RTLConfig
from coal_tm.emulator import CoalescedEmulator, EmulationResult


def emulate_single(config: RTLConfig, x: np.ndarray) -> EmulationResult:
    emu = CoalescedEmulator(config.tas, config.weights, config.classes, config.clauses, config.features)
    return emu.predict(x)


@dataclass
class BatchEmulationResult:
    predictions: np.ndarray
    class_sums: np.ndarray


def emulate_batch(config: RTLConfig, n_vectors: int | None = None, test_data: Path | None = None) -> BatchEmulationResult:
    test_data = test_data or config.test_data
    if test_data is None:
        raise ValueError("no test_data given (neither the --test-data flag nor config.test_data)")

    raw = np.loadtxt(test_data, dtype=int, ndmin=2)
    # Accept files with or without a trailing label column -- unlike
    # validate (which needs labels to check accuracy), emulate only ever
    # needs the features.
    X = raw[:, :-1] if raw.shape[1] == config.features + 1 else raw
    if X.shape[1] != config.features:
        raise ValueError(f"{test_data} has {X.shape[1]} feature columns, expected {config.features}")
    if n_vectors is not None:
        if X.shape[0] < n_vectors:
            raise ValueError(f"{test_data} has only {X.shape[0]} rows, need {n_vectors}")
        X = X[:n_vectors]

    emu = CoalescedEmulator(config.tas, config.weights, config.classes, config.clauses, config.features)
    predictions, class_sums = emu.predict_batch(X)
    return BatchEmulationResult(predictions=predictions, class_sums=class_sums)
