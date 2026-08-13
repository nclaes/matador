"""Bit-exact Python reference model of the generated Coalesced-TM hardware.

Reads the same TAs.txt/weights.txt the RTL generator (coal_tm.rtl) reads —
not TMU's in-memory classifier state — and reproduces exactly what
TM_Hard_Coded_Clause_Blocks.sv / Adder_new / TM_argmax.sv compute:

  1. Per clause: AND of the included literals — except a clause with zero
     included literals anywhere ("all-exclude") is forced to 0, matching
     TMU's own C reference (ClauseBank.c: "Make empty clauses false") and
     the generated RTL's `all_exclude_indexes` special case. A block that
     locally has no included literals but whose clause has real includes
     in another block is a different case — that block contributes a
     vacuous `1` to the running AND across blocks, same as TMU's transform
     across the full (non-tiled) literal vector.
  2. Per class: signed weighted sum of the (shared) clause outputs.
  3. argmax over class sums.

This is deliberately a second, independent implementation of the same
computation coal_tm.rtl bakes into Verilog — the same "spec vs. hardware"
role production_ready's CycleAccurateModel plays, sized down to one
backend. Comparing this against TMCoalescedClassifier.predict() on the
same TAs/weights checks that the training-time export preserved the
model faithfully; comparing it against RTL simulation checks the
generated hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from coal_tm import artifacts


@dataclass
class EmulationResult:
    predicted_class: int
    class_sums: np.ndarray
    clause_outputs: np.ndarray


class CoalescedEmulator:
    def __init__(self, tas_path: Path, weights_path: Path, classes: int, clauses: int, features: int):
        self.classes = classes
        self.clauses = clauses
        self.features = features
        # (clauses, 2*features) uint8, interleaved [feat0+, feat0-, feat1+, feat1-, ...]
        self.includes = artifacts.read_tas_includes(tas_path, clauses, features)
        # (classes, clauses) signed int
        self.weights = artifacts.read_weights(weights_path, classes, clauses)
        # Clauses with zero included literals anywhere — forced to output 0.
        self.all_exclude = ~self.includes.astype(bool).any(axis=1)

    def clause_outputs(self, x: np.ndarray) -> np.ndarray:
        """x: (features,) array of 0/1. Returns (clauses,) array of 0/1."""
        x = np.asarray(x, dtype=np.uint8)
        if x.shape != (self.features,):
            raise ValueError(f"expected x.shape == ({self.features},), got {x.shape}")

        literals = np.empty(2 * self.features, dtype=bool)
        literals[0::2] = x.astype(bool)          # feat_i positive literal
        literals[1::2] = ~x.astype(bool)          # feat_i negated literal

        required = self.includes.astype(bool)      # (clauses, 2*features)
        # A clause is satisfied when every literal it requires is true.
        # Literals it doesn't require impose no constraint.
        satisfied_per_literal = (~required) | literals
        output = satisfied_per_literal.all(axis=1)
        output &= ~self.all_exclude
        return output.astype(np.int32)

    def predict(self, x: np.ndarray) -> EmulationResult:
        clause_out = self.clause_outputs(x)
        class_sums = self.weights @ clause_out
        return EmulationResult(
            predicted_class=int(np.argmax(class_sums)),
            class_sums=class_sums,
            clause_outputs=clause_out,
        )

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (predictions[n], class_sums[n, classes])."""
        X = np.asarray(X)
        predictions = np.empty(X.shape[0], dtype=np.int64)
        class_sums = np.empty((X.shape[0], self.classes), dtype=np.int64)
        for i, x in enumerate(X):
            result = self.predict(x)
            predictions[i] = result.predicted_class
            class_sums[i] = result.class_sums
        return predictions, class_sums
