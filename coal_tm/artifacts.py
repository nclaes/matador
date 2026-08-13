"""TAs.txt / weights.txt — the two files that hand a trained Coalesced TM's
parameters from training into RTL generation (and into the emulator).

Single source of truth for both directions (write from a live TMCoalescedClassifier
during training, read back during RTL generation/emulation) so the layout can't
drift between the two the way it did in the old flow: the file format is only
defined here.

Literal order: TMU's clause_bank stores each clause's 2*features TA states as
"all positive literals, then all negated literals" (ta index 0..features-1 =
feature i's positive literal, features..2*features-1 = feature i's negated
literal). coal_tm.rtl's generated Verilog expects them interleaved per feature
(index 2i = feature i positive, 2i+1 = feature i negated) — this module is the
only place that reordering happens.

State vs. action: TAs.txt stores raw automaton states (0-255ish), not
pre-thresholded 0/1 include bits. A literal is Included when the automaton's
top state bit is set, i.e. state >= 128 — the same bit TMU's own
get_ta_action() reads. Storing raw states (rather than already-thresholded
action bits) keeps this file format compatible with externally-supplied
TAs.txt files that predate this package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

INCLUDE_THRESHOLD = 128


def write_tas(path: Path, tm: Any, clauses: int, features: int) -> None:
    """Write one clause's 2*features raw TA states per row, interleaved
    positive/negated per feature, as a flat whitespace-separated file."""
    with open(path, "w") as f:
        for clause in range(clauses):
            for feat in range(features):
                pos_state = tm.get_ta_state(clause, feat)
                neg_state = tm.get_ta_state(clause, features + feat)
                f.write(f"{int(pos_state)}\n{int(neg_state)}\n")


def read_tas_states(path: Path, clauses: int, features: int) -> np.ndarray:
    """Raw states, shape (clauses, 2*features), interleaved literal order."""
    vals = np.loadtxt(path, dtype=int)
    expected = clauses * features * 2
    if vals.size != expected:
        raise ValueError(
            f"{path}: expected clauses * features * 2 = {expected} values, "
            f"got {vals.size}"
        )
    return vals.reshape(clauses, features * 2)


def read_tas_includes(path: Path, clauses: int, features: int) -> np.ndarray:
    """Include-bit matrix, shape (clauses, 2*features), uint8 0/1 —
    read_tas_states() thresholded at INCLUDE_THRESHOLD."""
    return (read_tas_states(path, clauses, features) >= INCLUDE_THRESHOLD).astype(np.uint8)


def write_weights(path: Path, tm: Any, classes: int, clauses: int) -> None:
    """Write signed integer weights, flat, row-major (class, clause)."""
    with open(path, "w") as f:
        for c in range(classes):
            weights = tm.get_weights(c)
            for clause in range(clauses):
                f.write(f"{int(weights[clause])}\n")


def read_weights(path: Path, classes: int, clauses: int) -> np.ndarray:
    """Signed weight matrix, shape (classes, clauses)."""
    vals = np.loadtxt(path, dtype=int)
    expected = classes * clauses
    if vals.size != expected:
        raise ValueError(
            f"{path}: expected classes * clauses = {expected} values, "
            f"got {vals.size}"
        )
    return vals.reshape(classes, clauses)
