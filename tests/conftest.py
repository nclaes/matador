"""Shared fixtures: a tiny, hand-verifiable Coalesced-TM model.

16 features, 2 clauses, 2 classes -- wide enough for bus_width=8 (needed
for byte-aligned AXI tkeep, [(C_M00_AXIS_TDATA_WIDTH/8)-1:0]) to still
leave features > bus_width (2 packets/vector, avoiding the single-packet
backpressure limitation RTLConfig now rejects; see coal_tm/config.py).
  clause 0: fires iff x0=1 AND x1=0 (features 2..15 unused/excluded).
  clause 1: no included literals at all -- an "all-exclude" clause, which
            must be forced to 0 (never fires), per TMU's own convention
            (ClauseBank.c: "Make empty clauses false").

weights: class0 = [5, 100], class1 = [-3, 7]. Clause 1's weight is
deliberately large (100/7) so a test that got the all-exclude rule wrong
(treating clause 1 as vacuously true) would produce a wildly different,
easily-detected answer instead of silently passing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

FEATURES = 16
CLAUSES = 2
CLASSES = 2

# Interleaved [f0+, f0-, f1+, f1-, ..., f15+, f15-] per clause, raw states
# (0-255) -- only f0/f1 are ever included, the rest are all-zero (excluded).
_TAS_STATES = (
    [200, 0, 0, 200] + [0] * 28   # clause 0: include f0+ and f1-
    + [0] * 32                     # clause 1: all-exclude
)

# Flat, class-major: class0's clauses, then class1's clauses.
_WEIGHTS = [5, 100, -3, 7]


@pytest.fixture
def tiny_model(tmp_path: Path) -> dict:
    tas_path = tmp_path / "TAs.txt"
    weights_path = tmp_path / "weights.txt"
    tas_path.write_text("\n".join(str(v) for v in _TAS_STATES) + "\n")
    weights_path.write_text("\n".join(str(v) for v in _WEIGHTS) + "\n")
    return {
        "tas": tas_path,
        "weights": weights_path,
        "features": FEATURES,
        "clauses": CLAUSES,
        "classes": CLASSES,
    }
