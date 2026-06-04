"""Hardwired backend emulator.

Cycle-accurate Python model of hw_tm_accelerator.  Much simpler than the
tiled emulator because there is no tile ROM lookup loop — evaluation is
a single combinational pass, with optional pipeline latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matador.backends.hardwired.config import HardwiredAcceleratorConfig
    from matador.ir.tm_ir import TMIR


@dataclass
class HWClauseEvalEvent:
    clause_global: int
    fired: bool


@dataclass
class HWScoreEvent:
    cls: int
    pos_sum: int
    neg_sum: int
    score: int


@dataclass
class HWArgmaxEvent:
    scores: list[int]
    predicted_class: int


@dataclass
class HWInferenceTrace:
    """Cycle-ordered events from one hw_tm_accelerator inference."""
    feature_vector:        list[int]
    clause_eval_events:    list[HWClauseEvalEvent] = field(default_factory=list)
    score_events:          list[HWScoreEvent]      = field(default_factory=list)
    argmax_event:          HWArgmaxEvent | None    = None
    predicted_class:       int                     = -1
    pipeline_stages_waited: int                    = 0


class HardwiredEmulator:
    """Cycle-accurate emulator for hw_tm_accelerator.

    Models the same algorithm as the RTL:
      1. Clause evaluation: combinational AND of included literals.
      2. Score accumulation: per-class pos/neg sums → net score.
      3. Pipeline latency: ``pipeline_stages`` register cycles before score is valid.
      4. Argmax: linear scan over scores.

    The emulator does NOT model the AXI-Stream protocol or FIFO — it takes
    a flat feature vector and returns an HWInferenceTrace.
    """

    def __init__(self, tmir: "TMIR", config: "HardwiredAcceleratorConfig") -> None:
        self.tmir   = tmir
        self.config = config

        tmir.to_includes()
        arch = tmir.architecture

        self.N  = arch.n_features
        self.C  = arch.n_classes
        self.K  = arch.n_clauses_per_class
        self.CT = arch.n_clauses_total
        self.PS = config.pipeline_stages

        # Shape: (CT, 2*N) bool
        self.includes = tmir.representation.includes.reshape(self.CT, 2 * self.N).astype(bool)

    def run(self, input_vector: list[int]) -> HWInferenceTrace:
        """Emulate one inference.  Matches hw_tm_accelerator RTL behaviour."""
        trace = HWInferenceTrace(feature_vector=list(input_vector))

        # Build literal vector [x_0..x_{N-1}, ~x_0..~x_{N-1}]
        X = np.asarray(input_vector[:self.N], dtype=np.uint8)
        L = np.empty(2 * self.N, dtype=np.uint8)
        L[:self.N]  = X
        L[self.N:]  = 1 - X

        # ── Clause evaluation (combinational) ────────────────────────────
        clause_fires = np.zeros(self.CT, dtype=bool)
        for g in range(self.CT):
            mask = self.includes[g]
            if not mask.any():
                clause_fires[g] = False   # empty clause → 0 (inference convention)
            else:
                clause_fires[g] = bool(L[mask].all())
            trace.clause_eval_events.append(
                HWClauseEvalEvent(clause_global=g, fired=bool(clause_fires[g]))
            )

        # ── Score accumulation ────────────────────────────────────────────
        HALF_K = self.K // 2
        scores = []
        for c in range(self.C):
            base   = c * self.K
            pos_sum = int(clause_fires[base          : base + HALF_K].sum())
            neg_sum = int(clause_fires[base + HALF_K : base + self.K].sum())
            score   = pos_sum - neg_sum
            scores.append(score)
            trace.score_events.append(
                HWScoreEvent(cls=c, pos_sum=pos_sum, neg_sum=neg_sum, score=score)
            )

        # ── Pipeline latency ─────────────────────────────────────────────
        # The RTL waits actual_ps cycles in S_EVAL before reading the score.
        # We model this as a metadata annotation only (no observable side-effect
        # on the output since the emulator runs instantaneously).
        trace.pipeline_stages_waited = min(self.PS, max(1, int(np.ceil(np.log2(max(HALF_K, 2))))))

        # ── Argmax ────────────────────────────────────────────────────────
        predicted = int(np.argmax(scores))
        trace.argmax_event    = HWArgmaxEvent(scores=scores, predicted_class=predicted)
        trace.predicted_class = predicted

        return trace

    def unit_testbenches(self, vectors: list) -> dict[str, str]:
        """No unit testbenches for the hardwired backend (single-module design)."""
        return {}
