"""Trace event dataclasses for the TM accelerator emulator.

Every event records the cycle counter at the time it was emitted.
Events mirror the architectural boundaries visible in the RTL:
FIFO, tile ROM, clause_eval, score_acc, argmax, and the top-level FSM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class TraceEvent:
    cycle: int
    module: str


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------

@dataclass
class FSMEvent(TraceEvent):
    from_state: str
    to_state: str
    trigger: str

    def __init__(self, cycle: int, from_state: str, to_state: str, trigger: str):
        super().__init__(cycle=cycle, module="tm_accelerator")
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger


# ---------------------------------------------------------------------------
# FIFO
# ---------------------------------------------------------------------------

@dataclass
class FIFOPushEvent(TraceEvent):
    data: int
    tlast: bool
    wptr: int
    occupancy: int

    def __init__(self, cycle: int, data: int, tlast: bool, wptr: int, occupancy: int):
        super().__init__(cycle=cycle, module="axis_fifo")
        self.data = data
        self.tlast = tlast
        self.wptr = wptr
        self.occupancy = occupancy


@dataclass
class FIFOPopEvent(TraceEvent):
    data: int
    tlast: bool
    rptr: int
    occupancy: int

    def __init__(self, cycle: int, data: int, tlast: bool, rptr: int, occupancy: int):
        super().__init__(cycle=cycle, module="axis_fifo")
        self.data = data
        self.tlast = tlast
        self.rptr = rptr
        self.occupancy = occupancy


# ---------------------------------------------------------------------------
# Literal vector
# ---------------------------------------------------------------------------

@dataclass
class LiteralVectorEvent(TraceEvent):
    feat_slice_idx: int
    features_window: list   # list[int]
    literals: list          # list[int] — {~feat_pos, feat_pos}

    def __init__(self, cycle: int, feat_slice_idx: int,
                 features_window: list, literals: list):
        super().__init__(cycle=cycle, module="tm_accelerator")
        self.feat_slice_idx = feat_slice_idx
        self.features_window = features_window
        self.literals = literals


# ---------------------------------------------------------------------------
# Tile ROM
# ---------------------------------------------------------------------------

@dataclass
class ROMAccessEvent(TraceEvent):
    tile_addr: int
    feat_slice_idx: int
    clause_slice_idx: int
    ta_action_matrix: int   # raw tile word

    def __init__(self, cycle: int, tile_addr: int, feat_slice_idx: int,
                 clause_slice_idx: int, ta_action_matrix: int):
        super().__init__(cycle=cycle, module="tile_rom")
        self.tile_addr = tile_addr
        self.feat_slice_idx = feat_slice_idx
        self.clause_slice_idx = clause_slice_idx
        self.ta_action_matrix = ta_action_matrix


# ---------------------------------------------------------------------------
# TA action vector (per clause, per tile)
# ---------------------------------------------------------------------------

@dataclass
class TAActionVectorEvent(TraceEvent):
    clause_global: int
    clause_local_k: int
    ta_action_mask: int     # 2*FEAT_SLICE bits
    ta_actions_pos: int     # FEAT_SLICE bits — positive literal TA actions
    ta_actions_neg: int     # FEAT_SLICE bits — negated literal TA actions

    def __init__(self, cycle: int, clause_global: int, clause_local_k: int,
                 ta_action_mask: int, ta_actions_pos: int, ta_actions_neg: int):
        super().__init__(cycle=cycle, module="tile_rom")
        self.clause_global = clause_global
        self.clause_local_k = clause_local_k
        self.ta_action_mask = ta_action_mask
        self.ta_actions_pos = ta_actions_pos
        self.ta_actions_neg = ta_actions_neg


# ---------------------------------------------------------------------------
# Clause partial evaluation (one tile column, not final)
# ---------------------------------------------------------------------------

@dataclass
class ClausePartialEvalEvent(TraceEvent):
    clause_global: int
    feat_slice_idx: int
    clause_slice_idx: int
    partial_literals: int
    ta_action_mask: int
    partial_pass: bool
    tile_has_actions: bool
    is_first_slice: bool

    def __init__(self, cycle: int, clause_global: int, feat_slice_idx: int,
                 clause_slice_idx: int, partial_literals: int, ta_action_mask: int,
                 partial_pass: bool, tile_has_actions: bool, is_first_slice: bool):
        super().__init__(cycle=cycle, module="clause_eval")
        self.clause_global = clause_global
        self.feat_slice_idx = feat_slice_idx
        self.clause_slice_idx = clause_slice_idx
        self.partial_literals = partial_literals
        self.ta_action_mask = ta_action_mask
        self.partial_pass = partial_pass
        self.tile_has_actions = tile_has_actions
        self.is_first_slice = is_first_slice


# ---------------------------------------------------------------------------
# Clause running accumulation state
# ---------------------------------------------------------------------------

@dataclass
class ClauseRunningStateEvent(TraceEvent):
    clause_global: int
    clause_pass_running: bool
    clause_has_actions: bool

    def __init__(self, cycle: int, clause_global: int,
                 clause_pass_running: bool, clause_has_actions: bool):
        super().__init__(cycle=cycle, module="tm_accelerator")
        self.clause_global = clause_global
        self.clause_pass_running = clause_pass_running
        self.clause_has_actions = clause_has_actions


# ---------------------------------------------------------------------------
# Clause final result (after all feature slices)
# ---------------------------------------------------------------------------

@dataclass
class ClauseFinalResultEvent(TraceEvent):
    clause_global: int
    clause_active: bool
    polarity: str           # "positive" or "negative"
    class_idx: int

    def __init__(self, cycle: int, clause_global: int, clause_active: bool,
                 polarity: str, class_idx: int):
        super().__init__(cycle=cycle, module="tm_accelerator")
        self.clause_global = clause_global
        self.clause_active = clause_active
        self.polarity = polarity
        self.class_idx = class_idx


# ---------------------------------------------------------------------------
# Score accumulator
# ---------------------------------------------------------------------------

@dataclass
class ScoreVoteEvent(TraceEvent):
    score_cnt: int
    clause_global: int
    class_idx: int
    polarity: str
    clause_active: bool
    score_before: int
    score_after: int

    def __init__(self, cycle: int, score_cnt: int, clause_global: int,
                 class_idx: int, polarity: str, clause_active: bool,
                 score_before: int, score_after: int):
        super().__init__(cycle=cycle, module="score_acc")
        self.score_cnt = score_cnt
        self.clause_global = clause_global
        self.class_idx = class_idx
        self.polarity = polarity
        self.clause_active = clause_active
        self.score_before = score_before
        self.score_after = score_after


@dataclass
class ScoreStateEvent(TraceEvent):
    scores: list            # list[int]

    def __init__(self, cycle: int, scores: list):
        super().__init__(cycle=cycle, module="score_acc")
        self.scores = list(scores)


# ---------------------------------------------------------------------------
# Argmax
# ---------------------------------------------------------------------------

@dataclass
class ArgmaxEvent(TraceEvent):
    scores: list            # list[int]
    predicted_class: int

    def __init__(self, cycle: int, scores: list, predicted_class: int):
        super().__init__(cycle=cycle, module="argmax")
        self.scores = list(scores)
        self.predicted_class = predicted_class


# ---------------------------------------------------------------------------
# Top-level trace container
# ---------------------------------------------------------------------------

@dataclass
class InferenceTrace:
    input_features: list
    n_features: int
    n_classes: int
    n_clauses_total: int
    predicted_class: Optional[int] = None

    fsm_events: list = field(default_factory=list)
    fifo_events: list = field(default_factory=list)
    literal_events: list = field(default_factory=list)
    rom_events: list = field(default_factory=list)
    ta_action_events: list = field(default_factory=list)
    clause_partial_events: list = field(default_factory=list)
    clause_running_events: list = field(default_factory=list)
    clause_final_events: list = field(default_factory=list)
    score_vote_events: list = field(default_factory=list)
    score_state_events: list = field(default_factory=list)
    argmax_event: Optional[ArgmaxEvent] = None
