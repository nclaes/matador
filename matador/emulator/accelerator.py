"""TMAcceleratorEmulator — golden reference model mirroring the 5-state RTL FSM.

Produces an InferenceTrace with cycle-ordered events for every architectural
boundary: FIFO, tile ROM, clause_eval, score_acc, argmax, and FSM transitions.
The predicted_class matches the mathematical reference inference.
"""

from __future__ import annotations

import math

import numpy as np

from matador.emulator.argmax import ArgmaxEmulator
from matador.emulator.clause_eval import ClauseEvalEmulator
from matador.emulator.fifo import FIFOEmulator
from matador.emulator.score_acc import ScoreAccEmulator
from matador.emulator.tile_rom import TileROMEmulator
from matador.emulator.trace import (
    ClauseFinalResultEvent,
    ClausePartialEvalEvent,
    ClauseRunningStateEvent,
    FSMEvent,
    InferenceTrace,
    LiteralVectorEvent,
    TAActionVectorEvent,
)


class TMAcceleratorEmulator:
    """Software emulator that mirrors tm_accelerator.v state by state.

    Instantiate once per TMIR + config pair, then call run() for each
    feature vector. run() resets stateful sub-modules automatically.
    """

    def __init__(self, tmir, cfg) -> None:
        from matador.ir.tm_ir import TMIR  # noqa: F401 — type hint only
        if tmir.variant != "vanilla":
            raise ValueError(f"Emulator only supports 'vanilla' TM, got '{tmir.variant}'")

        self.tmir = tmir
        self.cfg = cfg
        self._derive_params()

    # ------------------------------------------------------------------
    # Parameter derivation (mirrors TMAccelerator._derive_params)
    # ------------------------------------------------------------------

    def _derive_params(self) -> None:
        arch = self.tmir.architecture
        self.n_features = arch.n_features
        self.n_literals = arch.n_literals
        self.n_classes = arch.n_classes
        self.n_clauses_pc = arch.n_clauses_per_class
        self.n_clauses_total = arch.n_clauses_total
        self.threshold = arch.threshold

        self.axis_dw = self.cfg.axis_data_width
        self.fifo_depth = self.cfg.fifo_depth
        self.feat_slice = self.cfg.feat_slice
        self.clause_slice = self.cfg.clause_slice

        self.n_beats = math.ceil(self.n_features / self.axis_dw)
        self.n_feat_slices = math.ceil(self.n_features / self.feat_slice)
        self.n_clause_slices = math.ceil(self.n_clauses_total / self.clause_slice)
        self.n_feat_padded = self.n_feat_slices * self.feat_slice
        self.half_clauses_pc = self.n_clauses_pc // 2

        # Compiled TA Include-action bits: shape (n_clauses_total, n_literals)
        ta = self.tmir.representation.ta_states
        n_states = self.tmir.hyperparameters.n_states
        inc = (ta > n_states // 2)
        if inc.ndim == 3:
            inc = inc.reshape(self.n_clauses_total, self.n_literals)
        self._ta_actions = inc.astype(bool)

    def _pack_beats(self, feature_bits: list) -> list:
        beats = []
        for k in range(self.n_beats):
            lo = k * self.axis_dw
            hi = min(lo + self.axis_dw, self.n_features)
            word = 0
            for i, bit in enumerate(feature_bits[lo:hi]):
                word |= (int(bit) & 1) << i
            beats.append(word)
        return beats

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(self, features: list) -> InferenceTrace:
        """Run one inference and return a complete InferenceTrace.

        Args:
            features: list of int (0 or 1), length == n_features.

        Returns:
            InferenceTrace with predicted_class and all architectural events.
        """
        if len(features) != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {len(features)}"
            )

        trace = InferenceTrace(
            input_features=list(features),
            n_features=self.n_features,
            n_classes=self.n_classes,
            n_clauses_total=self.n_clauses_total,
        )

        # Shared mutable cycle counter
        cycle_ref = [0]

        # Instantiate sub-modules
        fifo = FIFOEmulator(depth=self.fifo_depth, cycle_ref=cycle_ref)
        tile_rom = TileROMEmulator(
            ta_actions=self._ta_actions,
            n_features=self.n_features,
            n_clauses_total=self.n_clauses_total,
            n_literals=self.n_literals,
            feat_slice=self.feat_slice,
            clause_slice=self.clause_slice,
            cycle_ref=cycle_ref,
        )
        score_acc = ScoreAccEmulator(
            n_classes=self.n_classes,
            cycle_ref=cycle_ref,
        )
        clause_eval = ClauseEvalEmulator()
        argmax_mod = ArgmaxEmulator()

        # ── S_IDLE → S_RECV: stream beats into FIFO ──────────────────
        trace.fsm_events.append(FSMEvent(
            cycle=cycle_ref[0], from_state="S_IDLE", to_state="S_RECV",
            trigger="s_tvalid",
        ))

        beats = self._pack_beats(features)
        for b, data in enumerate(beats):
            tlast = (b == len(beats) - 1)
            ev = fifo.push(data, tlast)
            trace.fifo_events.append(ev)
            cycle_ref[0] += 1

        # Drain FIFO into feature register
        feature_reg = [0] * self.n_feat_padded
        for b in range(self.n_beats):
            data, tlast, ev = fifo.pop()
            trace.fifo_events.append(ev)
            lo = b * self.axis_dw
            for i in range(self.axis_dw):
                if lo + i < self.n_features:
                    feature_reg[lo + i] = (data >> i) & 1

        # ── S_RECV → S_COMPUTE ────────────────────────────────────────
        trace.fsm_events.append(FSMEvent(
            cycle=cycle_ref[0], from_state="S_RECV", to_state="S_COMPUTE",
            trigger="tlast",
        ))

        # Clause running state (mirrors clause_pass[] and clause_has_actions[] registers)
        clause_pass = [True] * self.n_clauses_total
        clause_has_actions = [False] * self.n_clauses_total

        for feat_slice_idx in range(self.n_feat_slices):
            feat_start = feat_slice_idx * self.feat_slice
            features_window = feature_reg[feat_start: feat_start + self.feat_slice]

            # Partial literals for this tile column: {~feat_pos, feat_pos}
            # Matches RTL layout: partial_lits = {feat_neg, feat_pos}
            lits_pos = features_window
            lits_neg = [1 - b for b in features_window]
            literal_list = lits_pos + lits_neg   # index 0..FS-1 = pos, FS..2FS-1 = neg
            partial_lits_int = sum(b << i for i, b in enumerate(literal_list))

            trace.literal_events.append(LiteralVectorEvent(
                cycle=cycle_ref[0],
                feat_slice_idx=feat_slice_idx,
                features_window=list(features_window),
                literals=literal_list,
            ))

            for clause_slice_idx in range(self.n_clause_slices):
                tile_data, rom_ev = tile_rom.read(feat_slice_idx, clause_slice_idx)
                trace.rom_events.append(rom_ev)

                for k in range(self.clause_slice):
                    g = clause_slice_idx * self.clause_slice + k
                    if g >= self.n_clauses_total:
                        continue

                    ta_actions_pos, ta_actions_neg, ta_action_mask = \
                        tile_rom.extract_ta_actions_k(tile_data, k)

                    trace.ta_action_events.append(TAActionVectorEvent(
                        cycle=cycle_ref[0],
                        clause_global=g,
                        clause_local_k=k,
                        ta_action_mask=ta_action_mask,
                        ta_actions_pos=ta_actions_pos,
                        ta_actions_neg=ta_actions_neg,
                    ))

                    fires, has_actions = clause_eval.eval(
                        partial_lits_int, ta_action_mask, 2 * self.feat_slice
                    )
                    is_first = (feat_slice_idx == 0)

                    trace.clause_partial_events.append(ClausePartialEvalEvent(
                        cycle=cycle_ref[0],
                        clause_global=g,
                        feat_slice_idx=feat_slice_idx,
                        clause_slice_idx=clause_slice_idx,
                        partial_literals=partial_lits_int,
                        ta_action_mask=ta_action_mask,
                        partial_pass=fires,
                        tile_has_actions=has_actions,
                        is_first_slice=is_first,
                    ))

                    # Empty tile (no Include actions in this feat slice) vacuously passes.
                    # Matches RTL: effective_pass = tile_pass | ~tile_has_actions
                    effective = fires or not has_actions
                    if is_first:
                        clause_pass[g] = effective
                        clause_has_actions[g] = has_actions
                    else:
                        clause_pass[g] = clause_pass[g] and effective
                        clause_has_actions[g] = clause_has_actions[g] or has_actions

                    trace.clause_running_events.append(ClauseRunningStateEvent(
                        cycle=cycle_ref[0],
                        clause_global=g,
                        clause_pass_running=clause_pass[g],
                        clause_has_actions=clause_has_actions[g],
                    ))

                cycle_ref[0] += 1   # one tile evaluation per cycle

        # ── S_COMPUTE → S_SCORE ───────────────────────────────────────
        trace.fsm_events.append(FSMEvent(
            cycle=cycle_ref[0], from_state="S_COMPUTE", to_state="S_SCORE",
            trigger="tiles_done",
        ))

        for g in range(self.n_clauses_total):
            class_idx = g // self.n_clauses_pc
            local_idx = g % self.n_clauses_pc
            is_positive = local_idx < self.half_clauses_pc
            polarity = "positive" if is_positive else "negative"
            clause_active = clause_pass[g] and clause_has_actions[g]

            trace.clause_final_events.append(ClauseFinalResultEvent(
                cycle=cycle_ref[0],
                clause_global=g,
                clause_active=clause_active,
                polarity=polarity,
                class_idx=class_idx,
            ))

            vote_ev = score_acc.update(class_idx, is_positive, clause_active)
            # Patch clause_global with actual g (score_acc uses internal counter)
            vote_ev.clause_global = g
            trace.score_vote_events.append(vote_ev)
            trace.score_state_events.append(score_acc.state_snapshot())
            cycle_ref[0] += 1

        # ── S_SCORE → S_DONE ──────────────────────────────────────────
        scores = score_acc.get_scores()
        pred_class, argmax_ev = argmax_mod.argmax(scores, cycle=cycle_ref[0])
        trace.argmax_event = argmax_ev
        trace.predicted_class = pred_class

        trace.fsm_events.append(FSMEvent(
            cycle=cycle_ref[0], from_state="S_SCORE", to_state="S_DONE",
            trigger="last_clause",
        ))
        trace.fsm_events.append(FSMEvent(
            cycle=cycle_ref[0] + 1, from_state="S_DONE", to_state="S_IDLE",
            trigger="m_axis_handshake",
        ))

        score_acc.clear()
        return trace
