"""Tests for matador.emulator — hardware-mirroring software emulator."""

from __future__ import annotations

import numpy as np
import pytest

from matador.config.schema import TMAcceleratorConfig
from matador.emulator.accelerator import TMAcceleratorEmulator
from matador.emulator.argmax import ArgmaxEmulator
from matador.emulator.clause_eval import ClauseEvalEmulator
from matador.emulator.fifo import FIFOEmulator
from matador.emulator.score_acc import ScoreAccEmulator
from matador.emulator.tile_rom import TileROMEmulator
from matador.emulator.trace import (
    ArgmaxEvent,
    ClauseFinalResultEvent,
    ClausePartialEvalEvent,
    FIFOPopEvent,
    FIFOPushEvent,
    FSMEvent,
    InferenceTrace,
    LiteralVectorEvent,
    ROMAccessEvent,
    ScoreStateEvent,
    ScoreVoteEvent,
    TAActionVectorEvent,
)
from matador.inference.reference import predict
from matador.ir.tm_ir import (
    Architecture,
    EvalVector,
    Hyperparameters,
    Representation,
    TMIR,
    Verification,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

INCLUDE = 200
EXCLUDE = 50


def _make_tmir(n_features=4, n_classes=2, n_clauses_pc=4, threshold=4) -> TMIR:
    n_literals = 2 * n_features
    ta = np.full((n_classes, n_clauses_pc, n_literals), EXCLUDE, dtype=np.int32)
    ta[0, 0, 0] = INCLUDE          # Class 0, clause 0: include f0 (pos)
    ta[0, 1, 1] = INCLUDE          # Class 0, clause 1: include f1 (pos)
    ta[1, 0, 0] = INCLUDE          # Class 1, clause 0: include f0 AND f1 (pos)
    ta[1, 0, 1] = INCLUDE
    ta[1, 2, n_features] = INCLUDE  # Class 1, clause 2: include NOT-f0 (neg)
    return TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=n_literals,
            n_classes=n_classes, n_clauses_per_class=n_clauses_pc,
            n_clauses_total=n_classes * n_clauses_pc,
            clause_organization="per_class", threshold=threshold,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
    )


def _make_cfg(tmp_path, axis_dw=32, feat_slice=4, clause_slice=4) -> TMAcceleratorConfig:
    tmir_path = tmp_path / "model.yaml"
    _make_tmir().to_yaml(tmir_path)
    return TMAcceleratorConfig.model_validate({
        "model_path": str(tmir_path),
        "output_dir": str(tmp_path),
        "axis_data_width": axis_dw,
        "fifo_depth": 16,
        "feat_slice": feat_slice,
        "clause_slice": clause_slice,
    })


def _make_emulator(tmp_path, **kwargs) -> TMAcceleratorEmulator:
    tmir = _make_tmir(**kwargs)
    cfg = _make_cfg(tmp_path)
    return TMAcceleratorEmulator(tmir, cfg)


# ---------------------------------------------------------------------------
# ClauseEvalEmulator
# ---------------------------------------------------------------------------

class TestClauseEvalEmulator:
    def setup_method(self):
        self.ce = ClauseEvalEmulator()

    def test_empty_mask_inactive(self):
        fires, has_actions = self.ce.eval(0b1111, 0b0000, 4)
        assert not fires
        assert not has_actions

    def test_single_include_lit_set(self):
        fires, has_actions = self.ce.eval(0b0001, 0b0001, 4)
        assert fires
        assert has_actions

    def test_single_include_lit_clear(self):
        fires, has_actions = self.ce.eval(0b0000, 0b0001, 4)
        assert not fires
        assert has_actions

    def test_all_included_all_set(self):
        fires, has_actions = self.ce.eval(0b1111, 0b1111, 4)
        assert fires

    def test_one_miss_inactive(self):
        # All included but bit 0 is 0
        fires, has_actions = self.ce.eval(0b1110, 0b1111, 4)
        assert not fires

    def test_non_included_bits_ignored(self):
        # bit 0 included and set; bits 1-3 not included but 0 — should still pass
        fires, _ = self.ce.eval(0b0001, 0b0001, 4)
        assert fires

    def test_8_literal_boundary(self):
        # 8 literals; include all; all set → fire
        fires, _ = self.ce.eval(0xFF, 0xFF, 8)
        assert fires


# ---------------------------------------------------------------------------
# FIFOEmulator
# ---------------------------------------------------------------------------

class TestFIFOEmulator:
    def setup_method(self):
        self.cycle_ref = [0]
        self.fifo = FIFOEmulator(depth=4, cycle_ref=self.cycle_ref)

    def test_push_pop_single(self):
        self.fifo.push(0xAB, tlast=True)
        data, tlast, _ = self.fifo.pop()
        assert data == 0xAB
        assert tlast is True

    def test_occupancy_tracks(self):
        assert self.fifo.occupancy == 0
        self.fifo.push(1, False)
        assert self.fifo.occupancy == 1
        self.fifo.pop()
        assert self.fifo.occupancy == 0

    def test_fifo_order_preserved(self):
        for i in range(4):
            self.fifo.push(i, tlast=(i == 3))
        for i in range(4):
            data, _, _ = self.fifo.pop()
            assert data == i

    def test_overflow_raises(self):
        for i in range(4):
            self.fifo.push(i, False)
        with pytest.raises(RuntimeError, match="overflow"):
            self.fifo.push(99, False)

    def test_underflow_raises(self):
        with pytest.raises(RuntimeError, match="underflow"):
            self.fifo.pop()

    def test_push_emits_event(self):
        ev = self.fifo.push(0x42, tlast=False)
        assert isinstance(ev, FIFOPushEvent)
        assert ev.data == 0x42
        assert ev.tlast is False

    def test_pop_emits_event(self):
        self.fifo.push(0x10, tlast=True)
        _, _, ev = self.fifo.pop()
        assert isinstance(ev, FIFOPopEvent)
        assert ev.data == 0x10

    def test_non_power_of_two_depth_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            FIFOEmulator(depth=3, cycle_ref=[0])


# ---------------------------------------------------------------------------
# ScoreAccEmulator
# ---------------------------------------------------------------------------

class TestScoreAccEmulator:
    def setup_method(self):
        self.cycle_ref = [0]
        self.acc = ScoreAccEmulator(n_classes=2, threshold=4, cycle_ref=self.cycle_ref)

    def test_positive_vote_increments(self):
        ev = self.acc.update(class_idx=0, is_positive=True, clause_active=True)
        assert ev.score_after == 1

    def test_negative_vote_decrements(self):
        ev = self.acc.update(class_idx=0, is_positive=False, clause_active=True)
        assert ev.score_after == -1

    def test_inactive_clause_no_change(self):
        ev = self.acc.update(class_idx=0, is_positive=True, clause_active=False)
        assert ev.score_after == 0

    def test_positive_saturation(self):
        for _ in range(10):
            ev = self.acc.update(0, True, True)
        assert ev.score_after == 4

    def test_negative_saturation(self):
        for _ in range(10):
            ev = self.acc.update(0, False, True)
        assert ev.score_after == -4

    def test_clear_resets(self):
        self.acc.update(0, True, True)
        self.acc.clear()
        assert self.acc.get_scores() == [0, 0]

    def test_independent_classes(self):
        self.acc.update(0, True, True)
        self.acc.update(1, False, True)
        scores = self.acc.get_scores()
        assert scores[0] == 1
        assert scores[1] == -1


# ---------------------------------------------------------------------------
# ArgmaxEmulator
# ---------------------------------------------------------------------------

class TestArgmaxEmulator:
    def setup_method(self):
        self.am = ArgmaxEmulator()

    def test_basic(self):
        idx, ev = self.am.argmax([1, 3, 2])
        assert idx == 1

    def test_tie_lower_index_wins(self):
        idx, _ = self.am.argmax([5, 5, 3])
        assert idx == 0

    def test_single_class(self):
        idx, _ = self.am.argmax([42])
        assert idx == 0

    def test_event_type(self):
        _, ev = self.am.argmax([1, 2])
        assert isinstance(ev, ArgmaxEvent)
        assert ev.predicted_class == 1


# ---------------------------------------------------------------------------
# TileROMEmulator
# ---------------------------------------------------------------------------

class TestTileROMEmulator:
    def setup_method(self):
        self.tmir = _make_tmir()
        n_states = self.tmir.hyperparameters.n_states
        ta = self.tmir.representation.ta_states
        inc = (ta > n_states // 2).reshape(
            self.tmir.architecture.n_clauses_total,
            self.tmir.architecture.n_literals,
        )
        self.rom = TileROMEmulator(
            ta_actions=inc,
            n_features=4, n_clauses_total=8, n_literals=8,
            feat_slice=4, clause_slice=4,
            cycle_ref=[0],
        )

    def test_tile_count(self):
        # n_feat_slices=1, n_clause_slices=2 → 2 tiles
        assert len(self.rom._rom) == self.rom.n_feat_slices * self.rom.n_clause_slices

    def test_read_returns_event(self):
        data, ev = self.rom.read(0, 0)
        assert isinstance(ev, ROMAccessEvent)
        assert ev.tile_addr == 0

    def test_clause0_includes_f0_pos(self):
        # Clause 0 includes literal 0 (positive f0)
        data, _ = self.rom.read(0, 0)
        ta_pos, ta_neg, mask = self.rom.extract_ta_actions_k(data, k=0)
        assert ta_pos & 1, "Clause 0 should include positive literal 0"

    def test_clause0_excludes_f1_pos(self):
        # Clause 0 does NOT include literal 1 (positive f1)
        data, _ = self.rom.read(0, 0)
        ta_pos, ta_neg, mask = self.rom.extract_ta_actions_k(data, k=0)
        assert not (ta_pos & 2), "Clause 0 should NOT include positive literal 1"


# ---------------------------------------------------------------------------
# TMAcceleratorEmulator — matches reference inference
# ---------------------------------------------------------------------------

class TestTMAcceleratorEmulator:
    def _check_matches_reference(self, tmir, cfg, features):
        emul = TMAcceleratorEmulator(tmir, cfg)
        trace = emul.run(features)
        X = np.array([features], dtype=np.uint8)
        ref_preds, _ = predict(tmir, X)
        assert trace.predicted_class == int(ref_preds[0]), (
            f"Emulator predicted {trace.predicted_class}, "
            f"reference predicted {int(ref_preds[0])}"
        )
        return trace

    def test_all_zeros_matches_reference(self, tmp_path):
        tmir = _make_tmir()
        cfg = _make_cfg(tmp_path)
        self._check_matches_reference(tmir, cfg, [0, 0, 0, 0])

    def test_all_ones_matches_reference(self, tmp_path):
        tmir = _make_tmir()
        cfg = _make_cfg(tmp_path)
        self._check_matches_reference(tmir, cfg, [1, 1, 1, 1])

    def test_f0_set_only(self, tmp_path):
        tmir = _make_tmir()
        cfg = _make_cfg(tmp_path)
        self._check_matches_reference(tmir, cfg, [1, 0, 0, 0])

    def test_f1_set_only(self, tmp_path):
        tmir = _make_tmir()
        cfg = _make_cfg(tmp_path)
        self._check_matches_reference(tmir, cfg, [0, 1, 0, 0])

    def test_f0_f1_set(self, tmp_path):
        tmir = _make_tmir()
        cfg = _make_cfg(tmp_path)
        self._check_matches_reference(tmir, cfg, [1, 1, 0, 0])

    def test_trace_has_fsm_events(self, tmp_path):
        emul = _make_emulator(tmp_path)
        trace = emul.run([0, 0, 0, 0])
        assert len(trace.fsm_events) >= 4, "Expected at least 4 FSM transitions"
        states = [ev.from_state for ev in trace.fsm_events]
        assert "S_IDLE" in states
        assert "S_RECV" in states
        assert "S_COMPUTE" in states
        assert "S_SCORE" in states

    def test_trace_has_fifo_events(self, tmp_path):
        emul = _make_emulator(tmp_path)
        trace = emul.run([1, 0, 1, 1])
        push_evs = [e for e in trace.fifo_events if isinstance(e, FIFOPushEvent)]
        pop_evs  = [e for e in trace.fifo_events if isinstance(e, FIFOPopEvent)]
        assert len(push_evs) == emul.n_beats
        assert len(pop_evs) == emul.n_beats

    def test_trace_has_clause_final_events(self, tmp_path):
        emul = _make_emulator(tmp_path)
        trace = emul.run([0, 0, 0, 0])
        assert len(trace.clause_final_events) == emul.n_clauses_total

    def test_trace_has_score_vote_events(self, tmp_path):
        emul = _make_emulator(tmp_path)
        trace = emul.run([0, 0, 0, 0])
        assert len(trace.score_vote_events) == emul.n_clauses_total

    def test_trace_has_argmax_event(self, tmp_path):
        emul = _make_emulator(tmp_path)
        trace = emul.run([0, 0, 0, 0])
        assert isinstance(trace.argmax_event, ArgmaxEvent)
        assert trace.argmax_event.predicted_class == trace.predicted_class

    def test_n_rom_reads(self, tmp_path):
        emul = _make_emulator(tmp_path)
        trace = emul.run([0, 0, 0, 0])
        expected = emul.n_feat_slices * emul.n_clause_slices
        assert len(trace.rom_events) == expected

    def test_wrong_feature_length_raises(self, tmp_path):
        emul = _make_emulator(tmp_path)
        with pytest.raises(ValueError, match="4 features"):
            emul.run([0, 1])

    def test_non_vanilla_raises(self, tmp_path):
        tmir = _make_tmir()
        object.__setattr__(tmir, "variant", "coalesced")
        cfg = _make_cfg(tmp_path)
        with pytest.raises(ValueError, match="vanilla"):
            TMAcceleratorEmulator(tmir, cfg)

    def test_multi_beat_matches_reference(self, tmp_path):
        """n_features=40 forces 2 AXI-Stream beats at 32-bit width."""
        tmir = _make_tmir(n_features=40, n_classes=3, n_clauses_pc=6)
        tmir.to_yaml(tmp_path / "m.yaml")
        cfg = TMAcceleratorConfig.model_validate({
            "model_path": str(tmp_path / "m.yaml"),
            "output_dir": str(tmp_path),
            "axis_data_width": 32,
            "fifo_depth": 16,
            "feat_slice": 8,
            "clause_slice": 8,
        })
        features = [i % 2 for i in range(40)]
        self._check_matches_reference(tmir, cfg, features)

    def test_repeated_runs_are_independent(self, tmp_path):
        """Score accumulator must be cleared between runs."""
        emul = _make_emulator(tmp_path)
        t1 = emul.run([1, 0, 0, 0])
        t2 = emul.run([1, 0, 0, 0])
        assert t1.predicted_class == t2.predicted_class
        # Final scores should be identical
        assert t1.argmax_event.scores == t2.argmax_event.scores

    def test_embedded_vectors_match(self, tmp_path):
        """If TMIR has test vectors, emulator should match every expected class."""
        tmir = _make_tmir()
        tmir.verification = Verification(test_vectors=[
            EvalVector(input=[0, 0, 0, 0], expected_class=0, expected_scores=[0, 0]),
            EvalVector(input=[1, 1, 0, 0], expected_class=1, expected_scores=[0, 0]),
        ])
        cfg = _make_cfg(tmp_path)
        emul = TMAcceleratorEmulator(tmir, cfg)
        for vec in tmir.verification.test_vectors:
            trace = emul.run(vec.input)
            X = np.array([vec.input], dtype=np.uint8)
            ref_preds, _ = predict(tmir, X)
            assert trace.predicted_class == int(ref_preds[0])
