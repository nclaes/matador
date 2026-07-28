"""Tests for matador.rtl.accelerator — RTL code generation."""

from __future__ import annotations

import re

import numpy as np
import pytest

from matador.config.schema import TMAcceleratorConfig
from matador.ir.tm_ir import (
    Architecture, EvalVector, Hyperparameters, Representation, TMIR, Verification,
)
from matador.rtl.accelerator import TMAccelerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INCLUDE = 200
EXCLUDE = 50


def _make_tmir(n_features=4, n_classes=2, n_clauses_pc=4, threshold=4) -> TMIR:
    n_literals = 2 * n_features
    ta = np.full((n_classes, n_clauses_pc, n_literals), EXCLUDE, dtype=np.int32)
    # Class 0, positive clause 0: include f0
    ta[0, 0, 0] = INCLUDE
    # Class 0, positive clause 1: include f1
    ta[0, 1, 1] = INCLUDE
    # Class 1, positive clause 0: include f0 AND f1
    ta[1, 0, 0] = INCLUDE
    ta[1, 0, 1] = INCLUDE
    # Class 1, negative clause 2: include NOT-f0
    ta[1, 2, n_features] = INCLUDE

    tmir = TMIR(
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
    return tmir


def _make_cfg(tmp_path, axis_dw=32) -> TMAcceleratorConfig:
    # Write a stub TMIR file so model_path exists
    tmir_path = tmp_path / "model.yaml"
    _make_tmir().to_yaml(tmir_path)
    return TMAcceleratorConfig.model_validate({
        "model_path": str(tmir_path),
        "output_dir": str(tmp_path),
        "axis_data_width": axis_dw,
        "fifo_depth": 16,
    })


def _make_accel(tmp_path, **kwargs) -> TMAccelerator:
    tmir = _make_tmir(**kwargs)
    cfg  = _make_cfg(tmp_path)
    return TMAccelerator(tmir, cfg)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_valid(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.axis_data_width == 32
    assert cfg.fifo_depth == 16


def test_config_rejects_bad_axis_width(tmp_path):
    from pydantic import ValidationError
    tmir_path = tmp_path / "m.yaml"
    _make_tmir().to_yaml(tmir_path)
    with pytest.raises(ValidationError, match="32 or 64"):
        TMAcceleratorConfig.model_validate({
            "model_path": str(tmir_path),
            "output_dir": str(tmp_path),
            "axis_data_width": 16,
        })


def test_config_rejects_non_power_of_two_fifo(tmp_path):
    from pydantic import ValidationError
    tmir_path = tmp_path / "m.yaml"
    _make_tmir().to_yaml(tmir_path)
    with pytest.raises(ValidationError, match="power of 2"):
        TMAcceleratorConfig.model_validate({
            "model_path": str(tmir_path),
            "output_dir": str(tmp_path),
            "fifo_depth": 12,
        })


def test_config_rejects_missing_model(tmp_path):
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="does not exist"):
        TMAcceleratorConfig.model_validate({
            "model_path": str(tmp_path / "ghost.yaml"),
            "output_dir": str(tmp_path),
        })


def test_rejects_non_vanilla(tmp_path):
    tmir = _make_tmir()
    object.__setattr__(tmir, "variant", "coalesced")
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match="vanilla"):
        TMAccelerator(tmir, cfg)


# ---------------------------------------------------------------------------
# Parameter derivation
# ---------------------------------------------------------------------------

def test_params_single_beat(tmp_path):
    accel = _make_accel(tmp_path, n_features=4)
    assert accel.n_beats == 1
    assert accel.n_literals == 8
    assert accel.n_clauses_total == 8   # 2 classes * 4 clauses


def test_params_multi_beat_32(tmp_path):
    accel = _make_accel(tmp_path, n_features=40)
    assert accel.n_beats == 2           # ceil(40/32)


def test_params_multi_beat_64(tmp_path):
    tmir = _make_tmir(n_features=40)
    cfg  = _make_cfg(tmp_path, axis_dw=64)
    accel = TMAccelerator(tmir, cfg)
    assert accel.n_beats == 1           # ceil(40/64)


def test_score_width_sufficient(tmp_path):
    # score_width is sized off half the clauses per class (the true
    # worst-case vote magnitude), not threshold -- score_acc.v no longer
    # clamps, so the register must hold the full unclamped sum.
    accel = _make_accel(tmp_path, n_clauses_pc=400, threshold=4)
    half_k = 400 // 2
    assert 2 ** (accel.score_width - 1) > half_k


# ---------------------------------------------------------------------------
# File generation
# ---------------------------------------------------------------------------

def test_generate_creates_all_files(tmp_path):
    accel   = _make_accel(tmp_path)
    rtl_dir = accel.generate()

    expected_src = [
        "axis_fifo.v", "clause_eval.v", "score_acc.v", "argmax.v", "tm_accelerator.v"
    ]
    expected_tb  = [
        "tb_axis_fifo.v", "tb_clause_eval.v", "tb_score_acc.v", "tb_argmax.v", "tb_system.v"
    ]
    expected_sim = [
        "run_iverilog.sh", "lint_verilator.sh", "run_xsim.sh", "waves.sh",
        "tb_system.gtkw", "tb_axis_fifo.gtkw", "tb_clause_eval.gtkw",
        "tb_score_acc.gtkw", "tb_argmax.gtkw",
    ]
    expected_verilator = ["Makefile", "tb_top.cpp"]

    for f in expected_src:
        assert (rtl_dir / "src" / f).exists(), f"missing src/{f}"
    for f in expected_tb:
        assert (rtl_dir / "tb" / f).exists(),  f"missing tb/{f}"
    for f in expected_sim:
        assert (rtl_dir / "sim" / f).exists(), f"missing sim/{f}"
    for f in expected_verilator:
        assert (rtl_dir / "sim" / "verilator" / f).exists(), f"missing sim/verilator/{f}"


def test_top_module_no_sv_constructs(tmp_path):
    accel = _make_accel(tmp_path)
    rtl_dir = accel.generate()
    src = (rtl_dir / "src" / "tm_accelerator.v").read_text()
    # Reject SV keywords
    for kw in ("logic", "always_ff", "always_comb", "interface", "typedef", "struct",
                "$fatal", "unique case"):
        assert kw not in src, f"SystemVerilog keyword {kw!r} found in RTL"


def test_top_module_no_timing_delays(tmp_path):
    accel = _make_accel(tmp_path)
    rtl_dir = accel.generate()
    for fname in ["axis_fifo.v", "clause_eval.v", "score_acc.v", "argmax.v", "tm_accelerator.v"]:
        text = (rtl_dir / "src" / fname).read_text()
        # '#' as a delay: matches `#<digits>` (e.g. #5, #10); not `#(` (parameter port)
        assert not re.search(r'#\s*\d', text), f"Timing delay (#N) found in {fname}"


def test_tile_rom_has_correct_tile_count(tmp_path):
    accel = _make_accel(tmp_path)
    rom_text = accel._tile_rom_init()
    lines = [l for l in rom_text.splitlines() if "tile_rom[" in l]
    assert len(lines) == accel.n_tiles


def test_tile_rom_correct_bit_width(tmp_path):
    accel = _make_accel(tmp_path)
    rom_text = accel._tile_rom_init()
    for line in rom_text.splitlines():
        if "tile_rom[" in line:
            m = re.search(r"(\d+)'h([0-9A-Fa-f]+)", line)
            assert m, f"No hex literal found in: {line}"
            assert int(m.group(1)) == accel.tile_width
            assert len(m.group(2)) == (accel.tile_width + 3) // 4


def test_include_bits_match_model(tmp_path):
    """Tile 0 (fi=0, ci=0) clause-0 slot should have literal-0 set and literal-1 clear."""
    tmir = _make_tmir()
    cfg  = _make_cfg(tmp_path)
    accel = TMAccelerator(tmir, cfg)
    rom_text = accel._tile_rom_init()

    # Tile address 0 corresponds to (feat_slice=0, clause_slice=0)
    first_line = next(l for l in rom_text.splitlines() if "tile_rom[    0]" in l)
    m = re.search(r"'h([0-9A-Fa-f]+)", first_line)
    tile_val = int(m.group(1), 16)
    FS = accel.feat_slice

    # Clause 0 (k=0 within tile): positive literal 0 is at bit k*2*FS + 0 = 0
    assert tile_val & (1 << 0), "Literal 0 should be included in tile-0 clause-0 slot"
    # Literal 1 should not be included (clause 0 only includes literal 0)
    assert not (tile_val & (1 << 1)), "Literal 1 should NOT be included in tile-0 clause-0 slot"


def test_beat_packing_single_beat(tmp_path):
    accel = _make_accel(tmp_path, n_features=4)
    # Features [1,0,1,1] → word = 0b1101 = 13
    beats = accel._pack_beats([1, 0, 1, 1])
    assert beats == [0b1101]


def test_beat_packing_multi_beat(tmp_path):
    tmir  = _make_tmir(n_features=4)  # n_features=4 but we override axis_dw
    cfg   = _make_cfg(tmp_path, axis_dw=32)
    accel = TMAccelerator(tmir, cfg)
    # n_features=4 < 32 → 1 beat; still verify packing is correct
    beats = accel._pack_beats([0, 1, 0, 1])
    assert len(beats) == 1
    assert beats[0] == 0b1010


def test_system_tb_embeds_test_vectors(tmp_path):
    tmir = _make_tmir()
    # Embed a known test vector
    tmir.verification = Verification(test_vectors=[
        EvalVector(input=[0, 0, 0, 0], expected_class=0, expected_scores=[0, 0]),
    ])
    cfg  = _make_cfg(tmp_path)
    accel = TMAccelerator(tmir, cfg)
    rtl_dir = accel.generate()

    tb_text = (rtl_dir / "tb" / "tb_system.v").read_text()
    # N_TESTS should reflect the embedded vector
    assert "N_TESTS         = 1" in tb_text
    # Expected class should appear
    assert "tv_exp[0] = " in tb_text


def test_system_tb_no_vectors_skips(tmp_path):
    tmir = _make_tmir()  # no verification block
    cfg  = _make_cfg(tmp_path)
    accel = TMAccelerator(tmir, cfg)
    rtl_dir = accel.generate()
    tb_text = (rtl_dir / "tb" / "tb_system.v").read_text()
    assert "N_TESTS         = 0" in tb_text
    assert "skipping" in tb_text


def test_iverilog_script_references_all_sources(tmp_path):
    accel   = _make_accel(tmp_path)
    rtl_dir = accel.generate()
    script  = (rtl_dir / "sim" / "run_iverilog.sh").read_text()
    for src in ["axis_fifo.v", "clause_eval.v", "score_acc.v", "argmax.v", "tm_accelerator.v"]:
        assert src in script


def test_verilator_script_top_module(tmp_path):
    accel   = _make_accel(tmp_path)
    rtl_dir = accel.generate()
    script  = (rtl_dir / "sim" / "lint_verilator.sh").read_text()
    assert "tm_accelerator" in script
    assert "--lint-only" in script
