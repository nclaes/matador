"""RTL generation and simulation regression tests.

The simulation tests reproduce, end to end, the exact bug chain found while
building this backend (a scrambled weight matrix, a missing `endmodule`, a
duplicate-declaration elaboration failure, an array-to-scalar port
mismatch, an iverilog-incompatible array-sliced port connection, an
undriven adder valid net, a reversed class-index array direction, and a
same-edge testbench/DUT race) -- if any of them regresses, `coal_tm sim`
stops compiling or stops matching the emulator, and these tests catch it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from coal_tm.config import RTLConfig
from coal_tm import rtl, testbench

HAVE_IVERILOG = shutil.which("iverilog") is not None
HAVE_VERILATOR = shutil.which("verilator") is not None

_SOURCES = rtl.GENERATED_SOURCES


def _make_config(tiny_model, tmp_path, test_data: Path | None = None) -> RTLConfig:
    return RTLConfig(
        output_dir=tmp_path / "out",
        tas=tiny_model["tas"],
        weights=tiny_model["weights"],
        classes=tiny_model["classes"],
        clauses=tiny_model["clauses"],
        features=tiny_model["features"],
        bus_width=8,
        test_data=test_data,
    )


def test_generate_writes_every_expected_source(tiny_model, tmp_path):
    config = _make_config(tiny_model, tmp_path)
    result = rtl.generate(config)
    names = {p.name for p in result.sources}
    assert names == set(_SOURCES)
    for name in _SOURCES:
        assert (result.rtl_dir / name).exists()


def test_hard_coded_weight_emits_endmodule(tiny_model, tmp_path):
    config = _make_config(tiny_model, tmp_path)
    result = rtl.generate(config)
    content = (result.rtl_dir / "hard_coded_weight.sv").read_text()
    assert content.strip().endswith("endmodule")


@pytest.mark.skipif(not HAVE_IVERILOG, reason="iverilog not installed")
def test_generated_rtl_compiles_cleanly_under_iverilog(tiny_model, tmp_path):
    config = _make_config(tiny_model, tmp_path)
    result = rtl.generate(config)
    vvp_out = tmp_path / "check.vvp"

    weights = np.loadtxt(config.weights, dtype=int).reshape(config.classes, config.clauses)
    width = rtl._weight_width(weights, config.clauses)

    harness = tmp_path / "harness.sv"
    harness.write_text(
        f"""\
`timescale 1ns/1ps
module harness;
  localparam CLAUSE_NUM = {config.clauses};
  localparam CLASS_NUM = {config.classes};
  logic clk=0, rst=0, valid=0;
  logic [CLAUSE_NUM-1:0] clauses_in = 0;
  logic signed [{width - 1}:0] class_sums [CLASS_NUM];
  logic adder_done;
  Adder_new #(.CLAUSE_NUM(CLAUSE_NUM), .CLASS_NUM(CLASS_NUM), .WEIGHT_LENGTH({width}), .STAGE_NUM(1))
    dut(.clk(clk), .rst(rst), .clauses(clauses_in), .class_sums(class_sums), .valid(valid), .adder_done(adder_done));
endmodule
"""
    )
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp_out), str(harness), *(str(result.rtl_dir / s) for s in _SOURCES)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr


@pytest.mark.skipif(not HAVE_IVERILOG, reason="iverilog not installed")
def test_generated_testbench_matches_emulator_end_to_end(tiny_model, tmp_path):
    """Real simulation: generate RTL + a testbench for 3 vectors (including
    the all-exclude-clause case), compile under iverilog, run it, and
    require every vector to pass -- i.e. the RTL simulation's predicted
    class matches coal_tm.emulator's, for actual hardware, not just the
    Python reference model in isolation."""
    test_data = tmp_path / "test_data.txt"
    test_data.write_text(
        "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"  # x0=1,x1=0 -> clause 0 fires
        "1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"  # x1=1 violates clause 0's ~x1
        "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"  # x0=0 violates clause 0's x0
    )

    config = _make_config(tiny_model, tmp_path, test_data=test_data)
    result = rtl.generate(config)
    tb_result = testbench.generate(config, n_vectors=3)

    vvp_out = tmp_path / "sim.vvp"
    proc = subprocess.run(
        ["iverilog", "-g2012", "-o", str(vvp_out), str(tb_result.testbench_path),
         *(str(result.rtl_dir / s) for s in _SOURCES)],
        cwd=tb_result.stimulus_path.parent, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr

    run = subprocess.run(["vvp", str(vvp_out)], cwd=tb_result.stimulus_path.parent,
                          capture_output=True, text=True, timeout=30)
    assert "FAIL" not in run.stdout, run.stdout
    assert "ALL 3 VECTORS PASSED" in run.stdout, run.stdout


def test_model_specific_files_differ_between_models_static_files_dont(tiny_model, tmp_path):
    """The claims RTL/README.md makes ("this is real, model-specific
    hardware -- not a generic template") verified directly, at the
    precision the README itself draws: TM_Hard_Coded_Clause_Blocks.sv and
    hard_coded_weight.sv are sensitive to the *trained values* (must
    differ between same-shape models with different TAs.txt/weights.txt);
    HCB_top.sv/TM_top.sv/axis_wrapper.sv are sensitive only to *shape*
    (legitimately identical for same-shape models, but must differ when
    clause count differs -- otherwise they'd just be templates too); the
    3 static IP files must stay byte-identical no matter what, and match
    the template source they're copied from."""
    config_a = RTLConfig(
        output_dir=tmp_path / "a",
        tas=tiny_model["tas"], weights=tiny_model["weights"],
        classes=tiny_model["classes"], clauses=tiny_model["clauses"],
        features=tiny_model["features"], bus_width=8,
    )
    result_a = rtl.generate(config_a)

    # Model B: same shape, different trained values -- clause 0 now
    # requires feature 2 (not feature 0/1), and every weight is negated.
    tas_b = tmp_path / "TAs_b.txt"
    weights_b = tmp_path / "weights_b.txt"
    # 2 clauses x 16 features x 2 literals = 64 values total (matching
    # tiny_model's shape).
    tas_states = [0] * 64
    tas_states[4] = 200
    tas_b.write_text("\n".join(str(v) for v in tas_states) + "\n")
    original_weights = [int(v) for v in tiny_model["weights"].read_text().split()]
    weights_b.write_text("\n".join(str(-v) for v in original_weights) + "\n")

    config_b = RTLConfig(
        output_dir=tmp_path / "b",
        tas=tas_b, weights=weights_b,
        classes=tiny_model["classes"], clauses=tiny_model["clauses"],
        features=tiny_model["features"], bus_width=8,
    )
    result_b = rtl.generate(config_b)

    value_sensitive = ["TM_Hard_Coded_Clause_Blocks.sv", "hard_coded_weight.sv"]
    shape_only = ["HCB_top.sv", "TM_top.sv", "axis_wrapper.sv"]
    assert set(value_sensitive) | set(shape_only) == set(rtl.MODEL_SPECIFIC_SOURCES)

    for name in value_sensitive:
        content_a = (result_a.rtl_dir / name).read_text()
        content_b = (result_b.rtl_dir / name).read_text()
        assert content_a != content_b, f"{name} should differ on trained values alone but is identical"

    for name in shape_only:
        content_a = (result_a.rtl_dir / name).read_text()
        content_b = (result_b.rtl_dir / name).read_text()
        assert content_a == content_b, f"{name} is shape-only and models A/B share a shape, but they differ"

    # Model C: different shape -- fewer clauses AND more features, enough
    # to push n_blocks from 2 (16 features / bus_width 8) to 3 (24
    # features), unlike models A/B which only varied trained values at a
    # fixed shape. This is needed because HCB_top.sv's content is driven
    # by n_blocks (packet count), not clause count directly -- clauses
    # only appears there as a passed-through Verilog parameter name, not
    # a baked-in literal -- so a clause-only change (as tried first) left
    # it byte-identical even though it's genuinely shape-derived.
    # TM_top.sv/axis_wrapper.sv do bake clause/feature counts as literals
    # and would differ either way. features must stay > bus_width --
    # single-packet models are rejected by RTLConfig (see its validator).
    tas_c = tmp_path / "TAs_c.txt"
    weights_c = tmp_path / "weights_c.txt"
    tas_c.write_text("\n".join(str(v) for v in [0] * 48) + "\n")  # 1 clause x 24 features x 2
    weights_c.write_text("\n".join(str(v) for v in original_weights[:1]) + "\n")  # 1 class x 1 clause
    config_c = RTLConfig(
        output_dir=tmp_path / "c",
        tas=tas_c, weights=weights_c,
        classes=1, clauses=1, features=24, bus_width=8,
    )
    result_c = rtl.generate(config_c)
    for name in shape_only:
        content_a = (result_a.rtl_dir / name).read_text()
        content_c = (result_c.rtl_dir / name).read_text()
        assert content_a != content_c, f"{name} should differ when model shape differs but is identical"

    for name in rtl._STATIC_TEMPLATES:
        content_a = (result_a.rtl_dir / name).read_bytes()
        content_b = (result_b.rtl_dir / name).read_bytes()
        content_c = (result_c.rtl_dir / name).read_bytes()
        template = (rtl._TEMPLATES_DIR / name).read_bytes()
        assert content_a == content_b == content_c == template, f"{name} should be byte-identical every time but isn't"


def test_readme_reflects_generate_then_testbench(tiny_model, tmp_path):
    config = _make_config(tiny_model, tmp_path)
    result = rtl.generate(config)
    readme = result.readme_path.read_text()

    assert str(tiny_model["clauses"]) in readme
    assert str(tiny_model["features"]) in readme
    assert "Not generated yet" in readme  # no testbench yet

    # The embedded verification counts must be correct, not just present --
    # extract them the same way an RTL engineer copy-pasting the README's
    # own commands would get them.
    weight_count = int((result.rtl_dir / "hard_coded_weight.sv").read_text().count("assign weights["))
    hcb_count = (result.rtl_dir / "TM_Hard_Coded_Clause_Blocks.sv").read_text().count("partial_clause[")
    assert f"expect {weight_count}" in readme
    assert f"expect {hcb_count}" in readme

    test_data = tmp_path / "test_data.txt"
    test_data.write_text("1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    tb_result = testbench.generate(config, n_vectors=1, test_data=test_data)

    readme = tb_result.readme_path.read_text()
    assert "Not generated yet" not in readme
    assert "run_iverilog.sh" in readme
    assert "run_verilator.sh" in readme
    assert "ALL 1 VECTORS PASSED" in readme


def test_readme_keeps_sim_section_across_a_bare_regenerate(tiny_model, tmp_path):
    """Regression test: re-running `generate` alone after `testbench` has
    already run must not wipe the README's "Simulating this bundle"
    section back to the "Not generated yet" placeholder -- tb/sim are
    still sitting on disk, untouched, and the README should keep saying so."""
    config = _make_config(tiny_model, tmp_path)
    rtl.generate(config)
    test_data = tmp_path / "test_data.txt"
    test_data.write_text("1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    tb_result = testbench.generate(config, n_vectors=1, test_data=test_data)
    assert "Not generated yet" not in tb_result.readme_path.read_text()

    regen_result = rtl.generate(config)
    readme = regen_result.readme_path.read_text()
    assert "Not generated yet" not in readme
    assert "ALL 1 VECTORS PASSED" in readme
    assert "Opening the waveforms" in readme


def test_sim_scripts_are_executable_and_reference_every_source(tiny_model, tmp_path):
    config = _make_config(tiny_model, tmp_path)
    rtl.generate(config)
    test_data = tmp_path / "test_data.txt"
    test_data.write_text("1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    tb_result = testbench.generate(config, n_vectors=1, test_data=test_data)

    for script_path in (tb_result.run_iverilog_path, tb_result.run_verilator_path):
        assert script_path.exists()
        assert os.access(script_path, os.X_OK), f"{script_path} is not executable"
        content = script_path.read_text()
        assert content.startswith("#!/usr/bin/env bash")
        for name in rtl.GENERATED_SOURCES:
            assert f"../{name}" in content, f"{script_path.name} doesn't reference {name}"


@pytest.mark.skipif(not HAVE_IVERILOG, reason="iverilog not installed")
def test_run_iverilog_script_works_standalone(tiny_model, tmp_path):
    """Runs the actual generated tb/run_iverilog.sh as a subprocess -- not
    a re-implementation of what it does -- confirming the script itself
    (paths, source list, waveform rename) works for real, the way an RTL
    engineer with no Python/coal_tm installed would use it."""
    config = _make_config(tiny_model, tmp_path)
    rtl.generate(config)
    test_data = tmp_path / "test_data.txt"
    test_data.write_text("1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    tb_result = testbench.generate(config, n_vectors=1, test_data=test_data)

    run = subprocess.run(
        ["bash", str(tb_result.run_iverilog_path)],
        capture_output=True, text=True, timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert "ALL 1 VECTORS PASSED" in run.stdout, run.stdout

    vcd_path = tb_result.run_iverilog_path.parent / "iverilog.vcd"
    assert vcd_path.exists()
    assert vcd_path.stat().st_size > 0
    assert vcd_path.read_text(errors="ignore").startswith("$date")


@pytest.mark.skipif(not HAVE_VERILATOR, reason="verilator not installed")
def test_run_verilator_script_works_standalone(tiny_model, tmp_path):
    config = _make_config(tiny_model, tmp_path)
    rtl.generate(config)
    test_data = tmp_path / "test_data.txt"
    test_data.write_text("1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")
    tb_result = testbench.generate(config, n_vectors=1, test_data=test_data)

    run = subprocess.run(
        ["bash", str(tb_result.run_verilator_path)],
        capture_output=True, text=True, timeout=60,
    )
    assert run.returncode == 0, run.stderr
    assert "ALL 1 VECTORS PASSED" in run.stdout, run.stdout

    vcd_path = tb_result.run_verilator_path.parent / "verilator.vcd"
    assert vcd_path.exists()
    assert vcd_path.stat().st_size > 0
