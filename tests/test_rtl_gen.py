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

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from coal_tm.config import RTLConfig
from coal_tm import rtl, testbench

HAVE_IVERILOG = shutil.which("iverilog") is not None

_SOURCES = [
    "TM_Hard_Coded_Clause_Blocks.sv", "HCB_top.sv", "TM_top.sv", "axis_wrapper.sv",
    "AXI_Interface.sv", "new_adder.sv", "TM_argmax.sv", "hard_coded_weight.sv",
]


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
