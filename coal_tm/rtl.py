"""Generate Coalesced-TM RTL from an existing TAs.txt/weights.txt pair.

Cleaned-up, bug-fixed port of MATADOR_main.py's inline coalesced_tm_write_*
functions. Deliberately independent of coal_tm.train — generating straight
from externally-supplied TAs/weights files (no training run in this tool at
all) is a first-class path, not a special case.

Fixes relative to the original:
  - Weight matrix shape/order comes from coal_tm.artifacts (one place,
    consistently used for both write and read) instead of a hand-rolled
    reshape with swapped (classes, clauses) arguments that silently
    scrambled which weight belonged to which (class, clause) pair.
  - hard_coded_weight.sv's weight width and every downstream WEIGHT_LENGTH
    parameter (axis_wrapper.sv, TM_top.sv, Adder_new, classify) are now the
    *same* computed width (see _weight_width) instead of two independently
    computed values that could disagree and silently truncate weights or
    overflow the class-sum accumulator.
  - write_weights() now actually emits `endmodule` (the original silently
    produced a truncated, unclosed module).
  - TM_top.sv no longer declares `argmax`/`adder_done`/`adder_en` twice
    (once as ports, once again as plain `logic`, a copy-paste artifact
    that fails elaboration), no longer carries ~26 dead signals
    (partial_clause_reg_0..12/valid_reg_0..13) left over from before
    HCB_top.sv was factored out, and wires Adder_new's `valid` input to
    the signal that's actually driven (`adder_en`) instead of an
    undriven, floating net the adder could never actually start from.
  - `partial_clause` is a flat [CLAUSE_NUM-1:0] vector throughout, not a
    `[CLASS_NUM]`-indexed array with only index 0 ever driven — Coalesced
    TM's clause bank is shared across classes (only the weights are
    per-class), so the array dimension was never meaningful; it also made
    HCB_top.sv's array connect into Adder_new's plain vector port, an
    array-to-scalar mismatch.
  - Output paths are resolved relative to output_dir / this package, not
    the working directory, and template copies are checked for success.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from coal_tm import artifacts
from coal_tm.config import RTLConfig

_STATIC_TEMPLATES = ["AXI_Interface.sv", "new_adder.sv", "TM_argmax.sv"]
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "utils" / "RTL_templates"

# The complete set of source files a generated bundle needs to compile,
# filenames only (relative to RTL/) -- the single source of truth for
# coal_tm.cli's `sim` command and the standalone run_iverilog.sh/
# run_verilator.sh scripts coal_tm.testbench writes, so they can't drift
# out of sync with each other or with what generate() actually produces.
# The first 5 are regenerated fresh per model (see MODEL_SPECIFIC_SOURCES
# below); the last 3 are static IP, byte-identical to utils/RTL_templates/
# every time (see copy_static_templates).
MODEL_SPECIFIC_SOURCES = [
    "TM_Hard_Coded_Clause_Blocks.sv",
    "HCB_top.sv",
    "TM_top.sv",
    "axis_wrapper.sv",
    "hard_coded_weight.sv",
]
GENERATED_SOURCES = MODEL_SPECIFIC_SOURCES + list(_STATIC_TEMPLATES)


@dataclass
class RTLArtifacts:
    rtl_dir: Path
    sources: list[Path] = field(default_factory=list)
    readme_path: Path | None = None


def _weight_width(weights: np.ndarray, clauses: int) -> int:
    """Bits for one signed weight, wide enough that summing up to `clauses`
    of them (Adder_new's accumulator) cannot overflow."""
    max_abs = int(np.abs(weights).max()) if weights.size else 0
    weight_bits = max_abs.bit_length() + 1  # magnitude + sign bit
    growth_bits = max(1, math.ceil(math.log2(max(clauses, 2))))
    return weight_bits + growth_bits + 1


def _twos_complement(value: int, bits: int) -> str:
    return format(value & ((1 << bits) - 1), f"0{bits}b")


def _clause_literal_terms(include_row: np.ndarray, lit_range: int) -> list[str]:
    terms = []
    for k in range(lit_range):
        if include_row[k]:
            feat = k // 2
            terms.append(f"x[{feat}]" if k % 2 == 0 else f"~x[{feat}]")
    return terms


def write_hard_coded_clause_blocks(
    path: Path,
    includes: np.ndarray,
    clauses: int,
    features: int,
    bus_width: int,
    clause_expressions_path: Path | None = None,
) -> int:
    """includes: (clauses, 2*features) 0/1 array (coal_tm.artifacts.read_tas_includes).
    Returns the number of HCB blocks (packets) written."""
    all_exclude = ~includes.astype(bool).any(axis=1)

    if clause_expressions_path is not None:
        with open(clause_expressions_path, "w") as f:
            for j in range(clauses):
                terms = _clause_literal_terms(includes[j], features * 2)
                print(f"clause {j}: {' & '.join(terms)}", file=f)

    n_blocks = math.ceil(features / bus_width)
    with open(path, "w") as f:
        print("`timescale 1ns / 1ps\n", file=f)
        starting = 0
        for i in range(n_blocks):
            block_features = min(bus_width, features - i * bus_width)
            lit_range = block_features * 2
            block = includes[:, starting:starting + lit_range]
            starting += lit_range

            if i == 0:
                print(f"module HCB_{i} (x, partial_clause, clk, rst, valid);", file=f)
                print(f"\toutput\tlogic [{clauses - 1}:0] partial_clause;", file=f)
            else:
                print(f"module HCB_{i} (x, partial_clause, partial_clause_prev, clk, rst, valid);", file=f)
                print(f"\tinput\tlogic [{clauses - 1}:0] partial_clause_prev;", file=f)
                print(f"\toutput\tlogic [{clauses - 1}:0] partial_clause;", file=f)
            print("\tinput\tlogic clk;", file=f)
            print("\tinput\tlogic rst;", file=f)
            print(f"\tinput\tlogic [{bus_width - 1}:0] x;", file=f)
            print("\tinput\tlogic valid;", file=f)
            print("\talways @(posedge clk) begin", file=f)
            print("\t\tif (rst) begin", file=f)
            print(f"\t\t\tpartial_clause <= {{{clauses}{{1'b0}}}};", file=f)
            print("\t\tend", file=f)
            print("\t\telse if (valid) begin", file=f)
            for j in range(clauses):
                terms = _clause_literal_terms(block[j], lit_range)
                if not terms:
                    if i == 0:
                        value = "1'b0" if all_exclude[j] else "1'b1"
                        print(f"\t\t\tpartial_clause[{j}] \t<= {value};", file=f)
                    else:
                        print(f"\t\t\tpartial_clause[{j}] \t<= partial_clause_prev[{j}] & 1'b1;", file=f)
                else:
                    expr = " & ".join(terms)
                    if i == 0:
                        print(f"\t\t\tpartial_clause[{j}] \t<= {expr};", file=f)
                    else:
                        print(f"\t\t\tpartial_clause[{j}] \t<= partial_clause_prev[{j}] & {expr};", file=f)
            print("\t\tend", file=f)
            print("\tend", file=f)
            print("endmodule\n", file=f)
    return n_blocks


def write_hcb_top(path: Path, n_blocks: int, clauses: int) -> None:
    with open(path, "w") as f:
        print("`timescale 1ns / 1ps\n", file=f)
        print("module HCB_top #(", file=f)
        print("\tparameter PACKETS_NUM,", file=f)
        print("\tparameter CLAUSE_NUM,", file=f)
        print("\tparameter C_S00_AXIS_TDATA_WIDTH", file=f)
        print("\t)", file=f)
        print("\t(", file=f)
        print("\tinput clk,", file=f)
        print("\tinput rst,", file=f)
        print("\tinput [C_S00_AXIS_TDATA_WIDTH - 1:0] x,", file=f)
        print("\tinput valid,", file=f)
        print("\toutput HCB_done,", file=f)
        print("\toutput [CLAUSE_NUM - 1:0] partial_clause", file=f)
        print("\t);", file=f)

        regs = [f"partial_clause_reg_{i}" for i in range(n_blocks)]
        for reg in regs:
            print(f"\tlogic [CLAUSE_NUM - 1:0] {reg};", file=f)
        print(f"\tassign partial_clause = {regs[-1]};\n", file=f)

        print(
            "\t// Registered packet counter + combinational decode, replacing an\n"
            "\t// earlier one-hot rotate design where the select signal was derived\n"
            "\t// combinationally from a register updating on the SAME edge an HCB_i\n"
            "\t// instance (itself posedge-triggered) read it -- a same-edge race whose\n"
            "\t// outcome depended on simulator scheduling order, observed as clause\n"
            "\t// accumulation lagging by one full inference. Here, live_ctr tracks the\n"
            "\t// position of the packet CURRENTLY on x; every cycle, that position\n"
            "\t// (and the data itself) is registered into sel_ctr/x_reg/sel_valid\n"
            "\t// together, one pipeline stage later. HT_en_ctrl is then a purely\n"
            "\t// combinational decode of sel_ctr -- since sel_ctr only changes via NBA\n"
            "\t// and is stable for the entire following cycle, any posedge-triggered\n"
            "\t// reader sees a value that has already been settled since the previous\n"
            "\t// edge, never one changing on the edge it's read.", file=f,
        )
        ctr_bits = max(1, math.ceil(math.log2(n_blocks))) if n_blocks > 1 else 1
        print(f"\tlogic [{ctr_bits - 1}:0] live_ctr;", file=f)
        print(f"\tlogic [{ctr_bits - 1}:0] sel_ctr;", file=f)
        print("\tlogic sel_valid;", file=f)
        print(f"\tlogic [C_S00_AXIS_TDATA_WIDTH - 1:0] x_reg;", file=f)
        print("\talways @(posedge clk) begin", file=f)
        print("\t\tif (rst) begin", file=f)
        print("\t\t\tlive_ctr  <= '0;", file=f)
        print("\t\t\tsel_ctr   <= '0;", file=f)
        print("\t\t\tsel_valid <= 1'b0;", file=f)
        print("\t\t\tx_reg     <= '0;", file=f)
        print("\t\tend", file=f)
        print("\t\telse begin", file=f)
        print("\t\t\tsel_ctr   <= live_ctr;", file=f)
        print("\t\t\tx_reg     <= x;", file=f)
        print("\t\t\tsel_valid <= valid;", file=f)
        print("\t\t\tif (valid) begin", file=f)
        print(f"\t\t\t\tlive_ctr <= (live_ctr == {n_blocks - 1}) ? '0 : live_ctr + 1'b1;", file=f)
        print("\t\t\tend", file=f)
        print("\t\tend", file=f)
        print("\tend", file=f)
        print("\tlogic [PACKETS_NUM - 1:0] HT_en_ctrl;", file=f)
        print("\tgenerate", file=f)
        print("\t\tgenvar gi;", file=f)
        print("\t\tfor (gi = 0; gi < PACKETS_NUM; gi = gi + 1) begin", file=f)
        print(f"\t\t\tassign HT_en_ctrl[gi] = sel_valid && (sel_ctr == gi);", file=f)
        print("\t\tend", file=f)
        print("\tendgenerate", file=f)
        print(
            "\n\t// HT_en_ctrl[PACKETS_NUM-1] is the TRIGGER for HCB_inst_{last}'s own\n"
            "\t// write of partial_clause_reg_{last} (that instance's always block reacts\n"
            "\t// to it one cycle later, same as every other HCB_i). HCB_done must signal\n"
            "\t// once that write has actually landed, not on the trigger cycle itself --\n"
            "\t// otherwise every downstream consumer (Adder_new via adder_en) samples\n"
            "\t// partial_clause one full inference before its true final value is ready.\n"
            "\t// One more register stage here closes exactly that gap.",
            file=f,
        )
        print("\tlogic HCB_done_reg;", file=f)
        print("\talways @(posedge clk) begin", file=f)
        print("\t\tif (rst) begin", file=f)
        print("\t\t\tHCB_done_reg <= 1'b0;", file=f)
        print("\t\tend", file=f)
        print("\t\telse begin", file=f)
        print("\t\t\tHCB_done_reg <= HT_en_ctrl[PACKETS_NUM - 1];", file=f)
        print("\t\tend", file=f)
        print("\tend", file=f)
        print("\tassign HCB_done = HCB_done_reg;\n", file=f)

        print(f"\tHCB_0 HCB_inst_0(.clk(clk), .rst(rst), .x(x_reg), .valid(HT_en_ctrl[0]), .partial_clause({regs[0]}));\n", file=f)
        for i in range(1, n_blocks):
            print(
                f"\tHCB_{i} HCB_inst_{i}(.clk(clk), .rst(rst), .x(x_reg), .valid(HT_en_ctrl[{i}]), "
                f".partial_clause_prev({regs[i - 1]}), .partial_clause({regs[i]}));\n",
                file=f,
            )
        print("endmodule", file=f)


def write_tm_top(path: Path, clauses: int, weight_width: int) -> None:
    with open(path, "w") as f:
        print("`timescale 1ns / 1ps\n", file=f)
        print("module Hard_Coded_Inference_Top #(", file=f)
        print("\tparameter STAGE_NUM,", file=f)
        print("\tparameter CLAUSE_NUM,", file=f)
        print("\tparameter CLASS_NUM,", file=f)
        print("\tparameter WEIGHT_LENGTH,", file=f)
        print("\tparameter C_S00_AXIS_TDATA_WIDTH,", file=f)
        print("\tparameter C_M00_AXIS_TDATA_WIDTH,", file=f)
        print("\tparameter PACKETS_NUM", file=f)
        print(") (", file=f)
        print("\tinput  logic clk,", file=f)
        print("\tinput  logic rst,", file=f)
        print("\tinput  logic [C_S00_AXIS_TDATA_WIDTH - 1:0] x,", file=f)
        print("\tinput  logic valid,", file=f)
        print("\tinput  logic last,", file=f)
        print("\toutput logic s_axis_tready,", file=f)
        print("\tinput  logic m00_axis_tready,", file=f)
        print("\tinput  logic [C_M00_AXIS_TDATA_WIDTH-1:0] packet_counter,", file=f)
        print("\toutput logic [CLAUSE_NUM - 1:0] clauses,", file=f)
        print("\toutput logic signed [WEIGHT_LENGTH - 1:0] class_sums [CLASS_NUM],", file=f)
        print("\toutput logic adder_en,", file=f)
        print("\toutput logic adder_done,", file=f)
        print("\toutput logic argmax,", file=f)
        print("\toutput logic finish,", file=f)
        print("\toutput logic last_out,", file=f)
        print("\toutput logic [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep,", file=f)
        print("\toutput logic [C_M00_AXIS_TDATA_WIDTH-1:0] y", file=f)
        print(");", file=f)
        print(f"\tlogic [{clauses - 1}:0] partial_clause;", file=f)
        print("\tlogic HCB_done;", file=f)
        print("\tassign clauses = partial_clause;\n", file=f)

        print("\tHCB_top #(", file=f)
        print("\t\t.PACKETS_NUM(PACKETS_NUM),", file=f)
        print("\t\t.CLAUSE_NUM(CLAUSE_NUM),", file=f)
        print("\t\t.C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH)", file=f)
        print("\t) HT (", file=f)
        print("\t\t.clk(clk), .rst(rst), .x(x), .valid(valid),", file=f)
        print("\t\t.HCB_done(HCB_done), .partial_clause(partial_clause)", file=f)
        print("\t);\n", file=f)

        print("\t// One cycle after the last packet, kick off the weighted sum;", file=f)
        print("\t// argmax fires once that sum is ready and the output side is free.", file=f)
        print("\talways @(posedge clk) begin", file=f)
        print("\t\tadder_en <= HCB_done;", file=f)
        print("\t\targmax   <= adder_done && m00_axis_tready;", file=f)
        print("\tend\n", file=f)

        print(
            "\t// Backpressure: HCB_done -> adder_en -> adder_done -> argmax takes 3\n"
            "\t// registered cycles (4 including HCB_done's own edge), fixed by this\n"
            "\t// pipeline's structure. Without stalling new input for that long,\n"
            "\t// consecutive vectors that arrive faster than the pipeline drains --\n"
            "\t// which happens whenever PACKETS_NUM is small, e.g. a model whose\n"
            "\t// features fit in one bus_width-wide packet -- start a new HCB_done\n"
            "\t// before the previous inference's argmax has been read, corrupting\n"
            "\t// results. s_axis_tready deasserts to stall new packets (valid =\n"
            "\t// tvalid && tready, computed upstream in axis_wrapper.sv) until the\n"
            "\t// pipeline is clear. busy is combinational on HCB_done itself, not\n"
            "\t// just the registered countdown -- gating tready only from the cycle\n"
            "\t// *after* HCB_done leaves a one-cycle gap where the very next packet\n"
            "\t// (which would immediately retrigger HCB_done, colliding with the\n"
            "\t// in-flight inference) still gets accepted.", file=f,
        )
        print("\tlogic [2:0] busy_countdown;", file=f)
        print("\tlogic busy;", file=f)
        print("\tassign busy = (busy_countdown != 0) || HCB_done;", file=f)
        print("\tassign s_axis_tready = !busy;", file=f)
        print("\talways @(posedge clk) begin", file=f)
        print("\t\tif (rst) begin", file=f)
        print("\t\t\tbusy_countdown <= 0;", file=f)
        print("\t\tend", file=f)
        print("\t\telse if (HCB_done && busy_countdown == 0) begin", file=f)
        print("\t\t\tbusy_countdown <= 4;", file=f)
        print("\t\tend", file=f)
        print("\t\telse if (busy_countdown > 0) begin", file=f)
        print("\t\t\tbusy_countdown <= busy_countdown - 3'd1;", file=f)
        print("\t\tend", file=f)
        print("\tend\n", file=f)

        print("\tAdder_new #(", file=f)
        print("\t\t.STAGE_NUM(STAGE_NUM), .CLAUSE_NUM(CLAUSE_NUM),", file=f)
        print("\t\t.CLASS_NUM(CLASS_NUM), .WEIGHT_LENGTH(WEIGHT_LENGTH)", file=f)
        print("\t) add_inst (", file=f)
        print("\t\t.clk(clk), .rst(rst), .clauses(partial_clause),", file=f)
        print("\t\t.class_sums(class_sums), .valid(adder_en), .adder_done(adder_done)", file=f)
        print("\t);\n", file=f)

        print("\tclassify #(", file=f)
        print("\t\t.CLASS_NUM(CLASS_NUM), .WEIGHT_LENGTH(WEIGHT_LENGTH),", file=f)
        print("\t\t.C_M00_AXIS_TDATA_WIDTH(C_M00_AXIS_TDATA_WIDTH)", file=f)
        print("\t) classify_inst (", file=f)
        print("\t\t.c_sum(class_sums), .y(y), .last(last), .m00_axis_tready(m00_axis_tready),", file=f)
        print("\t\t.m00_axis_tkeep(m00_axis_tkeep), .its_business_time(argmax),", file=f)
        print("\t\t.finish(finish), .last_out(last_out), .clk(clk), .rst(rst)", file=f)
        print("\t);\n", file=f)
        print("endmodule", file=f)


def write_axis_wrapper(
    path: Path,
    bus_width: int,
    n_blocks: int,
    adder_stages: int,
    clauses: int,
    classes: int,
    features: int,
    weight_width: int,
) -> None:
    with open(path, "w") as f:
        print("module axis_wrapper_top #(", file=f)
        print("\tparameter integer PACKETS = %d," % n_blocks, file=f)
        print("\tparameter integer C_S00_AXIS_TDATA_WIDTH = %d," % bus_width, file=f)
        print("\tparameter integer C_M00_AXIS_TDATA_WIDTH = %d," % bus_width, file=f)
        print("\tparameter STAGE_NUM = %d," % adder_stages, file=f)
        print("\tparameter CLAUSE_NUM = %d," % clauses, file=f)
        print("\tparameter CLASS_NUM = %d," % classes, file=f)
        print("\tparameter WEIGHT_LENGTH = %d," % weight_width, file=f)
        print("\tparameter FEATURE_NUM = %d," % features, file=f)
        print("\tparameter PACKETS_NUM = (FEATURE_NUM - 1)/C_S00_AXIS_TDATA_WIDTH + 1", file=f)
        print("\t) (", file=f)
        print(
            """
    input  wire    s00_axis_aclk,
    input  wire    s00_axis_aresetn,
    input  wire    [C_S00_AXIS_TDATA_WIDTH-1 : 0] s00_axis_tdata,
    input  wire    [(C_S00_AXIS_TDATA_WIDTH/8)-1 : 0] s00_axis_tstrb,
    input  wire    s00_axis_tlast,
    input  wire    s00_axis_tvalid,
    output wire    s00_axis_tready,
    output wire [CLAUSE_NUM - 1:0] clauses,
    output logic signed [WEIGHT_LENGTH-1:0] class_sums [CLASS_NUM],
    output logic adder_en,
    output logic adder_done,
    output logic argmax,
    input  wire  m00_axis_aclk,
    input  wire  m00_axis_aresetn,
    input  wire  m00_axis_tready,
    output wire  m00_axis_tvalid,
    output reg   [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep,
    output wire  [C_M00_AXIS_TDATA_WIDTH-1 : 0] m00_axis_tdata,
    output wire  [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tstrb,
    output wire  m00_axis_tlast
    );

    logic inference_complete;
    logic last_complete;
    logic valid;

    assign m00_axis_tvalid = inference_complete;
    assign m00_axis_tlast  = last_complete;
    // s00_axis_tready is driven by the tm instance below (real
    // backpressure -- see Hard_Coded_Inference_Top's busy_countdown).
    assign valid            = s00_axis_tvalid && s00_axis_tready;

    Hard_Coded_Inference_Top #(
        .STAGE_NUM(STAGE_NUM),
        .CLAUSE_NUM(CLAUSE_NUM),
        .CLASS_NUM(CLASS_NUM),
        .WEIGHT_LENGTH(WEIGHT_LENGTH),
        .C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH),
        .C_M00_AXIS_TDATA_WIDTH(C_M00_AXIS_TDATA_WIDTH),
        .PACKETS_NUM(PACKETS_NUM)
    )
    tm (
        .x(s00_axis_tdata),
        .clk(m00_axis_aclk),
        .rst(~s00_axis_aresetn),
        .valid(valid),
        .s_axis_tready(s00_axis_tready),
        .m00_axis_tready(m00_axis_tready),
        .m00_axis_tkeep(m00_axis_tkeep),
        .packet_counter(),
        .y(m00_axis_tdata),
        .clauses(clauses),
        .class_sums(class_sums),
        .adder_en(adder_en),
        .adder_done(adder_done),
        .argmax(argmax),
        .finish(inference_complete),
        .last(s00_axis_tlast),
        .last_out(last_complete)
    );

endmodule
""",
            file=f,
        )


def write_weights(path: Path, weights: np.ndarray, width: int) -> None:
    """weights: (classes, clauses) signed int array — coal_tm.artifacts'
    canonical shape, so no transpose/reshape guesswork here."""
    classes, clauses = weights.shape
    with open(path, "w") as f:
        print("module hard_coded_weight #(", file=f)
        print("\tparameter CLAUSE_NUM", file=f)
        print("\t)", file=f)
        print("\t(", file=f)
        print(f"\toutput logic signed [{width - 1}:0] weights [{classes}][CLAUSE_NUM]);\n", file=f)
        for c in range(classes):
            for clause in range(clauses):
                bits = _twos_complement(int(weights[c][clause]), width)
                print(f"\tassign weights[{c}][{clause}] \t=\t{width}'b{bits};", file=f)
        print("endmodule", file=f)


def copy_static_templates(dest_dir: Path) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in _STATIC_TEMPLATES:
        src = _TEMPLATES_DIR / name
        if not src.exists():
            raise FileNotFoundError(f"missing static RTL template: {src}")
        dst = dest_dir / name
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def generate(config: RTLConfig) -> RTLArtifacts:
    rtl_dir = config.output_dir / "RTL"
    rtl_dir.mkdir(parents=True, exist_ok=True)

    includes = artifacts.read_tas_includes(config.tas, config.clauses, config.features)
    weights = artifacts.read_weights(config.weights, config.classes, config.clauses)
    width = _weight_width(weights, config.clauses)

    sources: list[Path] = []

    hcb_blocks_path = rtl_dir / "TM_Hard_Coded_Clause_Blocks.sv"
    n_blocks = write_hard_coded_clause_blocks(
        hcb_blocks_path, includes, config.clauses, config.features, config.bus_width,
        clause_expressions_path=config.output_dir / "raw_clause_expressions.txt",
    )
    sources.append(hcb_blocks_path)

    hcb_top_path = rtl_dir / "HCB_top.sv"
    write_hcb_top(hcb_top_path, n_blocks, config.clauses)
    sources.append(hcb_top_path)

    tm_top_path = rtl_dir / "TM_top.sv"
    write_tm_top(tm_top_path, config.clauses, width)
    sources.append(tm_top_path)

    axis_wrapper_path = rtl_dir / "axis_wrapper.sv"
    write_axis_wrapper(
        axis_wrapper_path, config.bus_width, n_blocks, config.adder_stages,
        config.clauses, config.classes, config.features, width,
    )
    sources.append(axis_wrapper_path)

    weights_path = rtl_dir / "hard_coded_weight.sv"
    write_weights(weights_path, weights, width)
    sources.append(weights_path)

    sources.extend(copy_static_templates(rtl_dir))

    from coal_tm.readme import write_rtl_readme  # local import: readme.py imports this module
    readme_path = write_rtl_readme(rtl_dir, config, width, n_blocks)

    return RTLArtifacts(rtl_dir=rtl_dir, sources=sources, readme_path=readme_path)
