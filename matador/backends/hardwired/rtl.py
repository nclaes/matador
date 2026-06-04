"""Hardwired backend RTL generator — HCB streaming architecture.

Implements the Hard Coded Block (HCB) style from the original Matador
utils/legacty_scripts/TM_RTL_gen.py, redesigned for parameterised
TMIR-driven generation.

Architecture
------------
Features arrive over AXI-Stream one beat (AXIS_DATA_WIDTH bits) at a
time.  For each beat, every clause's partial result is AND-reduced with
the contribution of that beat's feature literals.  This is exactly the
HCB chain from the original design:

  AXI beat 0 → hcb_result &= contribution_beat_0(fifo_m_tdata)
  AXI beat 1 → hcb_result &= contribution_beat_1(fifo_m_tdata)
  ...
  AXI beat N-1 → hcb_result &= contribution_beat_{N-1}(fifo_m_tdata)

After all N beats, hcb_result[g] = 1 iff clause g fires on this input.
A balanced binary adder tree then accumulates per-class scores.

No feature buffer.  The clause evaluation pipeline depth equals N_BEATS
(one registered AND stage per AXI beat), plus pipeline_stages for the
adder tree.

Generated files
---------------
  src/hcb_chain.v          — HCB per-beat clause contribution logic
  src/hw_score.v           — adder trees + argmax
  src/hw_tm_accelerator.v  — top-level FSM + integration
  src/axis_fifo.v          — shared synchronous FIFO
  tb/tb_hcb_chain.v        — unit test for HCB clause eval
  tb/tb_hw_system.v        — full system testbench
  sim/run_iverilog.sh      — compile + simulate
  sim/lint_verilator.sh    — Verilator lint
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

from matador.backends.base import ResourceEstimate, RTLArtifacts, RTLBackend


# ---------------------------------------------------------------------------
# Backend entry point
# ---------------------------------------------------------------------------

class HardwiredBackend(RTLBackend):
    """Hardwired HCB-streaming TM accelerator."""

    @property
    def name(self) -> str:
        return "hardwired"

    @property
    def config_class(self) -> type:
        from matador.backends.hardwired.config import HardwiredAcceleratorConfig
        return HardwiredAcceleratorConfig

    @property
    def emulator_class(self):
        from matador.backends.hardwired.emulator import HardwiredEmulator
        return HardwiredEmulator

    def generate(self, tmir, config) -> RTLArtifacts:
        if tmir.architecture.clause_organization != "per_class":
            raise ValueError(
                "The hardwired backend currently supports only per_class "
                "clause organisation (vanilla TM)."
            )
        gen = _HardwiredGenerator(tmir, config)
        return gen.run()

    def resource_estimate(self, tmir, config) -> ResourceEstimate:
        arch = tmir.architecture
        tmir.to_includes()
        n_inc = int(tmir.representation.includes.sum())
        return ResourceEstimate(
            notes=(
                f"{n_inc} include bits across {arch.n_clauses_total} clauses. "
                f"HCB pipeline depth: {math.ceil(arch.n_features / config.axis_data_width)} beats. "
                "Run Vivado for exact LUT/FF counts."
            )
        )


# ---------------------------------------------------------------------------
# Internal generator
# ---------------------------------------------------------------------------

class _HardwiredGenerator:
    def __init__(self, tmir, config):
        tmir.to_includes()
        arch = tmir.architecture

        self.tmir   = tmir
        self.config = config

        self.N  = arch.n_features
        self.C  = arch.n_classes
        self.K  = arch.n_clauses_per_class
        self.CT = arch.n_clauses_total
        self.DW = config.axis_data_width
        self.PS = config.pipeline_stages

        self.HALF_K  = self.K // 2
        self.N_BEATS = math.ceil(self.N / self.DW)
        self.SCORE_W = max(2, math.ceil(math.log2(self.HALF_K + 1)) + 2)
        self.CLASS_W = max(1, math.ceil(math.log2(self.C)))
        self.TREE_DEPTH = max(1, math.ceil(math.log2(max(self.HALF_K, 2))))
        self.actual_ps  = min(self.PS, self.TREE_DEPTH)
        self._reg_levels = self._compute_reg_levels()

        # includes: (CT, 2*N) bool
        self.includes = tmir.representation.includes.reshape(self.CT, 2 * self.N).astype(bool)

        # Pre-compute HCB_INIT: 1 for clauses with any included literal
        self._hcb_init = 0
        for g in range(self.CT):
            if self.includes[g].any():
                self._hcb_init |= (1 << g)

    # ------------------------------------------------------------------ helpers

    def _compute_reg_levels(self) -> set:
        p, d = self.actual_ps, self.TREE_DEPTH
        if p <= 0:
            return set()
        if p >= d:
            return set(range(d))
        step = d / p
        return {int((i + 0.5) * step) for i in range(p)}

    # ------------------------------------------------------------------ HCB chain module

    def _gen_hcb_chain(self) -> str:
        """Generate hcb_chain.v — per-beat clause contribution wires + accumulation."""
        N, DW, CT, NB = self.N, self.DW, self.CT, self.N_BEATS
        init_hex = f"{self._hcb_init:0{(CT + 3) // 4}X}"
        bcw = max(1, math.ceil(math.log2(NB + 1)))

        lines = []
        lines.append("`timescale 1ns/1ps")
        lines.append("// hcb_chain — Hard Coded Block streaming clause evaluator")
        lines.append("// Generated by Matador.  One registered AND stage per AXI beat.")
        lines.append("// Clause g fires iff all its included literals are satisfied by the")
        lines.append("// feature sequence.  Empty clauses are initialised to 0.")
        lines.append(f"module hcb_chain #(")
        lines.append(f"    parameter N_FEATURES      = {N},")
        lines.append(f"    parameter N_CLAUSES_TOTAL = {CT},")
        lines.append(f"    parameter AXIS_DATA_WIDTH = {DW},")
        lines.append(f"    parameter N_BEATS         = {NB}")
        lines.append(f")(")
        lines.append(f"    input  wire                       clk,")
        lines.append(f"    input  wire                       aresetn,")
        lines.append(f"    input  wire                       beat_valid,  // high in S_RECV when FIFO output is valid")
        lines.append(f"    input  wire [{bcw-1}:0]             beat_cnt,    // which beat we are processing")
        lines.append(f"    input  wire [AXIS_DATA_WIDTH-1:0]  beat_data,   // fifo_m_tdata")
        lines.append(f"    input  wire                       arming,      // pulse to re-arm for next inference")
        lines.append(f"    output wire [N_CLAUSES_TOTAL-1:0] clause_fires // valid after last beat + 1 cycle")
        lines.append(f");")
        lines.append(f"")
        lines.append(f"    // HCB_INIT: 1 for clauses with at least one included literal, 0 for empty.")
        lines.append(f"    localparam [{CT-1}:0] HCB_INIT = {CT}'h{init_hex};")
        lines.append(f"")
        lines.append(f"    reg [{CT-1}:0] hcb_result;")
        lines.append(f"    assign clause_fires = hcb_result;")
        lines.append(f"")

        # Per-clause combinational beat contribution wires
        lines.append(f"    // Per-clause beat contribution: depends on beat_cnt and beat_data.")
        lines.append(f"    // For beat b: contribution = AND of included literals of clause g")
        lines.append(f"    //             in feature range [b*DW .. (b+1)*DW-1].")
        lines.append(f"    // Beats with no included literals for clause g contribute 1'b1.")

        for g in range(self.CT):
            row = self.includes[g]
            beat_exprs: dict[int, str] = {}
            for b in range(NB):
                lo, hi = b * DW, min((b + 1) * DW, N)
                pos = [f - lo for f in range(lo, hi) if row[f]]
                neg = [f - lo for f in range(lo, hi) if row[N + f]]
                terms = [f"beat_data[{i}]"  for i in pos] + \
                        [f"(~beat_data[{i}])" for i in neg]
                if terms:
                    beat_exprs[b] = " & ".join(terms)

            lines.append(f"    wire hcb_c_{g:05d};")
            if not beat_exprs:
                lines.append(f"    assign hcb_c_{g:05d} = 1'b1;  // empty clause")
            else:
                parts = [f"(beat_cnt == {b}) ? ({expr}) :" for b, expr in sorted(beat_exprs.items())]
                parts.append("1'b1")
                lines.append(f"    assign hcb_c_{g:05d} = {' '.join(parts)};")

        # Build concatenation for vectorised AND (MSB first for Verilog concat)
        contrib_concat = ", ".join(f"hcb_c_{g:05d}" for g in range(CT - 1, -1, -1))

        lines.append(f"")
        lines.append(f"    // HCB accumulation register — one AND stage per beat.")
        lines.append(f"    // Re-armed to HCB_INIT at reset or when arming pulse is seen.")
        lines.append(f"    always @(posedge clk) begin")
        lines.append(f"        if (!aresetn || arming) begin")
        lines.append(f"            hcb_result <= HCB_INIT;")
        lines.append(f"        end else if (beat_valid) begin")
        lines.append(f"            hcb_result <= hcb_result & {{{contrib_concat}}};")
        lines.append(f"        end")
        lines.append(f"    end")
        lines.append(f"")
        lines.append(f"endmodule")
        return "\n".join(lines)

    # ------------------------------------------------------------------ adder tree

    def _adder_tree(self, input_sigs: list[str], prefix: str) -> tuple[list[str], list[str], str]:
        if not input_sigs:
            d = [f"    wire [0:0] {prefix}_z;"]
            a = [f"    assign {prefix}_z = 1'b0;"]
            return d, a, f"{prefix}_z"

        decls:   list[str] = []
        assigns: list[str] = []
        current  = list(input_sigs)
        cur_w    = 1
        level    = 0

        while len(current) > 1:
            n_out  = math.ceil(len(current) / 2)
            next_w = cur_w + 1
            is_reg = level in self._reg_levels
            next_sigs: list[str] = []
            kw = "reg" if is_reg else "wire"

            for i in range(n_out):
                sn = f"{prefix}_l{level}_{i}"
                decls.append(f"    {kw} [{next_w-1}:0] {sn};")
                next_sigs.append(sn)

            if is_reg:
                assigns.append(f"    always @(posedge clk) begin")
                for i in range(n_out):
                    sn = f"{prefix}_l{level}_{i}"
                    a  = current[2 * i]
                    b  = current[2 * i + 1] if 2 * i + 1 < len(current) else f"{cur_w}'d0"
                    assigns.append(f"        {sn} <= {a} + {b};")
                assigns.append(f"    end")
            else:
                for i in range(n_out):
                    sn = f"{prefix}_l{level}_{i}"
                    a  = current[2 * i]
                    b  = current[2 * i + 1] if 2 * i + 1 < len(current) else f"{cur_w}'d0"
                    assigns.append(f"    assign {sn} = {a} + {b};")

            current = next_sigs
            cur_w   = next_w
            level  += 1

        return decls, assigns, current[0]

    # ------------------------------------------------------------------ score module

    def _gen_hw_score(self) -> str:
        """Generate hw_score.v — adder trees, signed net scores, and argmax."""
        SW = self.SCORE_W
        CW = self.CLASS_W

        lines = []
        lines.append("`timescale 1ns/1ps")
        lines.append("// hw_score — adder trees + argmax for hw_tm_accelerator")
        lines.append("// Generated by Matador.")
        lines.append(f"module hw_score #(")
        lines.append(f"    parameter N_CLASSES       = {self.C},")
        lines.append(f"    parameter N_CLAUSES_TOTAL = {self.CT},")
        lines.append(f"    parameter SCORE_WIDTH     = {SW},")
        lines.append(f"    parameter CLASS_BITS      = {CW}")
        lines.append(f")(")
        lines.append(f"    input  wire                       clk,")
        lines.append(f"    input  wire [N_CLAUSES_TOTAL-1:0] clause_fires,")
        lines.append(f"    output wire [CLASS_BITS-1:0]       predicted_class")
        lines.append(f");")
        lines.append(f"")

        all_decls:   list[str] = []
        all_assigns: list[str] = []
        score_sigs:  list[str] = []

        for c in range(self.C):
            base_g = c * self.K
            pos_sigs = [f"clause_fires[{base_g + k}]"               for k in range(self.HALF_K)]
            neg_sigs = [f"clause_fires[{base_g + self.HALF_K + k}]" for k in range(self.HALF_K)]

            pd, pa, pos_final = self._adder_tree(pos_sigs, f"pos_c{c}")
            nd, na, neg_final = self._adder_tree(neg_sigs, f"neg_c{c}")

            all_decls  += pd + nd
            all_assigns += pa + na

            ssig = f"score_{c}"
            sum_w = math.ceil(math.log2(self.HALF_K + 1)) + 1
            all_decls.append(f"    wire signed [{SW-1}:0] {ssig};")
            all_assigns.append(
                f"    assign {ssig} = "
                f"$signed({{{{1'b0, {pos_final}}}}})"
                f" - $signed({{{{1'b0, {neg_final}}}}});"
            )
            score_sigs.append(ssig)

        lines += all_decls
        lines.append(f"")
        lines += all_assigns
        lines.append(f"")

        # Argmax — linear scan
        lines.append(f"    // Linear-scan argmax")
        lines.append(f"    wire signed [{SW-1}:0] mx_score_0 = {score_sigs[0]};")
        lines.append(f"    wire [{CW-1}:0]        mx_idx_0   = {CW}'d0;")
        for k in range(1, self.C):
            sc_k    = score_sigs[k]
            prev_sc = f"mx_score_{k-1}"
            prev_ix = f"mx_idx_{k-1}"
            lines.append(
                f"    wire signed [{SW-1}:0] mx_score_{k} = "
                f"({sc_k} > {prev_sc}) ? {sc_k} : {prev_sc};"
            )
            lines.append(
                f"    wire [{CW-1}:0]        mx_idx_{k}   = "
                f"({sc_k} > {prev_sc}) ? {CW}'d{k} : {prev_ix};"
            )
        lines.append(f"    assign predicted_class = mx_idx_{self.C-1};")
        lines.append(f"")
        lines.append(f"endmodule")
        return "\n".join(lines)

    # ------------------------------------------------------------------ top module

    def _gen_hw_tm_accelerator(self) -> str:
        N, C, K, CT = self.N, self.C, self.K, self.CT
        DW, FD, PS = self.DW, self.config.fifo_depth, self.PS
        SW  = self.SCORE_W
        CW  = self.CLASS_W
        NB  = self.N_BEATS
        APS = self.actual_ps
        bcw = max(1, math.ceil(math.log2(NB + 1)))
        ecw = max(1, math.ceil(math.log2(APS + 2)))
        fp  = self.tmir.fingerprint()

        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // =============================================================================
        // hw_tm_accelerator — Hardwired HCB TM Accelerator
        // Generated by Matador  |  model: {fp}
        //
        // HCB pipeline: {NB} beats × 1 register stage + {APS} adder-tree stages
        // Total inference latency ≈ {NB + APS + 4} cycles
        // =============================================================================
        module hw_tm_accelerator #(
            parameter N_FEATURES      = {N},
            parameter N_CLASSES       = {C},
            parameter N_CLAUSES_PC    = {K},
            parameter N_CLAUSES_TOTAL = {CT},
            parameter AXIS_DATA_WIDTH = {DW},
            parameter FIFO_DEPTH      = {FD},
            parameter PIPELINE_STAGES = {PS},
            parameter N_BEATS         = {NB},
            parameter SCORE_WIDTH     = {SW},
            parameter CLASS_BITS      = {CW}
        )(
            input  wire                       clk,
            input  wire                       aresetn,
            input  wire [AXIS_DATA_WIDTH-1:0] s_axis_tdata,
            input  wire                       s_axis_tvalid,
            output wire                       s_axis_tready,
            input  wire                       s_axis_tlast,
            output reg  [AXIS_DATA_WIDTH-1:0] m_axis_tdata,
            output reg                        m_axis_tvalid,
            input  wire                       m_axis_tready,
            output reg                        m_axis_tlast
        );

            localparam S_IDLE = 2'd0;
            localparam S_RECV = 2'd1;
            localparam S_EVAL = 2'd2;
            localparam S_DONE = 2'd3;

            reg [1:0]     state;
            reg [{bcw-1}:0] beat_cnt;
            reg [{ecw-1}:0] eval_cnt;

            // ── FIFO ─────────────────────────────────────────────────────────
            wire                       fifo_s_tready;
            wire                       fifo_m_tvalid;
            wire [AXIS_DATA_WIDTH-1:0] fifo_m_tdata;
            wire                       fifo_m_tlast;

            assign s_axis_tready = fifo_s_tready;

            axis_fifo #(.DATA_WIDTH(AXIS_DATA_WIDTH), .DEPTH(FIFO_DEPTH)) u_fifo (
                .clk    (clk),
                .rst_n  (aresetn),
                .s_tvalid(s_axis_tvalid), .s_tready(fifo_s_tready),
                .s_tdata (s_axis_tdata),  .s_tlast (s_axis_tlast),
                .m_tvalid(fifo_m_tvalid), .m_tready(state == S_RECV),
                .m_tdata (fifo_m_tdata),  .m_tlast (fifo_m_tlast)
            );

            // ── HCB clause evaluation ────────────────────────────────────────
            wire [{CT-1}:0] clause_fires;

            hcb_chain #(
                .N_FEATURES({N}), .N_CLAUSES_TOTAL({CT}),
                .AXIS_DATA_WIDTH({DW}), .N_BEATS({NB})
            ) u_hcb (
                .clk        (clk),
                .aresetn    (aresetn),
                .beat_valid (fifo_m_tvalid && state == S_RECV),
                .beat_cnt   (beat_cnt),
                .beat_data  (fifo_m_tdata),
                .arming     (state == S_IDLE),
                .clause_fires(clause_fires)
            );

            // ── Score accumulation + argmax ──────────────────────────────────
            wire [CLASS_BITS-1:0] predicted_class;

            hw_score #(
                .N_CLASSES({C}), .N_CLAUSES_TOTAL({CT}),
                .SCORE_WIDTH({SW}), .CLASS_BITS({CW})
            ) u_score (
                .clk           (clk),
                .clause_fires  (clause_fires),
                .predicted_class(predicted_class)
            );

            // ── Beat counter ─────────────────────────────────────────────────
            always @(posedge clk) begin
                if (!aresetn || state == S_IDLE)
                    beat_cnt <= 0;
                else if (state == S_RECV && fifo_m_tvalid)
                    beat_cnt <= beat_cnt + 1;
            end

            // ── FSM ──────────────────────────────────────────────────────────
            always @(posedge clk) begin
                if (!aresetn) begin
                    state         <= S_IDLE;
                    eval_cnt      <= 0;
                    m_axis_tvalid <= 1'b0;
                    m_axis_tlast  <= 1'b0;
                    m_axis_tdata  <= {{{DW}{{1'b0}}}};
                end else begin
                    case (state)
                        S_IDLE: begin
                            m_axis_tvalid <= 1'b0;
                            if (fifo_m_tvalid) state <= S_RECV;
                        end
                        S_RECV: begin
                            if (fifo_m_tvalid && fifo_m_tlast) begin
                                state    <= S_EVAL;
                                eval_cnt <= 0;
                            end
                        end
                        S_EVAL: begin
                            if (eval_cnt == {APS}) begin
                                state         <= S_DONE;
                                m_axis_tdata  <= {{{{{DW-CW}{{1'b0}}}}, predicted_class}};
                                m_axis_tvalid <= 1'b1;
                                m_axis_tlast  <= 1'b1;
                            end else
                                eval_cnt <= eval_cnt + 1;
                        end
                        S_DONE: begin
                            if (m_axis_tvalid && m_axis_tready) begin
                                m_axis_tvalid <= 1'b0;
                                m_axis_tlast  <= 1'b0;
                                state <= S_IDLE;
                            end
                        end
                    endcase
                end
            end

        endmodule
        """)

    # ------------------------------------------------------------------ axis_fifo

    def _gen_axis_fifo(self) -> str:
        DW = self.DW
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        module axis_fifo #(
            parameter DATA_WIDTH = {DW},
            parameter DEPTH      = 16
        )(
            input  wire                  clk,
            input  wire                  rst_n,
            input  wire                  s_tvalid,
            output wire                  s_tready,
            input  wire [DATA_WIDTH-1:0] s_tdata,
            input  wire                  s_tlast,
            output reg                   m_tvalid,
            input  wire                  m_tready,
            output reg  [DATA_WIDTH-1:0] m_tdata,
            output reg                   m_tlast
        );
            localparam AW  = $clog2(DEPTH);
            localparam DW1 = DATA_WIDTH + 1;
            reg [DW1-1:0] mem  [0:DEPTH-1];
            reg [AW:0]    wptr, rptr;
            wire full  = (wptr[AW] != rptr[AW]) & (wptr[AW-1:0] == rptr[AW-1:0]);
            wire empty = (wptr == rptr);
            assign s_tready = ~full;
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    wptr <= {{(AW+1){{1'b0}}}}; rptr <= {{(AW+1){{1'b0}}}};
                    m_tvalid <= 1'b0; m_tdata <= {{DATA_WIDTH{{1'b0}}}}; m_tlast <= 1'b0;
                end else begin
                    if (s_tvalid & ~full) begin mem[wptr[AW-1:0]] <= {{s_tlast,s_tdata}}; wptr <= wptr+1'b1; end
                    if (~m_tvalid | m_tready) begin
                        if (~empty) begin {{m_tlast,m_tdata}} <= mem[rptr[AW-1:0]]; rptr <= rptr+1'b1; m_tvalid <= 1'b1; end
                        else m_tvalid <= 1'b0;
                    end
                end
            end
        endmodule
        """)

    # ------------------------------------------------------------------ testbenches

    def _gen_tb_hcb_chain(self) -> str:
        """Unit testbench for hcb_chain — drives individual beats, checks clause_fires."""
        DW, CT, NB = self.DW, self.CT, self.N_BEATS
        bcw = max(1, math.ceil(math.log2(NB + 1)))

        vecs = []
        if self.tmir.verification and self.tmir.verification.test_vectors:
            vecs = self.tmir.verification.test_vectors[:4]

        # Pre-compute expected clause fires for each test vector
        import numpy as np
        vec_decls  = []
        check_code = []
        for vi, tv in enumerate(vecs):
            feats = tv.input[:self.N]
            X = np.array(feats, dtype=np.uint8)
            L = np.concatenate([X, 1 - X])
            expected = np.zeros(CT, dtype=bool)
            for g in range(CT):
                mask = self.includes[g]
                expected[g] = bool(L[mask].all()) if mask.any() else False
            exp_int = int(sum(int(b) << i for i, b in enumerate(expected)))
            exp_hex = f"{exp_int:0{(CT+3)//4}X}"

            # Beats
            beats = []
            for b in range(NB):
                word = 0
                for bit in range(DW):
                    gi = b * DW + bit
                    if gi < len(feats) and feats[gi]:
                        word |= (1 << bit)
                beats.append(word)

            vec_decls.append(f"    // vec {vi}: expected_class={tv.expected_class}")
            vec_decls.append(f"    reg [{DW-1}:0] beats_{vi} [{NB-1}:0];")
            check_code.append(f"            // vec {vi}")
            for b, w in enumerate(beats):
                check_code.append(f"            beats_{vi}[{b}] = {DW}'h{w:0{DW//4}X};")

        init_block = "\n".join(check_code)
        decl_block = "\n".join(vec_decls)

        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_hcb_chain — unit testbench for hcb_chain
        // Auto-generated by Matador.  Drives {len(vecs)} test vector(s) beat-by-beat.
        module tb_hcb_chain;
            parameter CLK_HALF = 5;
            parameter DW    = {DW};
            parameter CT    = {CT};
            parameter N_BEATS = {NB};

            reg clk, aresetn, beat_valid, arming;
            reg [{bcw-1}:0] beat_cnt;
            reg [DW-1:0]  beat_data;
            wire [CT-1:0] clause_fires;

            hcb_chain #(.N_FEATURES({self.N}), .N_CLAUSES_TOTAL(CT),
                        .AXIS_DATA_WIDTH(DW), .N_BEATS(N_BEATS)) dut (
                .clk(clk), .aresetn(aresetn),
                .beat_valid(beat_valid), .beat_cnt(beat_cnt),
                .beat_data(beat_data), .arming(arming),
                .clause_fires(clause_fires)
            );

        {decl_block}

            integer v, b, pass_cnt, fail_cnt;

            initial clk = 0;
            always #CLK_HALF clk = ~clk;

            initial begin
                $dumpfile("tb_hcb_chain.vcd");
                $dumpvars(0, tb_hcb_chain);

        {init_block}

                aresetn    = 1'b0; beat_valid = 1'b0; arming = 1'b0;
                beat_cnt   = 0;    beat_data  = {{DW{{1'b0}}}};
                @(posedge clk); #1; aresetn = 1'b1;
                pass_cnt = 0; fail_cnt = 0;

        """) + self._tb_hcb_vec_loop(vecs) + textwrap.dedent(f"""\
                $display("HCB chain: %0d passed, %0d failed", pass_cnt, fail_cnt);
                if (fail_cnt == 0) $display("PASS"); else $display("FAIL");
                $finish;
            end
        endmodule
        """)

    def _tb_hcb_vec_loop(self, vecs) -> str:
        if not vecs:
            return "        $display(\"No test vectors embedded.\");\n"
        import numpy as np
        lines = []
        for vi, tv in enumerate(vecs):
            feats = tv.input[:self.N]
            X = np.array(feats, dtype=np.uint8)
            L = np.concatenate([X, 1 - X])
            expected = np.zeros(self.CT, dtype=bool)
            for g in range(self.CT):
                mask = self.includes[g]
                expected[g] = bool(L[mask].all()) if mask.any() else False
            exp_int = int(sum(int(b) << i for i, b in enumerate(expected)))
            exp_hex = f"{exp_int:0{(self.CT+3)//4}X}"

            lines += [
                f"        // ── vec {vi} ──",
                f"        arming = 1'b1; @(posedge clk); #1; arming = 1'b0;",
                f"        @(posedge clk); #1;",
                f"        beat_cnt = 0;",
                f"        for (b = 0; b < N_BEATS; b = b + 1) begin",
                f"            beat_data  = beats_{vi}[b];",
                f"            beat_valid = 1'b1;",
                f"            beat_cnt   = b;",
                f"            @(posedge clk); #1;",
                f"        end",
                f"        beat_valid = 1'b0;",
                f"        @(posedge clk); #1;  // wait for last stage to register",
                f"        if (clause_fires === {self.CT}'h{exp_hex}) begin",
                f"            $display(\"vec {vi}: PASS\"); pass_cnt = pass_cnt + 1;",
                f"        end else begin",
                f"            $display(\"vec {vi}: FAIL expected={self.CT}'h{exp_hex} got=%h\", clause_fires);",
                f"            fail_cnt = fail_cnt + 1;",
                f"        end",
            ]
        return "\n".join(f"        {l.lstrip()}" if l.startswith("        ") else l for l in lines) + "\n"

    def _gen_tb_system(self) -> str:
        DW, NB = self.DW, self.N_BEATS
        APS = self.actual_ps
        total_latency = NB + APS + 6
        vecs = self.tmir.verification.test_vectors[:8] if (self.tmir.verification and self.tmir.verification.test_vectors) else []

        beat_cases = self._tb_beat_cases(vecs)

        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_hw_system — system testbench for hw_tm_accelerator (HCB design)
        module tb_hw_system;
            parameter CLK_HALF = 5;
            parameter DW       = {DW};
            parameter N_BEATS  = {NB};
            parameter N_VEC    = {max(len(vecs), 1)};

            reg clk, aresetn, s_tvalid, s_tlast, m_tready;
            reg [DW-1:0] s_tdata;
            wire s_tready;
            wire [DW-1:0] m_tdata;
            wire m_tvalid, m_tlast;

            hw_tm_accelerator dut (
                .clk(clk), .aresetn(aresetn),
                .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid),
                .s_axis_tready(s_tready), .s_axis_tlast(s_tlast),
                .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid),
                .m_axis_tready(m_tready), .m_axis_tlast(m_tlast)
            );

        """) + self._tb_vec_decls(vecs) + textwrap.dedent(f"""\
            integer v, b, pass_cnt, fail_cnt, timeout_cnt;

            initial clk = 0;
            always #CLK_HALF clk = ~clk;

            initial begin
                $dumpfile("tb_hw_system.vcd");
                $dumpvars(0, tb_hw_system);
        """) + self._tb_vec_inits(vecs) + textwrap.dedent(f"""\
                aresetn = 1'b0; s_tvalid = 1'b0; s_tlast = 1'b0;
                m_tready = 1'b1; s_tdata = {{DW{{1'b0}}}};
                @(posedge clk); #1; @(posedge clk); #1;
                aresetn = 1'b1; @(posedge clk); #1;
                pass_cnt = 0; fail_cnt = 0;

                for (v = 0; v < N_VEC; v = v + 1) begin
                    for (b = 0; b < N_BEATS; b = b + 1) begin
                        s_tvalid = 1'b1;
                        s_tlast  = (b == N_BEATS-1) ? 1'b1 : 1'b0;
                        case (v)
        {beat_cases}
                        endcase
                        @(posedge clk); #1;
                        while (!s_tready) @(posedge clk);
                    end
                    s_tvalid = 1'b0; s_tlast = 1'b0;
                    // Poll for m_tvalid — handshake completes in one cycle so
                    // a fixed repeat() would miss the pulse. Timeout after 2x
                    // expected latency to catch a stuck FSM.
                    timeout_cnt = 0;
                    while (!m_tvalid && timeout_cnt < {(total_latency + 4) * 2}) begin
                        @(posedge clk); #1;
                        timeout_cnt = timeout_cnt + 1;
                    end
                    if (m_tvalid) begin
                        $display("vec %0d: predicted=%0d  PASS", v, m_tdata[{max(1, self.CLASS_W)-1}:0]);
                        pass_cnt = pass_cnt + 1;
                    end else begin
                        $display("vec %0d: FAIL (m_tvalid not asserted within %0d cycles)", v, timeout_cnt);
                        fail_cnt = fail_cnt + 1;
                    end
                end
                $display("SUMMARY: %0d passed, %0d failed", pass_cnt, fail_cnt);
                if (fail_cnt == 0) $display("PASS"); else $display("FAIL");
                $finish;
            end
        endmodule
        """)

    def _tb_vec_decls(self, vecs) -> str:
        if not vecs:
            return ""
        lines = []
        for vi, tv in enumerate(vecs):
            lines.append(f"    reg [{self.DW-1}:0] vec{vi} [{self.N_BEATS-1}:0];")
        return "\n".join(lines) + "\n\n"

    def _tb_vec_inits(self, vecs) -> str:
        if not vecs:
            return ""
        lines = []
        for vi, tv in enumerate(vecs):
            for b in range(self.N_BEATS):
                word = 0
                for bit in range(self.DW):
                    gi = b * self.DW + bit
                    if gi < len(tv.input) and tv.input[gi]:
                        word |= (1 << bit)
                lines.append(f"        vec{vi}[{b}] = {self.DW}'h{word:0{self.DW//4}X};")
        return "\n".join(lines) + "\n"

    def _tb_beat_cases(self, vecs) -> str:
        if not vecs:
            return f"                        default: s_tdata = {{{self.DW}{{1'b0}}}};"
        lines = []
        for vi, tv in enumerate(vecs):
            lines.append(f"                        {vi}: s_tdata = vec{vi}[b];")
        lines.append(f"                        default: s_tdata = {{{self.DW}{{1'b0}}}};")
        return "\n".join(lines)

    def _gen_sim_scripts(self, rtl_dir: Path) -> dict[str, str]:
        src = rtl_dir / "src"
        tb  = rtl_dir / "tb"
        return {
            "run_iverilog.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                set -e
                cd "$(dirname "$0")"
                SRCS="{src}/axis_fifo.v {src}/hcb_chain.v {src}/hw_score.v {src}/hw_tm_accelerator.v"
                iverilog -g2001 -Wall -Wno-timescale -o tb_hcb_chain $SRCS {tb}/tb_hcb_chain.v && vvp tb_hcb_chain
                iverilog -g2001 -Wall -Wno-timescale -o tb_hw_system  $SRCS {tb}/tb_hw_system.v  && vvp tb_hw_system
                """),
            "lint_verilator.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                set -e
                cd "$(dirname "$0")"
                verilator --lint-only --Wall -sv \\
                    {src}/axis_fifo.v {src}/hcb_chain.v {src}/hw_score.v {src}/hw_tm_accelerator.v
                echo "Lint passed."
                """),
        }

    # ------------------------------------------------------------------ run

    def run(self) -> RTLArtifacts:
        out_dir = self.config.output_dir
        rtl_dir = out_dir / "RTL"
        src_dir = rtl_dir / "src"
        tb_dir  = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"
        for d in (src_dir, tb_dir, sim_dir):
            d.mkdir(parents=True, exist_ok=True)

        fifo_path  = src_dir / "axis_fifo.v";        fifo_path.write_text(self._gen_axis_fifo())
        hcb_path   = src_dir / "hcb_chain.v";        hcb_path.write_text(self._gen_hcb_chain())
        score_path = src_dir / "hw_score.v";         score_path.write_text(self._gen_hw_score())
        top_path   = src_dir / "hw_tm_accelerator.v"; top_path.write_text(self._gen_hw_tm_accelerator())

        tb_hcb  = tb_dir / "tb_hcb_chain.v";  tb_hcb.write_text(self._gen_tb_hcb_chain())
        tb_sys  = tb_dir / "tb_hw_system.v";  tb_sys.write_text(self._gen_tb_system())

        scripts = self._gen_sim_scripts(rtl_dir)
        sim_files = []
        for name, content in scripts.items():
            p = sim_dir / name
            p.write_text(content)
            p.chmod(0o755)
            sim_files.append(p)

        return RTLArtifacts(
            rtl_dir    = rtl_dir,
            sources    = [fifo_path, hcb_path, score_path, top_path],
            testbenches= [tb_hcb, tb_sys],
            sim_scripts= sim_files,
        )
