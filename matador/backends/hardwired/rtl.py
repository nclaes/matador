"""Hardwired backend RTL generator — original HCB module style.

Implements the Hard Coded Block (HCB) style from the original Matador
utils/legacty_scripts/TM_RTL_gen.py, adapted for TMIR-driven generation.

Architecture
------------
One separate HCB_i Verilog module per AXI-Stream beat.
HCB_0 sets initial partial clause values from the first beat's features.
HCB_i (i > 0) takes partial_clause_prev from the previous stage and ANDs
in the contribution of beat i's features.  Chained by HCB_top.

  HCB_0: partial_clause[g] = <literal_expr from beat 0>
  HCB_1: partial_clause[g] = partial_clause_prev[g] & <literal_expr from beat 1>
  ...
  HCB_{N-1}: final partial_clause = clause fire result

Each HCB_i has:  always @(posedge clk) begin if (valid) begin ... end end
HCB_top drives valid[i] high for exactly the clock when beat i arrives.

A balanced binary adder tree in hw_score.v accumulates per-class scores.

Generated files
---------------
  src/hcb_blocks.v         — HCB_0 … HCB_{N-1} modules + HCB_top
  src/hw_score.v           — adder trees + argmax
  src/hw_tm_accelerator.v  — top-level FSM + integration
  src/axis_fifo.v          — shared synchronous FIFO
  tb/tb_hcb_blocks.v       — unit test for HCB clause evaluation
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

    # ------------------------------------------------------------------ HCB modules

    def _gen_hcb_blocks(self) -> str:
        """Generate hcb_blocks.v — individual HCB_i modules + HCB_top.

        Follows the original HCB style from utils/legacty_scripts/TM_RTL_gen.py.
        Each HCB_i is a separate module with its own always block — one per beat.
        HCB_top chains them together; valid[i] fires for exactly the clock when
        beat i's data is on the x bus.
        """
        N, DW, CT, NB = self.N, self.DW, self.CT, self.N_BEATS

        lines = [
            "`timescale 1ns/1ps",
            "// =============================================================================",
            f"// hcb_blocks — Hard Coded Block clause evaluators ({NB} blocks)",
            "// Generated by Matador.",
            "//",
            "// HCB_0   : sets partial_clause[g] from beat 0 literals",
            "// HCB_i   : partial_clause[g] = partial_clause_prev[g] & <beat i literals>",
            "// HCB_top : chains HCB_0..HCB_{N-1}; valid[i] fires when beat i arrives",
            "// =============================================================================",
            "",
        ]

        # ── Individual HCB_i modules ──────────────────────────────────────────
        for i in range(NB):
            feat_lo = i * DW
            feat_hi = min(feat_lo + DW, N)

            if i == 0:
                lines.append(f"module HCB_{i} (x, partial_clause, clk, valid);")
                lines.append(f"\toutput\treg  [{CT-1}:0] partial_clause;")
            else:
                lines.append(f"module HCB_{i} (x, partial_clause, partial_clause_prev, clk, valid);")
                lines.append(f"\tinput\twire [{CT-1}:0] partial_clause_prev;")
                lines.append(f"\toutput\treg  [{CT-1}:0] partial_clause;")

            lines.append(f"\tinput\twire clk;")
            lines.append(f"\tinput\twire [{DW-1}:0] x;")
            lines.append(f"\tinput\twire valid;")
            lines.append("")
            lines.append(f"\talways @(posedge clk) begin")
            lines.append(f"\t\tif (valid) begin")

            for g in range(CT):
                row = self.includes[g]
                pos = [f - feat_lo for f in range(feat_lo, feat_hi) if row[f]]
                neg = [f - feat_lo for f in range(feat_lo, feat_hi) if row[N + f]]
                terms = [f"x[{b}]" for b in pos] + [f"(~x[{b}])" for b in neg]

                if not terms:
                    if i == 0:
                        if self.includes[g].any():
                            # Non-empty clause, no literals in beat 0: AND identity
                            lines.append(f"\t\t\tpartial_clause[{g}] <= 1'b1;")
                        else:
                            # Truly empty clause: always 0 (inference convention)
                            lines.append(f"\t\t\tpartial_clause[{g}] <= 1'b0;")
                    else:
                        # No contribution from this beat: pass through
                        lines.append(f"\t\t\tpartial_clause[{g}] <= partial_clause_prev[{g}];")
                else:
                    expr = " & ".join(terms)
                    if i == 0:
                        lines.append(f"\t\t\tpartial_clause[{g}] <= {expr};")
                    else:
                        lines.append(f"\t\t\tpartial_clause[{g}] <= partial_clause_prev[{g}] & {expr};")

            lines.append(f"\t\tend")
            lines.append(f"\tend")
            lines.append(f"endmodule")
            lines.append("")
            lines.append("")

        # ── HCB_top ───────────────────────────────────────────────────────────
        lines += [
            f"module HCB_top #(",
            f"\tparameter PACKETS_NUM = {NB},",
            f"\tparameter CLAUSE_NUM  = {CT},",
            f"\tparameter DATA_WIDTH  = {DW}",
            f")(",
            f"\tinput  wire                    clk,",
            f"\tinput  wire [DATA_WIDTH-1:0]   x,",
            f"\tinput  wire [PACKETS_NUM-1:0]  valid,",
            f"\toutput wire [CLAUSE_NUM-1:0]   partial_clause",
            f");",
            "",
        ]

        for i in range(NB):
            lines.append(f"\twire [{CT-1}:0] partial_clause_reg_{i};")
        lines.append("")
        lines.append(f"\tassign partial_clause = partial_clause_reg_{NB-1};")
        lines.append("")

        for i in range(NB):
            lines.append(f"\tHCB_{i} HCB_inst_{i} (")
            lines.append(f"\t\t.clk   (clk),")
            lines.append(f"\t\t.x     (x),")
            lines.append(f"\t\t.valid (valid[{i}]),")
            if i == 0:
                lines.append(f"\t\t.partial_clause(partial_clause_reg_{i})")
            else:
                lines.append(f"\t\t.partial_clause_prev(partial_clause_reg_{i-1}),")
                lines.append(f"\t\t.partial_clause     (partial_clause_reg_{i})")
            lines.append(f"\t);")
            lines.append("")

        lines.append("endmodule")
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
        lines.append(f"    input  wire [N_CLAUSES_TOTAL-1:0] partial_clause,")
        lines.append(f"    output wire [CLASS_BITS-1:0]       predicted_class")
        lines.append(f");")
        lines.append(f"")

        all_decls:   list[str] = []
        all_assigns: list[str] = []
        score_sigs:  list[str] = []

        for c in range(self.C):
            base_g = c * self.K
            pos_sigs = [f"partial_clause[{base_g + k}]"               for k in range(self.HALF_K)]
            neg_sigs = [f"partial_clause[{base_g + self.HALF_K + k}]" for k in range(self.HALF_K)]

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

            // ── HCB valid signals — one per beat ────────────────────────────
            // valid[i] is high for exactly the clock when beat i is on fifo_m_tdata.
            wire [{NB-1}:0] hcb_valid;
{chr(10).join(f"            assign hcb_valid[{i}] = fifo_m_tvalid & (state == S_RECV) & (beat_cnt == {i});" for i in range(NB))}

            // ── HCB_top clause evaluation ─────────────────────────────────────
            wire [{CT-1}:0] partial_clause;

            HCB_top #(
                .PACKETS_NUM({NB}), .CLAUSE_NUM({CT}), .DATA_WIDTH({DW})
            ) u_hcb (
                .clk           (clk),
                .x             (fifo_m_tdata),
                .valid         (hcb_valid),
                .partial_clause(partial_clause)
            );

            // ── Score accumulation + argmax ──────────────────────────────────
            wire [CLASS_BITS-1:0] predicted_class;

            hw_score #(
                .N_CLASSES({C}), .N_CLAUSES_TOTAL({CT}),
                .SCORE_WIDTH({SW}), .CLASS_BITS({CW})
            ) u_score (
                .clk           (clk),
                .partial_clause(partial_clause),
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

    def _gen_tb_hcb_blocks(self) -> str:
        """Unit testbench for HCB_top — drives valid[i] + x, checks partial_clause."""
        import numpy as np
        DW, CT, NB = self.DW, self.CT, self.N_BEATS
        vecs = (self.tmir.verification.test_vectors[:4]
                if (self.tmir.verification and self.tmir.verification.test_vectors) else [])

        decl_lines, init_lines, test_lines = [], [], []
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

            decl_lines.append(f"    reg [{DW-1}:0] beats_{vi} [{NB-1}:0];  // vec {vi}, class {tv.expected_class}")
            for b in range(NB):
                word = sum((1 << bit) for bit in range(DW)
                           if (b * DW + bit) < len(feats) and feats[b * DW + bit])
                init_lines.append(f"        beats_{vi}[{b}] = {DW}'h{word:0{DW//4}X};")

            test_lines += [
                f"        // vec {vi} — drive beats one per clock, each HCB_i fires on its beat",
                f"        for (b = 0; b < N_BEATS; b = b + 1) begin",
                f"            x_in = beats_{vi}[b];",
                f"            valid_in = ({NB}'b1 << b);",
                f"            @(posedge clk); #1;",
                f"        end",
                f"        valid_in = {NB}'b0;",
                f"        @(posedge clk); #1;",
                f"        if (partial_clause === {CT}'h{exp_hex}) begin",
                f"            $display(\"vec {vi}: PASS\");  pass_cnt = pass_cnt + 1;",
                f"        end else begin",
                f"            $display(\"vec {vi}: FAIL  expected={CT}'h{exp_hex}  got=%h\", partial_clause);",
                f"            fail_cnt = fail_cnt + 1;",
                f"        end",
            ]

        decl_block = "\n".join(decl_lines)
        init_block = "\n".join(init_lines)
        test_block = "\n".join(test_lines) if test_lines else '        $display("No test vectors.");'

        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_hcb_blocks — unit testbench for HCB_top
        // Auto-generated by Matador.  Drives valid[i] + x for each beat.
        module tb_hcb_blocks;
            parameter CLK_HALF = 5;
            parameter DW      = {DW};
            parameter CT      = {CT};
            parameter N_BEATS = {NB};

            reg  clk;
            reg  [DW-1:0]      x_in;
            reg  [N_BEATS-1:0] valid_in;
            wire [CT-1:0]      partial_clause;

            HCB_top #(.PACKETS_NUM(N_BEATS), .CLAUSE_NUM(CT), .DATA_WIDTH(DW)) dut (
                .clk           (clk),
                .x             (x_in),
                .valid         (valid_in),
                .partial_clause(partial_clause)
            );

        {decl_block}

            integer v, b, pass_cnt, fail_cnt;

            initial clk = 0;
            always #CLK_HALF clk = ~clk;

            initial begin
                $dumpfile("tb_hcb_blocks.vcd");
                $dumpvars(0, tb_hcb_blocks);

        {init_block}

                x_in     = {{DW{{1'b0}}}};
                valid_in = {{N_BEATS{{1'b0}}}};
                @(posedge clk); #1;
                pass_cnt = 0; fail_cnt = 0;

        {test_block}

                $display("HCB blocks: %0d passed, %0d failed", pass_cnt, fail_cnt);
                if (fail_cnt == 0) $display("PASS"); else $display("FAIL");
                $finish;
            end
        endmodule
        """)

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
                SRCS="{src}/axis_fifo.v {src}/hcb_blocks.v {src}/hw_score.v {src}/hw_tm_accelerator.v"
                iverilog -g2001 -Wall -Wno-timescale -o tb_hcb_blocks $SRCS {tb}/tb_hcb_blocks.v && vvp tb_hcb_blocks
                iverilog -g2001 -Wall -Wno-timescale -o tb_hw_system   $SRCS {tb}/tb_hw_system.v  && vvp tb_hw_system
                """),
            "lint_verilator.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                set -e
                cd "$(dirname "$0")"
                verilator --lint-only --Wall -sv \\
                    {src}/axis_fifo.v {src}/hcb_blocks.v {src}/hw_score.v {src}/hw_tm_accelerator.v
                echo "Lint passed."
                """),
            "run_xsim.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                set -e
                cd "$(dirname "$0")"
                SRCS="{src}/axis_fifo.v {src}/hcb_blocks.v {src}/hw_score.v {src}/hw_tm_accelerator.v"
                xvlog --2001 $SRCS {tb}/tb_hw_system.v
                xelab -debug all tb_hw_system -snapshot tb_hw_system_snap
                xsim tb_hw_system_snap -runall
                """),
            "waves.sh": textwrap.dedent("""\
                #!/usr/bin/env bash
                # Usage: bash waves.sh [tb_name]
                # Opens GTKWave with the matching .gtkw for the given testbench.
                # Prefers FST waveforms (Verilator), falls back to VCD (iverilog).
                set -euo pipefail
                TB="${1:-tb_hw_system}"
                SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
                GTKW="$SCRIPT_DIR/$TB.gtkw"
                if [ ! -f "$GTKW" ]; then
                    echo "No .gtkw file for $TB at $GTKW"; exit 1
                fi
                WF="$SCRIPT_DIR/verilator/$TB.fst"
                if [ ! -f "$WF" ]; then WF="$SCRIPT_DIR/$TB.vcd"; fi
                if [ ! -f "$WF" ]; then
                    echo "No waveform for $TB. Run simulation first:"
                    echo "  bash run_iverilog.sh   (produces .vcd)"
                    echo "  make -C verilator run   (produces .fst)"; exit 1
                fi
                echo "Opening GTKWave: $WF  ($GTKW)"
                gtkwave "$WF" "$GTKW" &
                """),
        }

    # ------------------------------------------------------------------ verilator harness

    def _gen_verilator_makefile(self) -> str:
        return textwrap.dedent("""\
        # Auto-generated by matador generate — Verilator build for hw_tm_accelerator
        unexport VERILATOR_ROOT
        VERILATOR  ?= verilator
        SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
        SRC_DIR    := $(SCRIPT_DIR)../../src
        OBJ_DIR    := $(SCRIPT_DIR)obj_dir
        BIN        := $(OBJ_DIR)/Vhw_tm_accelerator

        ifeq ($(shell uname -s),Darwin)
          _VER := $(shell env -u VERILATOR_ROOT $(VERILATOR) --version 2>/dev/null | awk '{print $$2}' | cut -d. -f1)
          ifeq ($(filter 5 6 7 8 9,$(_VER)),)
            _HB := /opt/homebrew/bin/verilator
            _HB_VER := $(shell env -u VERILATOR_ROOT $(_HB) --version 2>/dev/null | awk '{print $$2}' | cut -d. -f1)
            ifneq ($(filter 5 6 7 8 9,$(_HB_VER)),)
              VERILATOR := $(_HB)
            endif
          endif
        endif
        VERILATOR := env -u VERILATOR_ROOT $(VERILATOR)

        VSRCS := \\
            $(SRC_DIR)/axis_fifo.v       \\
            $(SRC_DIR)/hcb_blocks.v      \\
            $(SRC_DIR)/hw_score.v        \\
            $(SRC_DIR)/hw_tm_accelerator.v

        VFLAGS := \\
            --cc --exe --build --trace-fst \\
            --language 1800-2005           \\
            --top-module hw_tm_accelerator \\
            --Mdir $(OBJ_DIR)              \\
            --CFLAGS "-std=c++17"          \\
            -Wall --Wno-TIMESCALEMOD --Wno-WIDTH --Wno-BLKSEQ

        .PHONY: all run clean
        all: $(BIN)
        $(BIN): $(VSRCS) $(SCRIPT_DIR)tb_top.cpp
        \trm -rf $(OBJ_DIR)
        \t$(VERILATOR) $(VFLAGS) $(VSRCS) $(SCRIPT_DIR)tb_top.cpp
        run: clean all
        \t@cd $(SCRIPT_DIR) && $(BIN)
        clean:
        \trm -rf $(OBJ_DIR)
        """)

    def _gen_verilator_tb_cpp(self) -> str:
        N, C, NB = self.N, self.C, self.N_BEATS
        DW, CW   = self.DW, self.CLASS_W
        APS      = self.actual_ps
        latency  = NB + APS + 50
        tdata_t  = "uint64_t" if DW == 64 else "uint32_t"

        vecs = (self.tmir.verification.test_vectors
                if (self.tmir.verification and self.tmir.verification.test_vectors) else [])
        n_tests  = len(vecs)
        arr_tv   = max(n_tests * NB, 1)
        arr_exp  = max(n_tests, 1)

        beat_lines, last_lines, exp_lines = [], [], []
        for idx, tv in enumerate(vecs):
            for b in range(NB):
                word = sum((1 << bit) for bit in range(DW)
                           if (b * DW + bit) < len(tv.input) and tv.input[b * DW + bit])
                is_last = (b == NB - 1)
                beat_lines.append(f"    {word}ULL,  // test {idx} beat {b}")
                last_lines.append(f"    {1 if is_last else 0},")
            exp_lines.append(f"    {tv.expected_class},  // test {idx}")

        beat_data = "\n".join(beat_lines) if beat_lines else "    0"
        last_data = "\n".join(last_lines) if last_lines else "    0"
        exp_data  = "\n".join(exp_lines)  if exp_lines  else "    0"

        return textwrap.dedent(f"""\
        // tb_top.cpp — Verilator C++ testbench for hw_tm_accelerator (HCB design)
        // Auto-generated by Matador.  Produces tb_hw_system.fst.
        #include "Vhw_tm_accelerator.h"
        #include "verilated.h"
        #include "verilated_fst_c.h"
        #include <cstdint>
        #include <cstdio>

        static const int N_BEATS    = {NB};
        static const int CLASS_BITS = {CW};
        static const int N_TESTS    = {n_tests};
        static const int MAX_TIMEOUT = {latency + 100};

        static const uint64_t tv_data[{arr_tv}] = {{
        {beat_data}
        }};
        static const int tv_last[{arr_tv}] = {{
        {last_data}
        }};
        static const int tv_exp[{arr_exp}] = {{
        {exp_data}
        }};

        static vluint64_t sim_time = 0;
        static Vhw_tm_accelerator *dut = nullptr;
        static VerilatedFstC      *tfp = nullptr;

        static void tick() {{
            dut->clk = 0; dut->eval(); if (tfp) tfp->dump(sim_time++);
            dut->clk = 1; dut->eval(); if (tfp) tfp->dump(sim_time++);
        }}

        int main(int argc, char **argv) {{
            Verilated::commandArgs(argc, argv);
            Verilated::traceEverOn(true);
            dut = new Vhw_tm_accelerator;
            tfp = new VerilatedFstC;
            dut->trace(tfp, 99);
            tfp->open("tb_hw_system.fst");

            dut->clk = 0; dut->aresetn = 0;
            dut->s_axis_tvalid = 0; dut->s_axis_tdata = 0;
            dut->s_axis_tlast  = 0; dut->m_axis_tready = 1;
            for (int i = 0; i < 8; i++) tick();
            dut->aresetn = 1;
            tick(); tick();

            int fail_cnt = 0, pass_cnt = 0;
            if (N_TESTS == 0) {{
                printf("tb_hw_system: no test vectors embedded\\n");
            }} else {{
                int input_beat = 0, output_count = 0;
                const int total_beats = N_TESTS * N_BEATS;
                const int max_cycles  = total_beats + N_TESTS * MAX_TIMEOUT;
                int cycle = 0;

                while (output_count < N_TESTS && cycle < max_cycles) {{
                    if (input_beat < total_beats) {{
                        dut->s_axis_tvalid = 1;
                        dut->s_axis_tdata  = ({tdata_t})tv_data[input_beat];
                        dut->s_axis_tlast  = tv_last[input_beat];
                    }} else {{
                        dut->s_axis_tvalid = 0; dut->s_axis_tlast = 0;
                    }}
                    dut->eval();
                    bool accepted = dut->s_axis_tvalid && dut->s_axis_tready;
                    tick(); cycle++;
                    if (accepted) input_beat++;
                    if (dut->m_axis_tvalid) {{
                        int got = (int)(dut->m_axis_tdata) & ((1 << CLASS_BITS) - 1);
                        if (got != tv_exp[output_count]) {{
                            printf("FAIL test %d: expected=%d got=%d\\n",
                                   output_count, tv_exp[output_count], got);
                            fail_cnt++;
                        }} else {{
                            printf("PASS test %d: class=%d\\n", output_count, got);
                            pass_cnt++;
                        }}
                        output_count++;
                    }}
                }}
                if (output_count < N_TESTS) {{
                    printf("TIMEOUT: received %d/%d outputs\\n", output_count, N_TESTS);
                    fail_cnt += N_TESTS - output_count;
                }}
            }}
            if (fail_cnt == 0) printf("ALL %d TESTS PASSED\\n", N_TESTS);
            else               printf("FAILED %d/%d\\n", fail_cnt, N_TESTS);
            if (tfp) {{ tfp->close(); delete tfp; }}
            dut->final(); delete dut;
            return (fail_cnt > 0) ? 1 : 0;
        }}
        """)

    # ------------------------------------------------------------------ gtkwave

    def _gen_gtkw_system(self) -> str:
        DW, CT, NB, SW, CW = self.DW, self.CT, self.N_BEATS, self.SCORE_W, self.CLASS_W
        disp_clauses = min(CT, 16)
        lines = [
            '[dumpfile] "tb_hw_system.vcd"',
            '[timestart] 0',
            '[size] 1920 1080',
            '[zoom] -22',
            '[pos] 0 0',
            '*-22 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1',
            '@800200', '-FSM',
            '@28', 'tb_hw_system.dut.state',
            '@28', 'tb_hw_system.dut.beat_cnt',
            '@28', 'tb_hw_system.dut.eval_cnt',
            '@800200', '-END_FSM',
            '@200', '',
            '@800200', '-AXI-Stream Input',
            '@28', 'tb_hw_system.s_tvalid',
            '@28', 'tb_hw_system.s_tready',
            '@28', f'tb_hw_system.s_tdata[{DW-1}:0]',
            '@28', 'tb_hw_system.s_tlast',
            '@800200', '-END_AXI-Stream Input',
            '@200', '',
            '@800200', '-HCB Pipeline',
            '@22', f'tb_hw_system.dut.hcb_valid[{NB-1}:0]',
            '@28', f'tb_hw_system.dut.partial_clause[{min(disp_clauses-1, CT-1)}:0]',
            '@800200', '-END_HCB Pipeline',
            '@200', '',
            '@800200', '-Scores',
        ]
        for c in range(min(self.C, 8)):
            lines += ['@408', f'tb_hw_system.dut.u_score.score_{c}[{SW-1}:0]']
        lines += [
            '@28', f'tb_hw_system.dut.u_score.predicted_class[{CW-1}:0]',
            '@800200', '-END_Scores',
            '@200', '',
            '@800200', '-AXI-Stream Output',
            '@28', 'tb_hw_system.m_tvalid',
            '@28', 'tb_hw_system.m_tready',
            '@28', f'tb_hw_system.m_tdata[{DW-1}:0]',
            '@28', 'tb_hw_system.m_tlast',
            '@800200', '-END_AXI-Stream Output',
        ]
        return "\n".join(lines) + "\n"

    def _gen_gtkw_hcb(self) -> str:
        DW, CT, NB = self.DW, self.CT, self.N_BEATS
        disp = min(CT, 16)
        lines = [
            '[dumpfile] "tb_hcb_blocks.vcd"',
            '[timestart] 0',
            '[size] 1920 1080',
            '[zoom] -22',
            '[pos] 0 0',
            '*-22 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1',
            '@800200', '-HCB Inputs',
            '@22', f'tb_hcb_blocks.valid_in[{NB-1}:0]',
            '@28', f'tb_hcb_blocks.x_in[{DW-1}:0]',
            '@800200', '-END_HCB Inputs',
            '@200', '',
            '@800200', '-Partial Clause',
            '@28', f'tb_hcb_blocks.partial_clause[{disp-1}:0]',
            '@800200', '-END_Partial Clause',
        ]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ readme

    def _gen_readme(self) -> str:
        N, C, K, CT = self.N, self.C, self.K, self.CT
        DW, NB, APS = self.DW, self.N_BEATS, self.actual_ps
        fp = self.tmir.fingerprint()
        prov = self.tmir.provenance
        fw_ver = prov.framework_version if prov else "?"
        epochs = prov.epochs if prov else "?"

        return textwrap.dedent(f"""\
        # hw_tm_accelerator — Hardwired TM Accelerator RTL

        Auto-generated by Matador.

        ## Model parameters

        | Parameter | Value |
        |---|---|
        | N_FEATURES | {N} |
        | N_CLASSES | {C} |
        | N_CLAUSES_PC | {K} |
        | N_CLAUSES_TOTAL | {CT} |
        | AXIS_DATA_WIDTH | {DW} |
        | N_BEATS | {NB} |
        | PIPELINE_STAGES | {self.PS} |
        | Model fingerprint | `{fp}` |
        | Framework | tmu {fw_ver} |
        | Training epochs | {epochs} |

        ## Quick start

        ```bash
        # iverilog (fast, produces .vcd)
        bash sim/run_iverilog.sh

        # Verilator (faster, produces .fst — requires Verilator ≥5.0)
        make -C sim/verilator run

        # GTKWave
        bash sim/waves.sh tb_hw_system

        # Lint only
        bash sim/lint_verilator.sh
        ```

        ## Architecture — HCB streaming pipeline

        Features arrive over AXI-Stream one beat ({DW} bits) at a time.
        Each beat drives one stage of the Hard Coded Block (HCB) chain:

          HCB_0  : sets partial_clause[g] from beat 0 literals
          HCB_i  : partial_clause[g] = partial_clause_prev[g] & <beat i literals>
          HCB_top: chains HCB_0..HCB_{{N-1}}; valid[i] fires when beat i arrives

        After {NB} beats the final `partial_clause` vector holds the clause fire
        results.  hw_score.v reduces these to per-class scores via a balanced
        binary adder tree ({APS} pipeline stage(s)), then argmax selects the
        predicted class.

        **FSM states:** S_IDLE → S_RECV ({NB} beats) → S_EVAL ({APS} cycles) → S_DONE

        **Total latency:** {NB} + {APS} + ~4 cycles

        ## Directory structure

        ```
        RTL/
          src/
            axis_fifo.v          AXI-Stream synchronous FIFO
            hcb_blocks.v         HCB_0..HCB_{{N-1}} + HCB_top
            hw_score.v           Adder trees + argmax
            hw_tm_accelerator.v  Top-level FSM + integration
          tb/
            tb_hcb_blocks.v      Unit test: HCB clause evaluation
            tb_hw_system.v       System test: full end-to-end inference
          sim/
            run_iverilog.sh      Compile + simulate (iverilog)
            lint_verilator.sh    Verilator lint pass
            run_xsim.sh          Vivado xsim simulation
            waves.sh             Open GTKWave
            tb_hw_system.gtkw    GTKWave save — system view
            tb_hcb_blocks.gtkw   GTKWave save — HCB clause view
            verilator/
              Makefile           Verilator build (FST waveform)
              tb_top.cpp         C++ testbench driver
        ```

        ## Module reference

        ### `HCB_top` (hcb_blocks.v)
        | Port | Dir | Width | Description |
        |---|---|---|---|
        | clk | in | 1 | Clock |
        | x | in | AXIS_DATA_WIDTH | Current AXI beat data |
        | valid | in | N_BEATS | Per-beat enable; valid[i] fires when beat i is on x |
        | partial_clause | out | N_CLAUSES_TOTAL | Clause fire results (valid after last beat + 1 cycle) |

        ### `hw_score` (hw_score.v)
        | Port | Dir | Width | Description |
        |---|---|---|---|
        | clk | in | 1 | Clock |
        | partial_clause | in | N_CLAUSES_TOTAL | Clause fire results |
        | predicted_class | out | CLASS_BITS | Argmax class index |

        ### `hw_tm_accelerator` (hw_tm_accelerator.v)
        AXI-Stream slave (feature input) + AXI-Stream master (class output).
        Same interface as the tiled accelerator — drop-in compatible.

        ## Waveform guide

        Open `tb_hw_system.gtkw` with `bash sim/waves.sh tb_hw_system`.

        Key signals to watch:
        1. `state` — FSM progress: 0=IDLE, 1=RECV, 2=EVAL, 3=DONE
        2. `hcb_valid` — which HCB is firing each cycle
        3. `partial_clause` — accumulates towards the correct value over {NB} beats
        4. `score_N` — per-class scores after the adder tree
        5. `predicted_class` — final argmax output
        6. `m_axis_tvalid` — output handshake; should fire once per inference
        """)

    # ------------------------------------------------------------------ run

    def run(self) -> RTLArtifacts:
        out_dir = self.config.output_dir
        rtl_dir = out_dir / "RTL"
        src_dir = rtl_dir / "src"
        tb_dir  = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"
        ver_dir = sim_dir / "verilator"
        for d in (src_dir, tb_dir, sim_dir, ver_dir):
            d.mkdir(parents=True, exist_ok=True)

        fifo_path  = src_dir / "axis_fifo.v";          fifo_path.write_text(self._gen_axis_fifo())
        hcb_path   = src_dir / "hcb_blocks.v";         hcb_path.write_text(self._gen_hcb_blocks())
        score_path = src_dir / "hw_score.v";            score_path.write_text(self._gen_hw_score())
        top_path   = src_dir / "hw_tm_accelerator.v";  top_path.write_text(self._gen_hw_tm_accelerator())

        tb_hcb = tb_dir / "tb_hcb_blocks.v";  tb_hcb.write_text(self._gen_tb_hcb_blocks())
        tb_sys = tb_dir / "tb_hw_system.v";   tb_sys.write_text(self._gen_tb_system())

        # Verilator harness
        mk_path  = ver_dir / "Makefile";    mk_path.write_text(self._gen_verilator_makefile())
        cpp_path = ver_dir / "tb_top.cpp";  cpp_path.write_text(self._gen_verilator_tb_cpp())

        # GTKWave save files
        gtkw_sys = sim_dir / "tb_hw_system.gtkw";  gtkw_sys.write_text(self._gen_gtkw_system())
        gtkw_hcb = sim_dir / "tb_hcb_blocks.gtkw"; gtkw_hcb.write_text(self._gen_gtkw_hcb())

        # README
        readme = rtl_dir / "README.md";  readme.write_text(self._gen_readme())

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
            sim_scripts= sim_files + [mk_path, cpp_path, gtkw_sys, gtkw_hcb, readme],
        )
