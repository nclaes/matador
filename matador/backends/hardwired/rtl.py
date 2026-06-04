"""Hardwired backend RTL generator.

TA Include actions are unrolled directly into AND-gate logic — one
``assign`` statement per clause.  A balanced binary adder tree with
optional pipeline stages accumulates per-class scores.

Architecture
------------
  AXI-Stream beats → axis_fifo → feature buffer
  → clause_fire[g] = AND(included literals)   [combinational, hardwired]
  → pos_sum[c], neg_sum[c]  via balanced binary adder tree
  → score[c] = pos_sum[c] - neg_sum[c]
  → predicted_class = argmax(score)
  → AXI-Stream output

FSM
---
  S_IDLE → S_RECV (buffer feature beats from FIFO)
         → S_EVAL (wait pipeline_stages cycles for adder tree)
         → S_DONE (output predicted class; return to S_IDLE)

Trade-off vs tiled
------------------
  + No tile ROM, no sequential clause-evaluation loop.
  + Lower latency: pipeline_stages + ~4 cycles vs N_FEAT_SLICES*N_CLAUSE_SLICES + N_CLAUSES.
  - LUT cost grows linearly with clause count; poor scaling for large models.
  - Weight support not yet implemented (vanilla only).
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
    """Hardwired combinational TM accelerator."""

    @property
    def name(self) -> str:
        return "hardwired"

    @property
    def config_class(self) -> type:
        from matador.backends.hardwired.config import HardwiredAcceleratorConfig
        return HardwiredAcceleratorConfig

    def generate(self, tmir, config) -> RTLArtifacts:
        if tmir.architecture.clause_organization != "per_class":
            raise ValueError(
                "The hardwired backend currently supports only per_class "
                "clause organisation (vanilla TM).  Coalesced support is planned."
            )
        if tmir.variant not in {"vanilla", "weighted"}:
            raise ValueError(
                f"The hardwired backend does not support variant '{tmir.variant}'.  "
                "Only 'vanilla' is supported."
            )

        gen = _HardwiredGenerator(tmir, config)
        return gen.run()

    def resource_estimate(self, tmir, config) -> ResourceEstimate:
        arch = tmir.architecture
        tmir.to_includes()
        n_inc = int(tmir.representation.includes.sum())
        return ResourceEstimate(
            notes=(
                f"Each included literal becomes one LUT input; "
                f"{n_inc} total include bits across {arch.n_clauses_total} clauses. "
                f"Adder tree: {arch.n_classes} classes × "
                f"{math.ceil(math.log2(max(arch.n_clauses_per_class // 2, 2)))} levels. "
                "Run Vivado synthesis for exact LUT/FF numbers."
            )
        )


# ---------------------------------------------------------------------------
# Internal generator
# ---------------------------------------------------------------------------

class _HardwiredGenerator:
    def __init__(self, tmir, config):
        from matador.backends.hardwired.config import HardwiredAcceleratorConfig

        tmir.to_includes()
        arch = tmir.architecture

        self.tmir   = tmir
        self.config = config

        self.N  = arch.n_features
        self.C  = arch.n_classes
        self.K  = arch.n_clauses_per_class  # clauses per class (must be even)
        self.CT = arch.n_clauses_total
        self.DW = config.axis_data_width
        self.PS = config.pipeline_stages    # requested pipeline stages

        self.HALF_K  = self.K // 2
        self.N_BEATS = math.ceil(self.N / self.DW)

        # Compute score width: signed, range [-(HALF_K), +HALF_K]
        self.SCORE_W = max(2, math.ceil(math.log2(self.HALF_K + 1)) + 2)

        # Class index bit width
        self.CLASS_W = max(1, math.ceil(math.log2(self.C)))

        # Adder tree depth for HALF_K inputs
        self.TREE_DEPTH = max(1, math.ceil(math.log2(max(self.HALF_K, 2))))

        # Actual pipeline stages inserted (capped at tree depth)
        self.actual_ps = min(self.PS, self.TREE_DEPTH)

        # Levels (0-indexed) where we insert a register stage
        self._reg_levels = self._compute_reg_levels()

        # Materialise includes: shape (CT, 2*N), dtype bool
        self.includes = tmir.representation.includes.reshape(self.CT, 2 * self.N).astype(bool)

    # ------------------------------------------------------------------ helpers

    def _compute_reg_levels(self) -> set:
        p = self.actual_ps
        d = self.TREE_DEPTH
        if p <= 0:
            return set()
        if p >= d:
            return set(range(d))
        step = d / p
        return {int((i + 0.5) * step) for i in range(p)}

    def _clause_sig(self, g: int) -> str:
        return f"cf_{g:05d}"

    # ------------------------------------------------------------------ adder tree

    def _adder_tree(self, input_sigs: list[str], prefix: str) -> tuple[list[str], list[str], str]:
        """Build a balanced binary adder tree.

        Returns:
            decls:      list of 'reg/wire [w:0] name;' lines
            assigns:    list of 'assign ...' and 'always @(posedge clk)' lines
            final_sig:  name of the output signal
        """
        if not input_sigs:
            d = [f"    wire [0:0] {prefix}_z;"]
            a = [f"    assign {prefix}_z = 1'b0;"]
            return d, a, f"{prefix}_z"

        decls:   list[str] = []
        assigns: list[str] = []

        current  = list(input_sigs)
        cur_w    = 1          # input signals are 1-bit
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

    # ------------------------------------------------------------------ sections

    def _section_clause_fires(self) -> list[str]:
        lines = ["    // ── Clause fires (hardwired combinational) " + "─" * 30]
        for g in range(self.CT):
            row      = self.includes[g]
            pos_bits = [i for i in range(self.N)     if row[i]]
            neg_bits = [i for i in range(self.N)     if row[self.N + i]]
            sig      = self._clause_sig(g)

            lines.append(f"    wire {sig};")
            terms = [f"feat_buf[{i}]"   for i in pos_bits] + \
                    [f"(~feat_buf[{i}])" for i in neg_bits]

            if not terms:
                lines.append(f"    assign {sig} = 1'b0;  // empty clause")
            else:
                expr = " & ".join(terms)
                lines.append(f"    assign {sig} = {expr};")

        return lines

    def _section_adder_trees(self) -> tuple[list[str], list[str], list[str]]:
        """Returns (decls, assigns, score_signal_names_per_class)."""
        all_decls:   list[str] = ["    // ── Adder tree declarations " + "─" * 35]
        all_assigns: list[str] = ["    // ── Adder tree logic " + "─" * 42]
        score_sigs:  list[str] = []

        for c in range(self.C):
            base_g = c * self.K
            pos_sigs = [self._clause_sig(base_g + k)               for k in range(self.HALF_K)]
            neg_sigs = [self._clause_sig(base_g + self.HALF_K + k) for k in range(self.HALF_K)]

            pd, pa, pos_final = self._adder_tree(pos_sigs, f"pos_c{c}")
            nd, na, neg_final = self._adder_tree(neg_sigs, f"neg_c{c}")

            all_decls  += pd + nd
            all_assigns += pa + na

            # Signed net score: promote to SCORE_W, subtract
            ssig = f"score_{c}"
            sum_w = math.ceil(math.log2(self.HALF_K + 1)) + 1
            all_decls.append(
                f"    wire signed [{self.SCORE_W-1}:0] {ssig};"
            )
            all_assigns.append(
                f"    assign {ssig} = "
                f"$signed({{{{1'b0, {pos_final}}}}})"
                f" - $signed({{{{1'b0, {neg_final}}}}});"
            )
            score_sigs.append(ssig)

        return all_decls, all_assigns, score_sigs

    def _section_argmax(self, score_sigs: list[str]) -> list[str]:
        """Linear-scan argmax over score signals."""
        lines = ["    // ── Argmax (combinational linear scan) " + "─" * 28]
        SW = self.SCORE_W
        CW = self.CLASS_W

        lines.append(f"    wire signed [{SW-1}:0] mx_score_0 = {score_sigs[0]};")
        lines.append(f"    wire [{CW-1}:0]        mx_idx_0   = {CW}'d0;")

        for k in range(1, self.C):
            prev_sc = f"mx_score_{k-1}"
            prev_ix = f"mx_idx_{k-1}"
            sc_k    = score_sigs[k]
            lines.append(
                f"    wire signed [{SW-1}:0] mx_score_{k} = "
                f"({sc_k} > {prev_sc}) ? {sc_k} : {prev_sc};"
            )
            lines.append(
                f"    wire [{CW-1}:0]        mx_idx_{k}   = "
                f"({sc_k} > {prev_sc}) ? {CW}'d{k} : {prev_ix};"
            )

        lines.append(f"    wire [{CW-1}:0] predicted_class = mx_idx_{self.C-1};")
        return lines

    def _section_feature_capture(self) -> list[str]:
        """Explicit case-statement beat capture (avoids variable part-select)."""
        DW = self.DW
        lines = ["    // ── Feature capture (one AXI beat per cycle) " + "─" * 25]
        lines.append("    always @(posedge clk) begin")
        lines.append("        if (!aresetn) begin")
        lines.append("            feat_buf <= {N_FEATURES{1'b0}};")
        lines.append("            beat_cnt <= 0;")
        lines.append("        end else if (fifo_m_tvalid && state == S_RECV) begin")
        lines.append("            case (beat_cnt)")

        for b in range(self.N_BEATS):
            lo = b * DW
            hi = min(lo + DW, self.N) - 1
            used = hi - lo + 1
            if used == DW:
                lines.append(f"                {b}: feat_buf[{hi}:{lo}] <= fifo_m_tdata;")
            else:
                lines.append(
                    f"                {b}: feat_buf[{hi}:{lo}] <= fifo_m_tdata[{used-1}:0];"
                )

        lines.append("            endcase")
        lines.append("            beat_cnt <= beat_cnt + 1;")
        lines.append("        end else if (state == S_IDLE) begin")
        lines.append("            beat_cnt <= 0;")
        lines.append("        end")
        lines.append("    end")
        return lines

    def _section_fsm(self) -> list[str]:
        ACTUAL_PS = self.actual_ps
        CW = self.CLASS_W
        DW = self.DW

        lines = ["    // ── FSM ─" + "─" * 55]
        lines.append("    always @(posedge clk) begin")
        lines.append("        if (!aresetn) begin")
        lines.append("            state         <= S_IDLE;")
        lines.append("            eval_cnt      <= 0;")
        lines.append("            m_axis_tvalid <= 1'b0;")
        lines.append("            m_axis_tlast  <= 1'b0;")
        lines.append(f"            m_axis_tdata  <= {{{DW}{{1'b0}}}};")
        lines.append("        end else begin")
        lines.append("            case (state)")
        lines.append("                S_IDLE: begin")
        lines.append("                    m_axis_tvalid <= 1'b0;")
        lines.append("                    if (fifo_m_tvalid)")
        lines.append("                        state <= S_RECV;")
        lines.append("                end")
        lines.append("                S_RECV: begin")
        lines.append("                    if (fifo_m_tvalid && fifo_m_tlast) begin")
        lines.append("                        state    <= S_EVAL;")
        lines.append("                        eval_cnt <= 0;")
        lines.append("                    end")
        lines.append("                end")
        lines.append("                S_EVAL: begin")
        lines.append(f"                    if (eval_cnt == {ACTUAL_PS}) begin")
        lines.append("                        state         <= S_DONE;")
        lines.append(
            f"                        m_axis_tdata  <= {{{{{DW - CW}{{1'b0}}}}, predicted_class}};"
        )
        lines.append("                        m_axis_tvalid <= 1'b1;")
        lines.append("                        m_axis_tlast  <= 1'b1;")
        lines.append("                    end else begin")
        lines.append("                        eval_cnt <= eval_cnt + 1;")
        lines.append("                    end")
        lines.append("                end")
        lines.append("                S_DONE: begin")
        lines.append("                    if (m_axis_tvalid && m_axis_tready) begin")
        lines.append("                        m_axis_tvalid <= 1'b0;")
        lines.append("                        m_axis_tlast  <= 1'b0;")
        lines.append("                        state <= S_IDLE;")
        lines.append("                    end")
        lines.append("                end")
        lines.append("            endcase")
        lines.append("        end")
        lines.append("    end")
        return lines

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

        fp = self.tmir.fingerprint()

        header = textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // =============================================================================
        // hw_tm_accelerator — Hardwired Tsetlin Machine Accelerator
        // Generated by Matador  |  model: {fp}
        //
        // Architecture: combinational AND-gate clause evaluation + balanced
        // binary adder tree ({APS} pipeline stage(s)).
        //
        // Each clause is a hardwired AND of its included literals — no tile ROM,
        // no sequential clause-evaluation loop.
        //
        // Latency: N_BEATS + {APS} + 3 cycles (RECV + EVAL pipeline + DONE handshake)
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

            reg [1:0]      state;
            reg [{bcw-1}:0]  beat_cnt;
            reg [{ecw-1}:0]  eval_cnt;
            reg [N_FEATURES-1:0] feat_buf;

            // ── FIFO (decouples producer from FSM) ───────────────────────
            wire                       fifo_s_tready;
            wire                       fifo_m_tvalid;
            wire [AXIS_DATA_WIDTH-1:0] fifo_m_tdata;
            wire                       fifo_m_tlast;

            assign s_axis_tready = fifo_s_tready;

            axis_fifo #(
                .DATA_WIDTH(AXIS_DATA_WIDTH),
                .DEPTH(FIFO_DEPTH)
            ) u_fifo (
                .clk    (clk),
                .rst_n  (aresetn),
                .s_tvalid(s_axis_tvalid),
                .s_tready(fifo_s_tready),
                .s_tdata (s_axis_tdata),
                .s_tlast (s_axis_tlast),
                .m_tvalid(fifo_m_tvalid),
                .m_tready(state == S_RECV),
                .m_tdata (fifo_m_tdata),
                .m_tlast (fifo_m_tlast)
            );

        """)

        clause_lines  = self._section_clause_fires()
        tree_decls, tree_assigns, score_sigs = self._section_adder_trees()
        argmax_lines  = self._section_argmax(score_sigs)
        capture_lines = self._section_feature_capture()
        fsm_lines     = self._section_fsm()

        body = "\n".join(
            ["    // ── Clause fires " + "─" * 46] +
            clause_lines +
            [""] +
            tree_decls +
            [""] +
            tree_assigns +
            [""] +
            argmax_lines +
            [""] +
            capture_lines +
            [""] +
            fsm_lines
        )

        footer = "\nendmodule\n"
        return header + body + footer

    # ------------------------------------------------------------------ axis_fifo

    def _gen_axis_fifo(self) -> str:
        """Shared axis_fifo — identical to tiled backend."""
        DW = self.DW
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // axis_fifo — synchronous AXI-Stream FIFO (shared with tiled backend)
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
            reg [AW:0]    wptr;
            reg [AW:0]    rptr;

            wire full  = (wptr[AW] != rptr[AW]) & (wptr[AW-1:0] == rptr[AW-1:0]);
            wire empty = (wptr == rptr);

            assign s_tready = ~full;

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    wptr     <= {{(AW+1){{1'b0}}}};
                    rptr     <= {{(AW+1){{1'b0}}}};
                    m_tvalid <= 1'b0;
                    m_tdata  <= {{DATA_WIDTH{{1'b0}}}};
                    m_tlast  <= 1'b0;
                end else begin
                    if (s_tvalid & ~full) begin
                        mem[wptr[AW-1:0]] <= {{s_tlast, s_tdata}};
                        wptr <= wptr + 1'b1;
                    end
                    if (~m_tvalid | m_tready) begin
                        if (~empty) begin
                            {{m_tlast, m_tdata}} <= mem[rptr[AW-1:0]];
                            rptr     <= rptr + 1'b1;
                            m_tvalid <= 1'b1;
                        end else begin
                            m_tvalid <= 1'b0;
                        end
                    end
                end
            end
        endmodule
        """)

    # ------------------------------------------------------------------ testbench

    def _gen_tb_system(self) -> str:
        DW, NB, C, PS = self.DW, self.N_BEATS, self.C, self.PS
        APS = self.actual_ps
        total_latency = NB + APS + 6

        tmir = self.tmir
        n_vecs = 0
        vec_lines = []
        if tmir.verification and tmir.verification.test_vectors:
            vecs = tmir.verification.test_vectors[:8]
            n_vecs = len(vecs)
            for vi, tv in enumerate(vecs):
                beats = []
                feats = tv.input
                for b in range(NB):
                    word = 0
                    for bit in range(DW):
                        gi = b * DW + bit
                        if gi < len(feats) and feats[gi]:
                            word |= (1 << bit)
                    beats.append(word)
                beat_strs = ", ".join(f"{DW}'h{w:0{DW//4}X}" for w in beats)
                vec_lines.append(f"    // vec {vi}: expected class {tv.expected_class}")
                vec_lines.append(f"    reg [{DW-1}:0] vec{vi} [{NB-1}:0];")

        init_lines = []
        for vi, tv in enumerate(tmir.verification.test_vectors[:8] if tmir.verification else []):
            beats = []
            feats = tv.input
            for b in range(NB):
                word = 0
                for bit in range(DW):
                    gi = b * DW + bit
                    if gi < len(feats) and feats[gi]:
                        word |= (1 << bit)
                beats.append(word)
            for b, w in enumerate(beats):
                init_lines.append(f"        vec{vi}[{b}] = {DW}'h{w:0{DW//4}X};")

        vecs_block    = "\n".join(vec_lines)
        init_block    = "\n".join(init_lines)
        n_vecs_param  = max(n_vecs, 1)

        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_hw_system — system testbench for hw_tm_accelerator
        // Auto-generated by Matador.  Drives {n_vecs} embedded test vector(s).
        module tb_hw_system;

            parameter CLK_HALF = 5;  // 100 MHz
            parameter DW       = {DW};
            parameter N_BEATS  = {NB};
            parameter N_VEC    = {n_vecs_param};

            reg clk, aresetn;
            reg [DW-1:0] s_tdata;
            reg s_tvalid, s_tlast;
            wire s_tready;
            wire [DW-1:0] m_tdata;
            wire m_tvalid, m_tlast;
            reg m_tready;

            // Instantiate DUT
            hw_tm_accelerator dut (
                .clk          (clk),
                .aresetn      (aresetn),
                .s_axis_tdata (s_tdata),
                .s_axis_tvalid(s_tvalid),
                .s_axis_tready(s_tready),
                .s_axis_tlast (s_tlast),
                .m_axis_tdata (m_tdata),
                .m_axis_tvalid(m_tvalid),
                .m_axis_tready(m_tready),
                .m_axis_tlast (m_tlast)
            );

            // Test vectors
        {vecs_block}

            integer v, b, pass_cnt, fail_cnt;
            reg [DW-1:0] received_class;

            initial clk = 0;
            always #CLK_HALF clk = ~clk;

            initial begin
                $dumpfile("tb_hw_system.vcd");
                $dumpvars(0, tb_hw_system);

        {init_block}

                aresetn  = 1'b0;
                s_tvalid = 1'b0;
                s_tlast  = 1'b0;
                m_tready = 1'b1;
                s_tdata  = {{DW{{1'b0}}}};
                @(posedge clk); #1;
                @(posedge clk); #1;
                aresetn  = 1'b1;
                @(posedge clk); #1;

                pass_cnt = 0;
                fail_cnt = 0;

                for (v = 0; v < N_VEC; v = v + 1) begin
                    // Send feature beats
                    for (b = 0; b < N_BEATS; b = b + 1) begin
                        s_tvalid = 1'b1;
                        s_tlast  = (b == N_BEATS - 1) ? 1'b1 : 1'b0;
                        case (v)
        """) + self._tb_beat_cases() + textwrap.dedent(f"""\
                        endcase
                        @(posedge clk); #1;
                        while (!s_tready) @(posedge clk);
                    end
                    s_tvalid = 1'b0;
                    s_tlast  = 1'b0;

                    // Wait for output
                    @(posedge clk); #1;
                    repeat ({total_latency + 4}) @(posedge clk);
                    if (m_tvalid) begin
                        received_class = m_tdata[{max(1,self.CLASS_W)-1}:0];
                        $display("vec %0d: predicted=%0d  tvalid=1", v, received_class);
                        pass_cnt = pass_cnt + 1;
                    end else begin
                        $display("FAIL vec %0d: m_tvalid not asserted within timeout", v);
                        fail_cnt = fail_cnt + 1;
                    end
                end

                $display("SUMMARY: %0d passed, %0d failed", pass_cnt, fail_cnt);
                if (fail_cnt == 0)
                    $display("PASS");
                else
                    $display("FAIL");
                $finish;
            end
        endmodule
        """)

    def _tb_beat_cases(self) -> str:
        if not (self.tmir.verification and self.tmir.verification.test_vectors):
            return "                        default: s_tdata = {DW{1'b0}};\n"
        lines = []
        vecs = self.tmir.verification.test_vectors[:8]
        DW, NB = self.DW, self.N_BEATS
        for vi, tv in enumerate(vecs):
            lines.append(f"                        {vi}: case (b)")
            for b in range(NB):
                word = 0
                for bit in range(DW):
                    gi = b * DW + bit
                    if gi < len(tv.input) and tv.input[gi]:
                        word |= (1 << bit)
                lines.append(
                    f"                            {b}: s_tdata = {DW}'h{word:0{DW//4}X};"
                )
            lines.append(f"                            default: s_tdata = {{{DW}{{1'b0}}}};")
            lines.append(f"                        endcase")
        lines.append(f"                        default: s_tdata = {{{DW}{{1'b0}}}};")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ sim scripts

    def _gen_iverilog_script(self, rtl_dir: Path) -> str:
        src = rtl_dir / "src"
        tb  = rtl_dir / "tb"
        return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # run_iverilog.sh — compile and simulate hw_tm_accelerator testbench
        set -e
        cd "$(dirname "$0")"
        iverilog -g2001 -Wall -Wno-timescale \\
            -o tb_hw_system \\
            {src}/axis_fifo.v \\
            {src}/hw_tm_accelerator.v \\
            {tb}/tb_hw_system.v
        vvp tb_hw_system
        """)

    def _gen_lint_script(self, rtl_dir: Path) -> str:
        src = rtl_dir / "src"
        return textwrap.dedent(f"""\
        #!/usr/bin/env bash
        # lint_verilator.sh — lint hw_tm_accelerator with Verilator
        set -e
        cd "$(dirname "$0")"
        verilator --lint-only --Wall -sv \\
            {src}/axis_fifo.v \\
            {src}/hw_tm_accelerator.v
        echo "Lint passed."
        """)

    # ------------------------------------------------------------------ run

    def run(self) -> RTLArtifacts:
        out_dir = self.config.output_dir
        rtl_dir = out_dir / "RTL"
        src_dir = rtl_dir / "src"
        tb_dir  = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"

        for d in (src_dir, tb_dir, sim_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Sources
        fifo_path = src_dir / "axis_fifo.v"
        top_path  = src_dir / "hw_tm_accelerator.v"
        fifo_path.write_text(self._gen_axis_fifo())
        top_path.write_text(self._gen_hw_tm_accelerator())

        # Testbench
        tb_path = tb_dir / "tb_hw_system.v"
        tb_path.write_text(self._gen_tb_system())

        # Sim scripts
        iv_script   = sim_dir / "run_iverilog.sh"
        lint_script = sim_dir / "lint_verilator.sh"
        iv_script.write_text(self._gen_iverilog_script(rtl_dir))
        lint_script.write_text(self._gen_lint_script(rtl_dir))
        iv_script.chmod(0o755)
        lint_script.chmod(0o755)

        return RTLArtifacts(
            rtl_dir    = rtl_dir,
            sources    = [fifo_path, top_path],
            testbenches= [tb_path],
            sim_scripts= [iv_script, lint_script],
        )
