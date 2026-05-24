"""Vanilla TM accelerator: software model + Verilog/testbench code generation.

Architecture (tiled feature-clause matrix):
  AXI-Stream feature beats → FIFO → FSM
  → tile ROM (FEAT_SLICE × CLAUSE_SLICE tiles) → CLAUSE_SLICE parallel clause_eval
  → clause_pass / clause_has_actions AND/OR accumulators (one bit per clause)
  → score_acc (N_CLAUSES_TOTAL cycles, S_SCORE state) → argmax → m_axis_tvalid/tdata/tlast

TA Semantics:
  The tile ROM stores COMPILED TA ACTION OUTPUTS from trained Tsetlin Automata.
  Each bit in the ROM is a resolved Include/Exclude action (not a raw TA state integer).
  ta_action_mask[l] = 1  →  TA for literal l issued an Include action  (state > n_states/2)
  ta_action_mask[l] = 0  →  TA for literal l issued an Exclude action  (state ≤ n_states/2)
  clause_eval fires when ALL ta_action_mask=1 literals are also 1 in the feature vector.

Iteration order in S_COMPUTE:
  outer: feat_slice 0 … N_FEAT_SLICES-1
  inner: clause_slice 0 … N_CLAUSE_SLICES-1
  Each cycle: CLAUSE_SLICE parallel partial clause evaluations over 2*FEAT_SLICE literals.

Generated files under <output_dir>/RTL/
  src/  axis_fifo.v  clause_eval.v  score_acc.v  argmax.v  tm_accelerator.v
  tb/   tb_axis_fifo.v  tb_clause_eval.v  tb_score_acc.v  tb_argmax.v  tb_system.v
  sim/  run_iverilog.sh  lint_verilator.sh  run_xsim.sh  waves.sh
        tb_system.gtkw  tb_axis_fifo.gtkw  tb_clause_eval.gtkw  tb_score_acc.gtkw  tb_argmax.gtkw
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Optional

import numpy as np

from matador.ir.tm_ir import TMIR

# TMAcceleratorConfig is canonical in matador.config.schema; re-export here
# so callers can do `from matador.rtl.accelerator import TMAccelerator, TMAcceleratorConfig`.
from matador.config.schema import TMAcceleratorConfig  # noqa: F401


# ---------------------------------------------------------------------------
# Software model + code-gen class
# ---------------------------------------------------------------------------

class TMAccelerator:
    """Derives hardware parameters from a TMIR and generates all RTL files."""

    def __init__(self, tmir: TMIR, cfg: "TMAcceleratorConfig") -> None:
        if tmir.variant not in ("vanilla",):
            raise ValueError(
                f"RTL generation currently supports vanilla TM only; got {tmir.variant!r}"
            )
        if tmir.architecture.clause_organization != "per_class":
            raise ValueError("RTL generation requires per_class clause organisation")
        self.tmir = tmir
        self.cfg  = cfg
        self._derive_params()

    # ------------------------------------------------------------------
    # Parameter derivation
    # ------------------------------------------------------------------

    def _derive_params(self) -> None:
        arch = self.tmir.architecture
        self.n_features      = arch.n_features
        self.n_literals      = arch.n_literals          # 2 * n_features
        self.n_classes       = arch.n_classes
        self.n_clauses_pc    = arch.n_clauses_per_class  # type: ignore[assignment]
        self.n_clauses_total = arch.n_clauses_total
        self.threshold       = arch.threshold
        self.axis_dw         = self.cfg.axis_data_width
        self.fifo_depth      = self.cfg.fifo_depth

        # Tile dimensions
        self.feat_slice   = self.cfg.feat_slice
        self.clause_slice = self.cfg.clause_slice
        self.n_feat_slices   = math.ceil(self.n_features      / self.feat_slice)
        self.n_clause_slices = math.ceil(self.n_clauses_total / self.clause_slice)
        self.n_feat_padded   = self.n_feat_slices   * self.feat_slice    # >= n_features
        self.n_clause_padded = self.n_clause_slices * self.clause_slice  # >= n_clauses_total
        self.tile_width      = self.clause_slice * 2 * self.feat_slice
        self.n_tiles         = self.n_feat_slices * self.n_clause_slices

        # AXI beats needed for one feature vector
        self.n_beats = (self.n_features + self.axis_dw - 1) // self.axis_dw

        # Register widths
        self.score_width     = max(4, int(math.ceil(math.log2(self.threshold + 1))) + 2)
        self.class_width     = max(1, int(math.ceil(math.log2(max(self.n_classes, 2)))))
        self.clause_cnt_w    = max(2, int(math.ceil(math.log2(self.n_clauses_total + 1))))
        self.beat_cnt_w      = max(1, int(math.ceil(math.log2(self.n_beats + 1))))
        self.feat_cnt_w      = max(1, int(math.ceil(math.log2(self.n_feat_slices + 1))))
        self.cs_cnt_w        = max(1, int(math.ceil(math.log2(self.n_clause_slices + 1))))

        # Resolve TA Include-action outputs → shape (n_clauses_total, n_literals) bool
        # Each True entry means the trained TA for that literal issued an Include action.
        self.tmir.to_includes()
        inc = self.tmir.representation.includes
        self._ta_actions = inc.reshape(self.n_clauses_total, self.n_literals).astype(bool)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self, output_dir: Optional[Path] = None) -> Path:
        """Write all generated files; returns the RTL root directory."""
        root = (output_dir or self.cfg.output_dir) / "RTL"
        src  = root / "src"
        tb   = root / "tb"
        sim  = root / "sim"
        for d in (src, tb, sim):
            d.mkdir(parents=True, exist_ok=True)

        (src / "axis_fifo.v"      ).write_text(self._gen_axis_fifo())
        (src / "clause_eval.v"    ).write_text(self._gen_clause_eval())
        (src / "score_acc.v"      ).write_text(self._gen_score_acc())
        (src / "argmax.v"         ).write_text(self._gen_argmax())
        (src / "tm_accelerator.v" ).write_text(self._gen_tm_top())

        (tb / "tb_axis_fifo.v"   ).write_text(self._gen_tb_axis_fifo())
        (tb / "tb_clause_eval.v" ).write_text(self._gen_tb_clause_eval())
        (tb / "tb_score_acc.v"   ).write_text(self._gen_tb_score_acc())
        (tb / "tb_argmax.v"      ).write_text(self._gen_tb_argmax())
        (tb / "tb_system.v"      ).write_text(self._gen_tb_system())

        (sim / "run_iverilog.sh"  ).write_text(self._gen_iverilog_script())
        (sim / "lint_verilator.sh").write_text(self._gen_verilator_lint())
        (sim / "run_xsim.sh"     ).write_text(self._gen_xsim_script())
        (sim / "waves.sh"        ).write_text(self._gen_waves_sh())
        for f in (sim / "run_iverilog.sh", sim / "lint_verilator.sh",
                  sim / "run_xsim.sh",     sim / "waves.sh"):
            f.chmod(0o755)

        # Verilator C++ harness
        verilator_dir = sim / "verilator"
        verilator_dir.mkdir(exist_ok=True)
        (verilator_dir / "Makefile"  ).write_text(self._gen_verilator_makefile())
        (verilator_dir / "tb_top.cpp").write_text(self._gen_verilator_tb_cpp())

        # GTKWave save files
        from matador.waves.gtkwave import GtkWaveGenerator
        gtkw = GtkWaveGenerator(self)
        gtkw.generate_all(sim)

        # Documentation
        (root / "README.md").write_text(self._gen_readme())

        return root

    # ------------------------------------------------------------------
    # ROM helpers
    # ------------------------------------------------------------------

    def _tile_rom_init(self) -> str:
        """Inline initial block for the tiled TA action ROM.

        Each ROM entry stores the compiled TA Include-action outputs for one tile.
        Layout per tile at address (fi * N_CLAUSE_SLICES + ci):
          bits [ k*2*FEAT_SLICE +: FEAT_SLICE      ] = TA Include actions for positive literals, clause k
          bits [ k*2*FEAT_SLICE + FEAT_SLICE +: FS ] = TA Include actions for negated  literals, clause k
        where k = 0 .. CLAUSE_SLICE-1 within this tile.
        bit = 1  →  TA issued Include action for that literal in that clause
        bit = 0  →  TA issued Exclude action (literal not tested by this clause)
        """
        FS  = self.feat_slice
        CS  = self.clause_slice
        NFS = self.n_feat_slices
        NCS = self.n_clause_slices
        TW  = self.tile_width

        lines = ["    initial begin"]
        for fi in range(NFS):
            for ci in range(NCS):
                tile_val = 0
                for k in range(CS):
                    g = ci * CS + k           # global clause index
                    if g >= self.n_clauses_total:
                        continue              # padding clause → zero include bits
                    for l_local in range(FS):
                        feat_global = fi * FS + l_local
                        if feat_global >= self.n_features:
                            continue          # padding feature → zero
                        # Positive literal (global index = feat_global)
                        if self._ta_actions[g, feat_global]:
                            tile_val |= 1 << (k * 2 * FS + l_local)
                        # Negated literal (global index = n_features + feat_global)
                        if self._ta_actions[g, self.n_features + feat_global]:
                            tile_val |= 1 << (k * 2 * FS + FS + l_local)

                addr    = fi * NCS + ci
                hex_str = f"{tile_val:0{(TW + 3) // 4}X}"
                lines.append(
                    f"        tile_rom[{addr:5d}] = {TW}'h{hex_str};"
                    f"  // feat_slice={fi} clause_slice={ci}"
                )
        lines.append("    end")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Decode-function helpers (used in _gen_tm_top)
    # ------------------------------------------------------------------

    def _class_decode_cases(self) -> str:
        lines = []
        for c in range(self.n_clauses_total):
            cls = c // self.n_clauses_pc
            lines.append(f"                {c:5d}: clause_to_class    = {cls};")
        lines.append(f"             default: clause_to_class    = 0;")
        return "\n".join(lines)

    def _polarity_decode_cases(self) -> str:
        half = self.n_clauses_pc // 2
        lines = []
        for c in range(self.n_clauses_total):
            pol = "1'b1" if (c % self.n_clauses_pc) < half else "1'b0"
            lines.append(f"                {c:5d}: clause_is_positive = {pol};")
        lines.append(f"             default: clause_is_positive = 1'b0;")
        return "\n".join(lines)

    def _beat0_assign(self) -> str:
        hi   = min(self.axis_dw, self.n_features) - 1
        bits = hi + 1
        return f"feature_reg[{hi}:0] <= fifo_m_tdata[{bits - 1}:0];"

    def _beat_recv_cases(self) -> str:
        if self.n_beats == 1:
            return "                // single-beat: no S_RECV cases needed"
        lines = ["                case (beat_cnt)"]
        for k in range(1, self.n_beats):
            lo   = k * self.axis_dw
            hi   = min((k + 1) * self.axis_dw, self.n_features) - 1
            bits = hi - lo + 1
            lines.append(
                f"                    {k}: feature_reg[{hi}:{lo}] <= fifo_m_tdata[{bits - 1}:0];"
            )
        lines.append("                    default: ;")
        lines.append("                endcase")
        return "\n".join(lines)

    def _pack_beats(self, feature_bits: list) -> list:
        beats = []
        for k in range(self.n_beats):
            lo   = k * self.axis_dw
            hi   = min(lo + self.axis_dw, self.n_features)
            word = 0
            for i, bit in enumerate(feature_bits[lo:hi]):
                word |= (int(bit) & 1) << i
            beats.append(word)
        return beats

    def _get_test_vectors(self):
        if self.tmir.verification and self.tmir.verification.test_vectors:
            return self.tmir.verification.test_vectors[:10]
        return []

    # ------------------------------------------------------------------
    # Source generators
    # ------------------------------------------------------------------

    def _gen_axis_fifo(self) -> str:
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // =============================================================================
        // axis_fifo — synchronous AXI-Stream FIFO
        // =============================================================================
        // PURPOSE
        //   Buffers incoming AXI-Stream beats so that the producer does not need to
        //   synchronise with the FSM state in tm_accelerator. In the TM accelerator
        //   context this allows all feature beats to be pushed before the FSM has
        //   finished any previous housekeeping.
        //
        // IMPLEMENTATION
        //   Two-pointer circular buffer. wptr and rptr are one bit wider than the
        //   address so that full (pointers equal modulo depth but differ in MSB) is
        //   distinguished from empty (pointers exactly equal) without a separate
        //   occupancy counter.
        //
        //   Output is REGISTERED (not fall-through): m_tdata/m_tvalid appear one
        //   cycle after the read pointer advances. The tm_accelerator FSM accounts
        //   for this latency by sampling m_tdata in the clock after m_tvalid rises.
        //
        // HANDSHAKE
        //   AXI-Stream: a beat transfers on a rising clock edge where
        //   s_tvalid AND s_tready are both high.
        //   s_tready is combinatorially asserted whenever the FIFO is not full.
        //
        // VERILOG-2001. No SystemVerilog. No timing constructs.
        // Depth must be a power of 2 (required for two-pointer full/empty logic).
        // =============================================================================
        module axis_fifo #(
            parameter DATA_WIDTH = 32,
            parameter DEPTH      = 16    // must be power of 2
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
            localparam DW1 = DATA_WIDTH + 1;  // +1 for tlast

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

    def _gen_clause_eval(self) -> str:
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // =============================================================================
        // clause_eval — combinatorial single-clause evaluator
        // =============================================================================
        // PURPOSE
        //   Implements one clause-firing test from the Tsetlin Machine inference rule.
        //   Given the Include-action bits for one clause (ta_action_mask) and the
        //   current literal values (literals), outputs whether the clause is active.
        //
        // TM CLAUSE RULE
        //   A Tsetlin Automaton (TA) is trained per (clause, literal) pair.
        //   When a TA's state exceeds n_states/2 it issues an Include action for its
        //   literal; below that midpoint it issues Exclude. After training, the states
        //   are thresholded to a single bit: Include = (state > n_states/2).
        //   This module receives those Include bits as ta_action_mask.
        //
        //   Firing rule:
        //     (1) The clause must have at least one Include  (ta_action_mask != 0)
        //     (2) Every Included literal must currently be 1
        //   An empty clause (all Exclude) is NEVER active regardless of literals.
        //
        // LITERAL ORDERING  — must match the ROM layout in tm_accelerator.v
        //   literals[l]             feature[l]          (positive literal)
        //   literals[N_LITERALS/2 + l]  NOT feature[l]  (negated  literal)
        //   ta_action_mask[l]       Include bit for literal l (same layout)
        //
        // COMBINATORIAL — no registers; output settles within the clock cycle.
        //   In tm_accelerator, CLAUSE_SLICE instances share one feature window.
        //   Only ta_action_mask differs across instances (one ROM row per clause).
        //   all_active and no_actions are exposed as named wires for waveform
        //   inspection in tb_clause_eval.
        //
        // VERILOG-2001. No SystemVerilog. No timing constructs.
        // =============================================================================
        module clause_eval #(
            parameter N_LITERALS = 16
        )(
            input  wire [N_LITERALS-1:0] literals,       // [F-1:0]=positive  [2F-1:F]=negated
            input  wire [N_LITERALS-1:0] ta_action_mask, // compiled TA Include-action bits
            output wire                  active
        );
            wire no_actions = ~|ta_action_mask;
            wire all_active = &(literals | ~ta_action_mask);
            assign active = all_active & ~no_actions;
        endmodule
        """)

    def _gen_score_acc(self) -> str:
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // =============================================================================
        // score_acc — per-class vote accumulator
        // =============================================================================
        // PURPOSE
        //   Maintains one signed score per class. During the S_SCORE phase of
        //   tm_accelerator, one clause result is presented each clock cycle. This
        //   module adds or subtracts 1 from the score of the targeted class.
        //
        // TM VOTING CONVENTION
        //   Each class has N_CLAUSES_PC clauses. The first half are positive (polarity=1)
        //   — an active positive clause increments score[class] by 1. The second half
        //   are negative (polarity=0) — an active negative clause decrements by 1.
        //   Only *active* clauses vote; inactive clauses leave scores unchanged.
        //
        // CLAMPING
        //   Scores are clamped to [-THRESHOLD, THRESHOLD-1]. The guard is:
        //     positive: update only if score < THRESHOLD  (stops at THRESHOLD-1)
        //     negative: update only if score > -THRESHOLD (stops at -(THRESHOLD-1))
        //   Guarded increment/decrement (not clip-after-add) avoids overflow.
        //
        // OUTPUT FORMAT
        //   scores_flat is a flat packed bus. Class j occupies bits [j*SW +: SW] as a
        //   SCORE_WIDTH-bit signed two's-complement integer. Unpack in Verilog with:
        //     $signed(scores_flat[j*SCORE_WIDTH +: SCORE_WIDTH])
        //
        // CLEAR PROTOCOL
        //   Assert clear for one clock cycle after each inference. clear has priority
        //   over valid: if both are asserted, all scores clear to 0.
        //   In tm_accelerator, score_clear is asserted in S_DONE and the module
        //   clears in the first cycle of the following S_IDLE.
        //
        // VERILOG-2001. No SystemVerilog. No timing constructs.
        // =============================================================================
        module score_acc #(
            parameter N_CLASSES   = 2,
            parameter THRESHOLD   = 4,
            parameter SCORE_WIDTH = 8
        )(
            input  wire                                   clk,
            input  wire                                   rst_n,
            input  wire                                   clear,
            input  wire                                   valid,
            input  wire [$clog2(N_CLASSES)-1:0]           cls,
            input  wire                                   polarity,
            input  wire                                   active,
            output wire [SCORE_WIDTH*N_CLASSES-1:0]       scores_flat
        );

            localparam [SCORE_WIDTH-1:0] THRESH = THRESHOLD[SCORE_WIDTH-1:0];

            reg signed [SCORE_WIDTH-1:0] scores [0:N_CLASSES-1];
            integer i;

            genvar g;
            generate
                for (g = 0; g < N_CLASSES; g = g + 1) begin : pack_scores
                    assign scores_flat[g*SCORE_WIDTH +: SCORE_WIDTH] = scores[g];
                end
            endgenerate

            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (i = 0; i < N_CLASSES; i = i + 1)
                        scores[i] = {{SCORE_WIDTH{{1'b0}}}};
                end else if (clear) begin
                    for (i = 0; i < N_CLASSES; i = i + 1)
                        scores[i] = {{SCORE_WIDTH{{1'b0}}}};
                end else if (valid & active) begin
                    if (polarity) begin
                        if ($signed(scores[cls]) < $signed(THRESH))
                            scores[cls] <= scores[cls] + 1;
                    end else begin
                        if ($signed(scores[cls]) > -$signed(THRESH))
                            scores[cls] <= scores[cls] - 1;
                    end
                end
            end

        endmodule
        """)

    def _gen_argmax(self) -> str:
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // =============================================================================
        // argmax — combinatorial tournament argmax over signed scores
        // =============================================================================
        // PURPOSE
        //   Finds the class with the highest score after the S_SCORE accumulation phase.
        //   The winning class index is latched into pred_class by tm_accelerator in S_DONE.
        //
        // ALGORITHM
        //   Linear scan: initialise winner = class 0, then compare classes 1..N-1 in
        //   order. Update winner only when a strictly greater score is found. Ties go
        //   to the lower-indexed class (the initial winner is never displaced by an equal).
        //
        // COMBINATORIAL
        //   No registers. Output pred_class settles within the same cycle that
        //   scores_flat is stable (i.e. the cycle after the last S_SCORE vote).
        //   tm_accelerator samples argmax_out in S_DONE: pred_class <= argmax_out.
        //
        // INPUT FORMAT
        //   scores_flat is a packed bus of N_CLASSES × SCORE_WIDTH signed integers.
        //   Class j occupies bits [j*SCORE_WIDTH +: SCORE_WIDTH] in two's complement.
        //   This matches the output format of score_acc.v.
        //
        // VERILOG-2001. No SystemVerilog. No timing constructs.
        // =============================================================================
        module argmax #(
            parameter N_CLASSES   = 2,
            parameter SCORE_WIDTH = 8,
            parameter CLASS_WIDTH = 1
        )(
            input  wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat,
            output reg  [CLASS_WIDTH-1:0]            pred_class
        );
            integer k;
            reg signed [SCORE_WIDTH-1:0] cur_max;
            reg signed [SCORE_WIDTH-1:0] score_k;

            always @(*) begin
                cur_max    = $signed(scores_flat[SCORE_WIDTH-1:0]);
                pred_class = {{CLASS_WIDTH{{1'b0}}}};
                for (k = 1; k < N_CLASSES; k = k + 1) begin
                    score_k = $signed(scores_flat[k*SCORE_WIDTH +: SCORE_WIDTH]);
                    if (score_k > cur_max) begin
                        cur_max    = score_k;
                        pred_class = k[CLASS_WIDTH-1:0];
                    end
                end
            end
        endmodule
        """)

    def _gen_tm_top(self) -> str:
        # Short aliases for readability
        N   = self.n_features
        NLP = self.n_feat_padded
        C   = self.n_classes
        K   = self.n_clauses_pc
        CT  = self.n_clauses_total
        T   = self.threshold
        NB  = self.n_beats
        AW  = self.axis_dw
        SW  = self.score_width
        CW  = self.class_width
        FD  = self.fifo_depth
        FS  = self.feat_slice
        CS  = self.clause_slice
        NFS = self.n_feat_slices
        NCS = self.n_clause_slices
        TW  = self.tile_width
        NT  = self.n_tiles
        CCW = self.clause_cnt_w
        BCW = self.beat_cnt_w
        FCW = self.feat_cnt_w
        CSCW = self.cs_cnt_w

        latency = NFS * NCS + CT + 6

        tile_rom_init         = self._tile_rom_init()
        class_decode_cases    = self._class_decode_cases()
        polarity_decode_cases = self._polarity_decode_cases()
        beat0_assign          = self._beat0_assign()
        beat_recv_cases       = self._beat_recv_cases()

        if NLP > N:
            padding_comment = (
                f"//   Bits [{NLP-1}:{N}] are permanently zero; the corresponding literal\n"
                f"//   bits in partial_lits are also zero (positive) or one (negated).\n"
                f"//   Tiles whose ta_action_mask has Include bits only in the padding zone\n"
                f"//   cannot fire, but such bits never arise in a correctly compiled ROM\n"
                f"//   because the model has no features beyond index {N-1}.\n"
            )
        else:
            padding_comment = (
                f"//   In this model N_FEATURES={N} divides exactly into FEAT_SLICE={FS}\n"
                f"//   (N_FEAT_SLICES={NFS}), so the feature register carries no padding bits.\n"
            )

        return (
            f"`timescale 1ns/1ps\n"
            f"// =============================================================================\n"
            f"// tm_accelerator — Vanilla Tsetlin Machine inference accelerator\n"
            f"// =============================================================================\n"
            f"// PURPOSE\n"
            f"//   Accepts a {N}-bit binary feature vector over AXI-Stream and outputs the\n"
            f"//   predicted class index from a trained {C}-class Tsetlin Machine.\n"
            f"//   Extracted TA Include-action outputs from the trained model are compiled\n"
            f"//   into an on-chip tile ROM at generation time. No multipliers, no\n"
            f"//   floating-point: all operations are bitwise.\n"
            f"//\n"
            f"// PARTIAL COMPUTE MATRIX AND TILING\n"
            f"//\n"
            f"//   The full TM inference problem is a ({CT} clauses) × ({N} features)\n"
            f"//   boolean evaluation: each clause must AND-test its included literals\n"
            f"//   across all {N} feature bits before a result is known.  Doing this in\n"
            f"//   one cycle would require {CT} × {2*N}-bit AND-reduce trees and a\n"
            f"//   ({CT} × {2*N})-bit ROM — impractical for even moderate model sizes.\n"
            f"//\n"
            f"//   Instead, the evaluation is decomposed into a partial compute matrix:\n"
            f"//\n"
            f"//                     clauses →\n"
            f"//              0   {CS}  {2*CS}  ..  {CT}\n"
            f"//           ┌────┬────┬────┬────┐\n"
            f"//     f  0  │ T  │ T  │ T  │ T  │   feat_slice 0  (features  0..{FS-1})\n"
            f"//     e  1  │ T  │ T  │ T  │ T  │   feat_slice 1  (features {FS}..{2*FS-1})\n"
            f"//     a  .  │ .  │ .  │ .  │ .  │\n"
            f"//     t  .  │ .  │ .  │ .  │ .  │\n"
            f"//     ↓ {NFS-1}  │ T  │ T  │ T  │ T  │   feat_slice {NFS-1} (features {(NFS-1)*FS}..{NLP-1})\n"
            f"//           └────┴────┴────┴────┘\n"
            f"//             {NCS} clause slices  ({NCS} columns)\n"
            f"//\n"
            f"//   Each cell T is one tile: FEAT_SLICE={FS} features × CLAUSE_SLICE={CS} clauses.\n"
            f"//   The grid has {NFS} rows × {NCS} columns = {NT} tiles total.\n"
            f"//   One tile is evaluated per clock cycle in S_COMPUTE.\n"
            f"//\n"
            f"//   Partial results are accumulated across feature slices (rows):\n"
            f"//     clause_pass[g]        initialised TRUE, then AND-reduced over all\n"
            f"//                           {NFS} row results for clause g.  A clause fires\n"
            f"//                           only if it passes in every feature window.\n"
            f"//     clause_has_actions[g] initialised FALSE, then OR-reduced.  Set if\n"
            f"//                           clause g has at least one TA Include action\n"
            f"//                           anywhere in its {NFS} tiles (non-empty clause).\n"
            f"//   Final firing condition: clause_pass[g] & clause_has_actions[g].\n"
            f"//   An all-Exclude clause (has_actions=0) never fires regardless of features.\n"
            f"//\n"
            f"//   Throughput: S_COMPUTE takes exactly {NT} cycles regardless of the\n"
            f"//   feature values.  The {CS}-wide parallelism across the clause dimension\n"
            f"//   means only {NCS} cycle-steps are needed per feature window, keeping\n"
            f"//   total inference latency at ≈{latency} cycles end-to-end.\n"
            f"//\n"
            f"// N_FEAT_PADDED = {NLP}  (= N_FEAT_SLICES × FEAT_SLICE = {NFS} × {FS})\n"
            f"//   N_FEATURES={N} may not be an exact multiple of FEAT_SLICE={FS}.\n"
            f"//   The feature register is widened to {NLP} bits so that every\n"
            f"//   feat_cnt window is exactly {FS} bits wide with no boundary arithmetic.\n"
            f"{padding_comment}"
            f"//\n"
            f"// FIVE-STATE FSM OVERVIEW\n"
            f"//\n"
            f"//   S_IDLE    Wait for first AXI beat from FIFO.\n"
            f"//             Capture beat 0 into feature_reg[{AW-1}:0].\n"
            f"//\n"
            f"//   S_RECV    Capture beats 1..{NB-1} into feature_reg via case(beat_cnt).\n"
            f"//             Transition to S_COMPUTE on fifo_m_tlast.\n"
            f"//\n"
            f"//   S_COMPUTE {NFS} rows × {NCS} columns = {NFS*NCS} tile cycles.\n"
            f"//             feat_cnt (outer loop, 0→{NFS-1}) selects the feature window;\n"
            f"//             clause_slice_cnt (inner loop, 0→{NCS-1}) selects the clause\n"
            f"//             group. Together they address one tile per cycle.\n"
            f"//             {CS} clause_eval instances run in parallel, each receiving\n"
            f"//             the same {2*FS}-bit partial_lits and a different {2*FS}-bit\n"
            f"//             ta_action_mask from the ROM row.  Results are AND-accumulated\n"
            f"//             into clause_pass[] and OR-accumulated into clause_has_actions[].\n"
            f"//             Empty tiles vacuously pass:\n"
            f"//               clause_pass[g] <= tile_pass[k] | ~tile_has_actions[k]\n"
            f"//\n"
            f"//   S_SCORE   {CT} cycles. score_cnt = 0..{CT-1}.\n"
            f"//             score_acc receives one clause result per cycle.\n"
            f"//             score_class  = score_cnt / {K}  (class index)\n"
            f"//             score_is_pos = (score_cnt % {K}) < {K//2}  (first half = positive)\n"
            f"//             score_active = clause_pass[score_cnt] & clause_has_actions[score_cnt]\n"
            f"//\n"
            f"//   S_DONE    Assert m_axis_tvalid/tlast/tdata with argmax result.\n"
            f"//             Hold until m_axis_tready (back-pressure supported).\n"
            f"//             Assert score_clear on the handshake cycle.\n"
            f"//\n"
            f"// TILE ROM LAYOUT\n"
            f"//   Depth    = {NT} entries  (N_FEAT_SLICES={NFS} × N_CLAUSE_SLICES={NCS})\n"
            f"//   Address  = feat_cnt × {NCS} + clause_slice_cnt\n"
            f"//            = row-major order: inner index is clause slice (column)\n"
            f"//   Width    = {TW} bits  (CLAUSE_SLICE={CS} clauses × 2 × FEAT_SLICE={FS} literals)\n"
            f"//   Within each entry, clause k occupies bits [{2*FS}×k+{2*FS}-1 : {2*FS}×k]:\n"
            f"//     bits [{2*FS}×k + {FS-1} : {2*FS}×k]        Include for positive literals 0..{FS-1}\n"
            f"//     bits [{2*FS}×k + {2*FS-1} : {2*FS}×k + {FS}]  Include for negated  literals 0..{FS-1}\n"
            f"//   This layout matches partial_lits = {{~feat_pos, feat_pos}}, so clause_eval\n"
            f"//   needs no bit reordering — ta_action_mask and literals are index-aligned.\n"
            f"//\n"
            f"// LATENCY   last AXI TLAST accepted → m_axis_tvalid: ≈{latency} clock cycles.\n"
            f"// VERILOG-2001. No SystemVerilog. Timing-free RTL.\n"
            f"// =============================================================================\n"
            f"module tm_accelerator #(\n"
            f"    // ── Frozen model parameters ─────────────────────────────────────────\n"
            f"    // N_FEATURES, N_CLAUSES_PC, N_BEATS are documentation-only; the RTL uses\n"
            f"    // concrete literals generated below.  Suppress Verilator UNUSED lint.\n"
            f"    /* verilator lint_off UNUSED */\n"
            f"    parameter N_FEATURES      = {N},\n"
            f"    parameter N_CLASSES       = {C},\n"
            f"    parameter N_CLAUSES_PC    = {K},\n"
            f"    parameter N_CLAUSES_TOTAL = {CT},\n"
            f"    parameter THRESHOLD       = {T},\n"
            f"    parameter N_BEATS         = {NB},\n"
            f"    /* verilator lint_on UNUSED */\n"
            f"    // ── Tile parameters (user-specified via accelerator_config.yaml) ─────\n"
            f"    parameter FEAT_SLICE      = {FS},\n"
            f"    parameter CLAUSE_SLICE    = {CS},\n"
            f"    parameter N_FEAT_SLICES   = {NFS},\n"
            f"    parameter N_CLAUSE_SLICES = {NCS},\n"
            f"    parameter N_FEAT_PADDED   = {NLP},  // FEAT_SLICE * N_FEAT_SLICES\n"
            f"    parameter TILE_WIDTH      = {TW},   // CLAUSE_SLICE * 2 * FEAT_SLICE\n"
            f"    parameter N_TILES         = {NT},   // N_FEAT_SLICES * N_CLAUSE_SLICES\n"
            f"    // ── Interface parameters ────────────────────────────────────────────\n"
            f"    parameter AXIS_DATA_WIDTH = {AW},\n"
            f"    parameter SCORE_WIDTH     = {SW},\n"
            f"    parameter CLASS_WIDTH     = {CW},\n"
            f"    parameter FIFO_DEPTH      = {FD}\n"
            f")(\n"
            f"    input  wire                       clk,\n"
            f"    input  wire                       rst_n,\n"
            f"    input  wire                       s_axis_tvalid,\n"
            f"    output wire                       s_axis_tready,\n"
            f"    input  wire [AXIS_DATA_WIDTH-1:0] s_axis_tdata,\n"
            f"    input  wire                       s_axis_tlast,\n"
            f"    // ── AXI-Stream output (predicted class) ─────────────────────────────\n"
            f"    // m_axis_tlast is wired to m_axis_tvalid (single-beat packet, always last).\n"
            f"    output reg                        m_axis_tvalid,\n"
            f"    input  wire                       m_axis_tready,\n"
            f"    output reg  [AXIS_DATA_WIDTH-1:0] m_axis_tdata,\n"
            f"    output reg                        m_axis_tlast,\n"
            f"    output reg                        busy\n"
            f");\n"
            f"\n"
            f"    localparam [2:0] S_IDLE    = 3'd0;\n"
            f"    localparam [2:0] S_RECV    = 3'd1;\n"
            f"    localparam [2:0] S_COMPUTE = 3'd2;\n"
            f"    localparam [2:0] S_SCORE   = 3'd3;\n"
            f"    localparam [2:0] S_DONE    = 3'd4;\n"
            f"\n"
            f"    localparam [{FCW}-1:0]  LAST_FEAT   = N_FEAT_SLICES   - 1;\n"
            f"    localparam [{CSCW}-1:0] LAST_CS     = N_CLAUSE_SLICES - 1;\n"
            f"    localparam [{CCW}-1:0]  LAST_CLAUSE = N_CLAUSES_TOTAL - 1;\n"
            f"\n"
            f"    // ── FIFO ─────────────────────────────────────────────────────────────\n"
            f"    wire                       fifo_m_tvalid;\n"
            f"    wire [AXIS_DATA_WIDTH-1:0] fifo_m_tdata;\n"
            f"    wire                       fifo_m_tlast;\n"
            f"    wire                       fifo_m_tready;\n"
            f"\n"
            f"    assign fifo_m_tready = (state == S_IDLE) | (state == S_RECV);\n"
            f"\n"
            f"    axis_fifo #(.DATA_WIDTH(AXIS_DATA_WIDTH), .DEPTH(FIFO_DEPTH)) u_fifo (\n"
            f"        .clk      (clk),           .rst_n    (rst_n),\n"
            f"        .s_tvalid (s_axis_tvalid), .s_tready (s_axis_tready),\n"
            f"        .s_tdata  (s_axis_tdata),  .s_tlast  (s_axis_tlast),\n"
            f"        .m_tvalid (fifo_m_tvalid), .m_tready (fifo_m_tready),\n"
            f"        .m_tdata  (fifo_m_tdata),  .m_tlast  (fifo_m_tlast)\n"
            f"    );\n"
            f"\n"
            f"    // ── FSM counters and state ────────────────────────────────────────────\n"
            f"    reg [2:0]      state;\n"
            f"    reg [{BCW}-1:0]  beat_cnt;\n"
            f"    // frame_tlast: latched from s_axis_tlast on the last beat of each frame.\n"
            f"    // Propagated to m_axis_tlast so the output marks the end of the same batch.\n"
            f"    reg            frame_tlast;\n"
            f"    reg [{FCW}-1:0]  feat_cnt;          // feature slice index\n"
            f"    reg [{CSCW}-1:0] clause_slice_cnt;  // clause slice index\n"
            f"    reg [{CCW}-1:0]  score_cnt;          // clause counter during S_SCORE\n"
            f"\n"
            f"    // ── Feature register (padded to N_FEAT_PADDED for clean slice indexing) ─\n"
            f"    reg [N_FEAT_PADDED-1:0] feature_reg;\n"
            f"\n"
            f"    // ── Partial literals for current feature slice ────────────────────────\n"
            f"    // feat_pos: FEAT_SLICE positive literal values from current window.\n"
            f"    // partial_lits = {{~feat_pos, feat_pos}}: layout matches ta_action_mask.\n"
            f"    wire [FEAT_SLICE-1:0]   feat_pos     = feature_reg[feat_cnt * FEAT_SLICE +: FEAT_SLICE];\n"
            f"    wire [2*FEAT_SLICE-1:0] partial_lits = {{~feat_pos, feat_pos}};\n"
            f"\n"
            f"    // ── Tile ROM (compiled TA action outputs) ────────────────────────────\n"
            f"    // Row address: feat_cnt * N_CLAUSE_SLICES + clause_slice_cnt\n"
            f"    // Row data (TILE_WIDTH bits): clause k at bits [k*2*FEAT_SLICE +: 2*FEAT_SLICE]\n"
            f"    //   [0..FS-1]    TA Include actions for positive literals\n"
            f"    //   [FS..2*FS-1] TA Include actions for negated  literals\n"
            f"    reg [TILE_WIDTH-1:0] tile_rom [0:N_TILES-1];\n"
            f"\n"
            f"{tile_rom_init}\n"
            f"\n"
            f"    wire [TILE_WIDTH-1:0] tile_rom_data =\n"
            f"        tile_rom[feat_cnt * N_CLAUSE_SLICES + clause_slice_cnt];\n"
            f"\n"
            f"    // ── Parallel tile evaluators (CLAUSE_SLICE clause_eval instances) ─────\n"
            f"    // Each clause_eval receives the 2*FEAT_SLICE-bit ta_action_mask from the tile ROM.\n"
            f"    // tile_pass[k]=1 iff clause k fires for this feature-slice window.\n"
            f"    // tile_has_actions[k]=1 iff clause k has at least one TA Include action in this tile.\n"
            f"    wire [CLAUSE_SLICE-1:0] tile_pass;\n"
            f"    wire [CLAUSE_SLICE-1:0] tile_has_actions;\n"
            f"\n"
            f"    genvar k;\n"
            f"    generate\n"
            f"        for (k = 0; k < CLAUSE_SLICE; k = k + 1) begin : tile_eval\n"
            f"            wire [2*FEAT_SLICE-1:0] ta_actions_k;\n"
            f"            assign ta_actions_k = tile_rom_data[k * 2 * FEAT_SLICE +: 2 * FEAT_SLICE];\n"
            f"\n"
            f"            clause_eval #(.N_LITERALS(2 * FEAT_SLICE)) u_ce (\n"
            f"                .literals      (partial_lits),\n"
            f"                .ta_action_mask(ta_actions_k),\n"
            f"                .active        (tile_pass[k])\n"
            f"            );\n"
            f"\n"
            f"            assign tile_has_actions[k] = |ta_actions_k;\n"
            f"        end\n"
            f"    endgenerate\n"
            f"\n"
            f"    // ── Clause running-AND/OR registers ──────────────────────────────────\n"
            f"    // clause_pass[g]        = AND of tile_pass    across all feature slices.\n"
            f"    // clause_has_actions[g] = OR  of tile_has_actions (detects empty clauses).\n"
            f"    // A clause is empty if no TA ever issued an Include action for any literal.\n"
            f"    reg clause_pass        [0:N_CLAUSES_TOTAL-1];\n"
            f"    reg clause_has_actions [0:N_CLAUSES_TOTAL-1];\n"
            f"\n"
            f"    // ── Score-accumulation decode (active during S_SCORE) ─────────────────\n"
            f"    function [{CW}-1:0] clause_to_class;\n"
            f"        input [{CCW}-1:0] c;\n"
            f"        begin\n"
            f"            case (c)\n"
            f"{class_decode_cases}\n"
            f"            endcase\n"
            f"        end\n"
            f"    endfunction\n"
            f"\n"
            f"    function clause_is_positive;\n"
            f"        input [{CCW}-1:0] c;\n"
            f"        begin\n"
            f"            case (c)\n"
            f"{polarity_decode_cases}\n"
            f"            endcase\n"
            f"        end\n"
            f"    endfunction\n"
            f"\n"
            f"    wire [{CW}-1:0] score_class  = clause_to_class(score_cnt);\n"
            f"    wire            score_is_pos = clause_is_positive(score_cnt);\n"
            f"    wire            score_active = clause_pass[score_cnt] & clause_has_actions[score_cnt];\n"
            f"\n"
            f"    // ── Score accumulator ─────────────────────────────────────────────────\n"
            f"    reg  score_clear;\n"
            f"    wire score_valid = (state == S_SCORE);\n"
            f"    wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat;\n"
            f"\n"
            f"    score_acc #(\n"
            f"        .N_CLASSES  (N_CLASSES),\n"
            f"        .THRESHOLD  (THRESHOLD),\n"
            f"        .SCORE_WIDTH(SCORE_WIDTH)\n"
            f"    ) u_score_acc (\n"
            f"        .clk(clk), .rst_n(rst_n), .clear(score_clear), .valid(score_valid),\n"
            f"        .cls(score_class), .polarity(score_is_pos), .active(score_active),\n"
            f"        .scores_flat(scores_flat)\n"
            f"    );\n"
            f"\n"
            f"    // ── Argmax (combinatorial) ────────────────────────────────────────────\n"
            f"    wire [{CW}-1:0] argmax_out;\n"
            f"    argmax #(.N_CLASSES(N_CLASSES), .SCORE_WIDTH(SCORE_WIDTH),\n"
            f"             .CLASS_WIDTH(CLASS_WIDTH)) u_argmax (\n"
            f"        .scores_flat(scores_flat), .pred_class(argmax_out)\n"
            f"    );\n"
            f"\n"
            f"    // ── Loop variables (module-level for procedural use in FSM) ───────────\n"
            f"    integer k_upd;    // clause-slice loop variable\n"
            f"    integer clause_g; // global clause index = clause_slice_cnt*CS + k_upd\n"
            f"    integer ri;       // reset-loop variable\n"
            f"\n"
            f"    // ── FSM ───────────────────────────────────────────────────────────────\n"
            f"    always @(posedge clk or negedge rst_n) begin\n"
            f"        if (!rst_n) begin\n"
            f"            state            <= S_IDLE;\n"
            f"            beat_cnt         <= {{{BCW}{{1'b0}}}};\n"
            f"            feat_cnt         <= {{{FCW}{{1'b0}}}};\n"
            f"            clause_slice_cnt <= {{{CSCW}{{1'b0}}}};\n"
            f"            score_cnt        <= {{{CCW}{{1'b0}}}};\n"
            f"            feature_reg      <= {{{NLP}{{1'b0}}}};\n"
            f"            m_axis_tvalid    <= 1'b0;\n"
            f"            m_axis_tdata     <= {{{AW}{{1'b0}}}};\n"
            f"            m_axis_tlast     <= 1'b0;\n"
            f"            frame_tlast      <= 1'b0;\n"
            f"            busy             <= 1'b0;\n"
            f"            score_clear      <= 1'b0;\n"
            f"            for (ri = 0; ri < N_CLAUSES_TOTAL; ri = ri + 1) begin\n"
            f"                clause_pass[ri]        = 1'b0;\n"
            f"                clause_has_actions[ri] = 1'b0;\n"
            f"            end\n"
            f"        end else begin\n"
            f"            m_axis_tvalid <= 1'b0;  // default: deassert when not in S_DONE\n"
            f"            m_axis_tlast  <= 1'b0;\n"
            f"            score_clear   <= 1'b0;  // default: no clear\n"
            f"\n"
            f"            case (state)\n"
            f"                // ─── S_IDLE phase ─────────────────────────────────────────\n"
            f"                //\n"
            f"                // The module rests here between inferences.\n"
            f"                // If score_clear was pulsed at the end of S_DONE, score_acc\n"
            f"                // zeros all {C} scores[] on this first posedge — guaranteed\n"
            f"                // before the next S_SCORE begins.\n"
            f"                //\n"
            f"                // S_IDLE doubles as the capture state for beat 0: when the\n"
            f"                // FIFO presents a valid word (fifo_m_tvalid=1) the FSM reads\n"
            f"                // it here and advances immediately, so no cycle is wasted\n"
            f"                // waiting before the first beat is consumed.\n"
            f"                //\n"
            f"                // Signals while idle (fifo_m_tvalid=0):\n"
            f"                //   busy           (1-bit) = 0\n"
            f"                //   beat_cnt      ({BCW}-bit) = 0  (held at zero)\n"
            f"                //   m_axis_tvalid  (1-bit) = 0  (default, from else branch)\n"
            f"                //\n"
            f"                // On first valid beat (fifo_m_tvalid=1):\n"
            f"                //   feature_reg[{AW-1}:0]  ({AW}-bit) <= fifo_m_tdata  // beat 0\n"
            f"                //   beat_cnt       ({BCW}-bit) <= 1\n"
            f"                //   busy           (1-bit) <= 1\n"
            + (
                f"                //   frame_tlast    (1-bit) <= fifo_m_tlast  // single-beat: capture now\n"
                f"                //   feat_cnt, clause_slice_cnt <= 0\n"
                f"                //   state                      <= S_COMPUTE  // skip S_RECV\n"
                if NB == 1 else
                f"                //   state          <= S_RECV   // beats 1..{NB-1} follow\n"
            ) +
            f"                //\n"
            f"                // Note — fifo_m_tready = (state==S_IDLE)|(state==S_RECV).\n"
            f"                //   The FIFO output is consumed only in these two states.\n"
            f"                //   During S_COMPUTE / S_SCORE / S_DONE the FIFO accumulates\n"
            f"                //   the next inference's beats, enabling back-to-back pipelining\n"
            f"                //   without stalling the upstream AXI-Stream source.\n"
            f"                // ──────────────────────────────────────────────────────────────\n"
            f"                S_IDLE: begin\n"
            f"                    busy     <= 1'b0;\n"
            f"                    beat_cnt <= {{{BCW}{{1'b0}}}};\n"
            f"                    if (fifo_m_tvalid) begin\n"
            f"                        {beat0_assign}\n"
            f"                        beat_cnt <= {{{BCW}{{1'b0}}}} + 1'b1;\n"
            f"                        busy     <= 1'b1;\n"
            + (
                # Single-beat vector: beat 0 is also the last beat of the frame
                f"                        frame_tlast      <= fifo_m_tlast;\n"
                f"                        feat_cnt         <= {{{FCW}{{1'b0}}}};\n"
                f"                        clause_slice_cnt <= {{{CSCW}{{1'b0}}}};\n"
                f"                        state            <= S_COMPUTE;\n"
                if NB == 1 else
                # Multi-beat vector: always move to S_RECV; frame end detected by beat counter
                f"                        state <= S_RECV;\n"
            ) +
            f"                    end\n"
            f"                end\n"
            f"\n"
            f"                // ─── S_RECV phase ─────────────────────────────────────────\n"
            f"                //\n"
            f"                // Captures beats 1..{NB-1} from the FIFO into feature_reg,\n"
            f"                // one beat per valid FIFO cycle.  If the FIFO is transiently\n"
            f"                // empty (fifo_m_tvalid=0) the FSM stalls and beat_cnt holds —\n"
            f"                // AXI back-pressure is absorbed by the FIFO without FSM change.\n"
            f"                //\n"
            f"                // Pseudocode (one iteration per valid FIFO beat):\n"
            f"                //\n"
            f"                //   for beat_cnt in 1..{NB-1}:      // SEQUENTIAL, stalls if FIFO empty\n"
            f"                //\n"
            f"                //     wait until fifo_m_tvalid = 1\n"
            f"                //     ══════════ ONE CLOCK CYCLE ══════════\n"
            f"                //\n"
            f"                //     feature_reg[beat_cnt×{AW} +: {AW}] ({AW}-bit) <= fifo_m_tdata\n"
            f"                //\n"
            f"                //     if beat_cnt == {NB-1}:          // last beat of this frame\n"
            f"                //       frame_tlast      (1-bit) <= fifo_m_tlast\n"
            f"                //         // 1 if this frame ends the AXI packet (batch-level tlast)\n"
            f"                //         // 0 for every intermediate frame in the batch\n"
            f"                //       feat_cnt         ({FCW}-bit) <= 0\n"
            f"                //       clause_slice_cnt ({CSCW}-bit) <= 0\n"
            f"                //       state                    <= S_COMPUTE\n"
            f"                //\n"
            f"                //     ══════════ CYCLE ENDS ══════════\n"
            f"                //\n"
            f"                // frame_tlast: captures fifo_m_tlast on beat {NB-1} only.\n"
            f"                //   The upstream AXI master sets tlast=1 on the last beat of\n"
            f"                //   the last frame in the batch; all other beats carry tlast=0.\n"
            f"                //   frame_tlast is held in a register until S_DONE, where it is\n"
            f"                //   forwarded to m_axis_tlast on the output handshake.\n"
            f"                //\n"
            f"                // After S_RECV exits: feature_reg[{NLP-1}:0] holds the full\n"
            f"                //   {N}-bit feature vector packed as {NB}×{AW}-bit AXI words:\n"
            f"                //     feature_reg[b×{AW} +: {AW}] = beat b,  b = 0..{NB-1}\n"
            + (
                f"                //     feature_reg[{NLP-1}:{N}] = 0 (padding)\n"
                if NLP > N else
                f"                //   No padding bits (N_FEATURES={N} = {NB}×{AW} exactly).\n"
            ) +
            f"                // ──────────────────────────────────────────────────────────────\n"
            f"                S_RECV: begin\n"
            f"                    if (fifo_m_tvalid) begin\n"
            f"{beat_recv_cases}\n"
            f"                        beat_cnt <= beat_cnt + 1'b1;\n"
            f"                        if (beat_cnt == {NB-1}) begin\n"
            f"                            frame_tlast      <= fifo_m_tlast;\n"
            f"                            feat_cnt         <= {{{FCW}{{1'b0}}}};\n"
            f"                            clause_slice_cnt <= {{{CSCW}{{1'b0}}}};\n"
            f"                            state            <= S_COMPUTE;\n"
            f"                        end\n"
            f"                    end\n"
            f"                end\n"
            f"\n"
            f"                // ─── S_COMPUTE phase of one inference ─────────────────────\n"
            f"                //\n"
            f"                // Entered after all {NB} AXI beats have been captured into\n"
            f"                // feature_reg[{NLP-1}:0] ({NLP} bits total, {FS} bits per window).\n"
            f"                // Exits after exactly {NFS}×{NCS} = {NT} clock cycles.\n"
            f"                //\n"
            f"                // Register state initialised once on entry (in the reset / S_DONE\n"
            f"                // clear path and in S_RECV transition):\n"
            f"                //\n"
            f"                //   clause_pass       [{CT-1}:0]  = all 1   (AND identity)\n"
            f"                //   clause_has_actions [{CT-1}:0]  = all 0   (OR  identity)\n"
            f"                //\n"
            f"                // The two nested loops below are SEQUENTIAL in hardware\n"
            f"                // (one tile per clock cycle).  The parallelism comes from\n"
            f"                // the {CS} clause_eval instances that all fire on the SAME cycle.\n"
            f"                //\n"
            f"                // Pseudocode (sequential loops / parallel instances labelled):\n"
            f"                //\n"
            f"                //   for feat_cnt in 0..{NFS-1}:              // Note A — outer, SEQUENTIAL\n"
            f"                //     for clause_slice_cnt in 0..{NCS-1}:   // Note B — inner, SEQUENTIAL\n"
            f"                //\n"
            f"                //       ══════════ ONE CLOCK CYCLE ══════════\n"
            f"                //\n"
            f"                //       // 1. ROM lookup — one address per cycle\n"
            f"                //       tile_addr  ({int(math.ceil(math.log2(NT+1)))}-bit) = feat_cnt × {NCS} + clause_slice_cnt\n"
            f"                //       tile ({TW}-bit) = tile_rom[tile_addr]\n"
            f"                //\n"
            f"                //       // 2. Feature window for this row\n"
            f"                //       feat_pos     ({FS}-bit) = feature_reg[feat_cnt×{FS} +: {FS}]\n"
            f"                //       partial_lits ({2*FS}-bit) = {{~feat_pos, feat_pos}}\n"
            f"                //                              bits  0..{FS-1} = positive literals (feat_pos)\n"
            f"                //                              bits {FS}..{2*FS-1} = negated  literals (~feat_pos)\n"
            f"                //\n"
            f"                //       // 3. Evaluate {CS} clauses in PARALLEL  — Note C\n"
            f"                //       for k in 0..{CS-1}:                  // PARALLEL: {CS} clause_eval instances\n"
            f"                //\n"
            f"                //         global_clause ({int(math.ceil(math.log2(CT+1)))}-bit)\n"
            f"                //                       = clause_slice_cnt × {CS} + k\n"
            f"                //         ta_mask ({2*FS}-bit) = tile[{2*FS}×k +: {2*FS}]\n"
            f"                //           bits  0..{FS-1} = Include bits for positive literals 0..{FS-1}\n"
            f"                //           bits {FS}..{2*FS-1} = Include bits for negated  literals 0..{FS-1}\n"
            f"                //\n"
            f"                //         // clause_eval internals — purely combinational  — Note D\n"
            f"                //         for l in 0..{2*FS-1}:              // PARALLEL: {2*FS}-input AND tree\n"
            f"                //           lit_ok[l] (1-bit) = partial_lits[l] | ~ta_mask[l]\n"
            f"                //                    // literal passes if it is 1, OR this clause excludes it\n"
            f"                //         tile_pass       (1-bit) = AND_reduce(lit_ok)   // all inclusions satisfied\n"
            f"                //         tile_has_actions(1-bit) = OR_reduce (ta_mask)  // clause is non-empty\n"
            f"                //\n"
            f"                //         // 4. Fold tile result into cross-row accumulators (registered)\n"
            f"                //         effective_pass (1-bit) = tile_pass | ~tile_has_actions\n"
            f"                //                    // empty tile (no Include actions) vacuously passes\n"
            f"                //         if feat_cnt == 0:   // first row: seed\n"
            f"                //           clause_pass[global_clause]        <= effective_pass\n"
            f"                //           clause_has_actions[global_clause] <= tile_has_actions\n"
            f"                //         else:              // subsequent rows: accumulate\n"
            f"                //           clause_pass[global_clause]        <= clause_pass[global_clause]\n"
            f"                //                                                & effective_pass\n"
            f"                //           clause_has_actions[global_clause] <= clause_has_actions[global_clause]\n"
            f"                //                                                | tile_has_actions\n"
            f"                //\n"
            f"                //       ══════════ CYCLE ENDS ══════════\n"
            f"                //\n"
            f"                // Note A — feat_cnt (outer loop, {int(math.ceil(math.log2(NFS+1)))}-bit reg):\n"
            f"                //   Advances only when the full inner sweep (all {NCS} clause slices)\n"
            f"                //   is complete.  Selects which {FS}-feature window of feature_reg\n"
            f"                //   drives partial_lits.  Combinatorially addresses the ROM row.\n"
            f"                //\n"
            f"                // Note B — clause_slice_cnt (inner loop, {CSCW}-bit reg):\n"
            f"                //   Increments every cycle.  Selects one column of the tile grid.\n"
            f"                //   Combinatorially addresses the ROM column.\n"
            f"                //   Together with feat_cnt it forms the full {int(math.ceil(math.log2(NT+1)))}-bit\n"
            f"                //   tile address with no multiplier: addr = feat_cnt×{NCS} + clause_slice_cnt.\n"
            f"                //\n"
            f"                // Note C — parallel clause_eval instances ({CS} total):\n"
            f"                //   Each instance is synthesised as independent combinational\n"
            f"                //   hardware sharing the same partial_lits bus.  Only ta_mask\n"
            f"                //   differs — it is sliced directly from tile_rom_data.\n"
            f"                //   All {CS} results (tile_pass[{CS-1}:0], tile_has_actions[{CS-1}:0])\n"
            f"                //   are available before the next clock edge.\n"
            f"                //\n"
            f"                // Note D — {2*FS}-input AND/OR trees inside clause_eval:\n"
            f"                //   No registers; all combinational.  The critical path is\n"
            f"                //   log2({2*FS}) ≈ {int(math.ceil(math.log2(2*FS)))} gate levels of AND.\n"
            f"                //   OR_reduce(ta_mask) detects empty clauses with the same depth.\n"
            f"                // ──────────────────────────────────────────────────────────────\n"
            f"                S_COMPUTE: begin\n"
            f"                    for (k_upd = 0; k_upd < CLAUSE_SLICE; k_upd = k_upd + 1) begin\n"
            f"                        clause_g = clause_slice_cnt * CLAUSE_SLICE + k_upd;\n"
            f"                        if (clause_g < N_CLAUSES_TOTAL) begin\n"
            f"                            if (feat_cnt == {{{FCW}{{1'b0}}}}) begin\n"
            f"                                // First feature slice: seed. Empty tile vacuously passes.\n"
            f"                                clause_pass[clause_g]        <= tile_pass[k_upd] | ~tile_has_actions[k_upd];\n"
            f"                                clause_has_actions[clause_g] <= tile_has_actions[k_upd];\n"
            f"                            end else begin\n"
            f"                                // Subsequent slices: AND pass / OR actions. Empty tile vacuously passes.\n"
            f"                                clause_pass[clause_g] <=\n"
            f"                                    clause_pass[clause_g] & (tile_pass[k_upd] | ~tile_has_actions[k_upd]);\n"
            f"                                clause_has_actions[clause_g] <=\n"
            f"                                    clause_has_actions[clause_g] | tile_has_actions[k_upd];\n"
            f"                            end\n"
            f"                        end\n"
            f"                    end\n"
            f"                    // Advance tile counters\n"
            f"                    if (clause_slice_cnt == LAST_CS) begin\n"
            f"                        clause_slice_cnt <= {{{CSCW}{{1'b0}}}};\n"
            f"                        if (feat_cnt == LAST_FEAT) begin\n"
            f"                            feat_cnt  <= {{{FCW}{{1'b0}}}};\n"
            f"                            score_cnt <= {{{CCW}{{1'b0}}}};\n"
            f"                            state     <= S_SCORE;\n"
            f"                        end else begin\n"
            f"                            feat_cnt <= feat_cnt + 1'b1;\n"
            f"                        end\n"
            f"                    end else begin\n"
            f"                        clause_slice_cnt <= clause_slice_cnt + 1'b1;\n"
            f"                    end\n"
            f"                end\n"
            f"\n"
            f"                // ─── S_SCORE phase of one inference ───────────────────────\n"
            f"                //\n"
            f"                // Entered immediately after S_COMPUTE finishes ({NT} cycles).\n"
            f"                // Exits after exactly {CT} clock cycles (one per clause).\n"
            f"                //\n"
            f"                // At this point clause_pass[0..{CT-1}] and\n"
            f"                // clause_has_actions[0..{CT-1}] hold the finalised S_COMPUTE\n"
            f"                // results.  score_cnt is the only counter; it was reset to 0\n"
            f"                // on the S_COMPUTE → S_SCORE transition.\n"
            f"                //\n"
            f"                // The score_acc sub-module is enabled (score_valid=1) throughout\n"
            f"                // S_SCORE.  It maintains {C} signed {SW}-bit scores clamped to\n"
            f"                // [-{T}, +{T}-1], one per class.  The argmax sub-module is\n"
            f"                // combinational and continuously selects the winning class from\n"
            f"                // scores_flat, but tm_accelerator only samples argmax_out in S_DONE.\n"
            f"                //\n"
            f"                // Clause ordering: clauses are laid out as\n"
            f"                //   [class 0 positive | class 0 negative | class 1 positive | ...]\n"
            f"                //   i.e. {K} clauses per class, first {K//2} are positive-polarity,\n"
            f"                //   last {K//2} are negative-polarity.\n"
            f"                //   score_class and score_is_pos are decoded from score_cnt\n"
            f"                //   via case statements (no division hardware needed).\n"
            f"                //\n"
            f"                // Pseudocode (one iteration = one clock cycle):\n"
            f"                //\n"
            f"                //   for score_cnt in 0..{CT-1}:             // SEQUENTIAL: {CT} cycles\n"
            f"                //\n"
            f"                //     ══════════ ONE CLOCK CYCLE ══════════\n"
            f"                //\n"
            f"                //     // 1. Decode which class and polarity this clause belongs to\n"
            f"                //     score_class  ({CW}-bit) = score_cnt / {K}     // class index 0..{C-1}\n"
            f"                //     score_is_pos  (1-bit) = (score_cnt % {K}) < {K//2}\n"
            f"                //                             // first {K//2} of each {K}: positive polarity\n"
            f"                //\n"
            f"                //     // 2. Determine if this clause fired\n"
            f"                //     score_active  (1-bit) = clause_pass[score_cnt]\n"
            f"                //                           & clause_has_actions[score_cnt]\n"
            f"                //                             // fires only if it passed AND is non-empty\n"
            f"                //\n"
            f"                //     // 3. Update the score for score_class  — inside score_acc\n"
            f"                //     if score_active:\n"
            f"                //       if score_is_pos:  scores[score_class] += 1  (clamped at +{T}-1)\n"
            f"                //       else:             scores[score_class] -= 1  (clamped at -{T})\n"
            f"                //     // (no update if clause did not fire)\n"
            f"                //\n"
            f"                //     ══════════ CYCLE ENDS ══════════\n"
            f"                //\n"
            f"                // After all {CT} iterations:\n"
            f"                //   scores[0..{C-1}] each hold a signed {SW}-bit integer in\n"
            f"                //   [{-T}, {T-1}].  scores_flat ({SW*C}-bit) packs them as\n"
            f"                //     scores_flat[j×{SW} +: {SW}] = scores[j]  (two's complement)\n"
            f"                //\n"
            f"                //   argmax_out ({CW}-bit) = index of max(scores[0..{C-1}])\n"
            f"                //                           ties broken by lower index\n"
            f"                //   (argmax is combinational; it reflects scores_flat continuously\n"
            f"                //    but is only latched into m_axis_tdata in S_DONE)\n"
            f"                // ──────────────────────────────────────────────────────────────\n"
            f"                // score_valid = (state==S_SCORE); score_acc sees one clause/cycle.\n"
            f"                S_SCORE: begin\n"
            f"                    score_cnt <= score_cnt + 1'b1;\n"
            f"                    if (score_cnt == LAST_CLAUSE) begin\n"
            f"                        state <= S_DONE;\n"
            f"                    end\n"
            f"                end\n"
            f"\n"
            f"                // ─── S_DONE phase of one inference ────────────────────────\n"
            f"                //\n"
            f"                // Entered on the cycle after score_cnt reaches {CT-1}.\n"
            f"                // Duration: 1 cycle minimum; held indefinitely until the\n"
            f"                // downstream consumer asserts m_axis_tready (back-pressure).\n"
            f"                //\n"
            f"                // On entry, argmax_out is already valid: the argmax sub-module\n"
            f"                // is combinational and has been tracking scores_flat throughout\n"
            f"                // S_SCORE.  No extra latency is needed to compute the result.\n"
            f"                //\n"
            f"                // Outputs driven this state:\n"
            f"                //   m_axis_tvalid  (1-bit)  = 1  (held until handshake)\n"
            f"                //   m_axis_tdata  ({AW}-bit)  = {{28'b0, argmax_out}}\n"
            f"                //                              bits [{CW-1}:0]   = winning class index\n"
            f"                //                              bits [{AW-1}:{CW}] = zero-padded\n"
            f"                //   m_axis_tlast   (1-bit)  = frame_tlast\n"
            f"                //                              1 if this was the last inference\n"
            f"                //                              in the AXI-Stream packet (batch),\n"
            f"                //                              0 for all intermediate inferences\n"
            f"                //\n"
            f"                // Handshake and cleanup (on the cycle m_axis_tready is sampled high):\n"
            f"                //   score_clear  (1-bit) <= 1   — pulses score_acc to zero all\n"
            f"                //                              {C} scores before the next inference\n"
            f"                //   busy         (1-bit) <= 0\n"
            f"                //   state               <= S_IDLE\n"
            f"                //\n"
            f"                // score_clear is asserted for exactly one cycle (the handshake\n"
            f"                // cycle).  score_acc clears scores[] on the following clock edge,\n"
            f"                // which is the first cycle of S_IDLE — guaranteed to complete\n"
            f"                // before the next S_SCORE begins.\n"
            f"                //\n"
            f"                // Pseudocode:\n"
            f"                //\n"
            f"                //   // Entry: latch result (registered, stable every cycle in S_DONE)\n"
            f"                //   m_axis_tvalid <= 1\n"
            f"                //   m_axis_tdata  <= {{({AW-CW})'b0, argmax_out ({CW}-bit)}}\n"
            f"                //   m_axis_tlast  <= frame_tlast (1-bit)\n"
            f"                //\n"
            f"                //   // Wait for downstream to accept\n"
            f"                //   while not m_axis_tready:   // back-pressure: hold outputs, do nothing\n"
            f"                //     ══════════ ONE CLOCK CYCLE ══════════\n"
            f"                //\n"
            f"                //   // Handshake cycle (m_axis_tready = 1)\n"
            f"                //   score_clear <= 1            // tells score_acc to zero scores[]\n"
            f"                //   busy        <= 0\n"
            f"                //   state       <= S_IDLE\n"
            f"                //   ══════════ FINAL CYCLE OF INFERENCE ══════════\n"
            f"                // ──────────────────────────────────────────────────────────────\n"
            f"                // m_axis_tvalid/tdata/tlast are held until m_axis_tready.\n"
            f"                S_DONE: begin\n"
            f"                    m_axis_tvalid <= 1'b1;\n"
            f"                    m_axis_tdata  <= {{{AW-CW}'b0, argmax_out}};\n"
            f"                    m_axis_tlast  <= frame_tlast;\n"
            f"                    if (m_axis_tready) begin\n"
            f"                        score_clear <= 1'b1;\n"
            f"                        busy        <= 1'b0;\n"
            f"                        state       <= S_IDLE;\n"
            f"                    end\n"
            f"                end\n"
            f"\n"
            f"                default: state <= S_IDLE;\n"
            f"            endcase\n"
            f"        end\n"
            f"    end\n"
            f"\n"
            f"endmodule\n"
        )

    # ------------------------------------------------------------------
    # Testbench generators
    # ------------------------------------------------------------------

    def _gen_tb_axis_fifo(self) -> str:
        DW = self.axis_dw
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_axis_fifo — unit test: push 8 beats, verify order + tlast placement
        module tb_axis_fifo;
            parameter DATA_WIDTH = {DW};
            parameter DEPTH      = 16;

            reg                   clk, rst_n;
            reg                   s_tvalid;
            wire                  s_tready;
            reg  [DATA_WIDTH-1:0] s_tdata;
            reg                   s_tlast;
            wire                  m_tvalid;
            reg                   m_tready;
            wire [DATA_WIDTH-1:0] m_tdata;
            wire                  m_tlast;

            axis_fifo #(.DATA_WIDTH(DATA_WIDTH), .DEPTH(DEPTH)) dut (
                .clk(clk), .rst_n(rst_n),
                .s_tvalid(s_tvalid), .s_tready(s_tready),
                .s_tdata(s_tdata),   .s_tlast(s_tlast),
                .m_tvalid(m_tvalid), .m_tready(m_tready),
                .m_tdata(m_tdata),   .m_tlast(m_tlast)
            );

            initial clk = 0;
            always #5 clk = ~clk;

            integer fail_cnt, i;

            initial begin
                $dumpfile("tb_axis_fifo.vcd");
                $dumpvars(0, tb_axis_fifo);
                rst_n = 0; s_tvalid = 0; s_tdata = 0; s_tlast = 0; m_tready = 0;
                repeat(4) @(posedge clk);
                rst_n = 1; @(posedge clk);

                for (i = 0; i < 8; i = i + 1) begin
                    s_tvalid = 1;
                    s_tdata  = i + 32'hA0;
                    s_tlast  = (i == 7) ? 1'b1 : 1'b0;
                    @(posedge clk);
                    while (!s_tready) @(posedge clk);
                end
                s_tvalid = 0; s_tlast = 0;

                fail_cnt = 0;
                m_tready = 1;
                for (i = 0; i < 8; i = i + 1) begin
                    while (!m_tvalid) @(posedge clk);
                    if (m_tdata !== (i + 32'hA0)) begin
                        $display("FAIL beat %0d: exp=%0h got=%0h", i, i+32'hA0, m_tdata);
                        fail_cnt = fail_cnt + 1;
                    end
                    if (m_tlast !== (i == 7 ? 1'b1 : 1'b0)) begin
                        $display("FAIL beat %0d: tlast exp=%0b got=%0b", i, (i==7), m_tlast);
                        fail_cnt = fail_cnt + 1;
                    end
                    @(posedge clk);
                end

                if (fail_cnt == 0) $display("tb_axis_fifo: ALL PASSED");
                else               $display("tb_axis_fifo: FAILED (%0d errors)", fail_cnt);
                $finish;
            end
        endmodule
        """)

    def _gen_tb_clause_eval(self) -> str:
        L = 2 * self.feat_slice   # clause_eval instantiated with 2*FEAT_SLICE literals
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_clause_eval — unit test: known-answer evaluation with tile literal width
        // N_LITERALS = 2*FEAT_SLICE = {L}
        module tb_clause_eval;
            parameter N_LITERALS = {L};

            reg  [N_LITERALS-1:0] literals;
            reg  [N_LITERALS-1:0] ta_action_mask;
            wire                  active;

            clause_eval #(.N_LITERALS(N_LITERALS)) dut (
                .literals(literals), .ta_action_mask(ta_action_mask), .active(active)
            );

            integer fail_cnt;

            task check;
                input [N_LITERALS-1:0] lit;
                input [N_LITERALS-1:0] inc;
                input                  exp;
                input [63:0]           id;
                begin
                    literals       = lit;
                    ta_action_mask = inc;
                    #1;
                    if (active !== exp) begin
                        $display("FAIL test%0d: lit=%b inc=%b exp=%b got=%b", id, lit, inc, exp, active);
                        fail_cnt = fail_cnt + 1;
                    end
                end
            endtask

            initial begin
                $dumpfile("tb_clause_eval.vcd");
                $dumpvars(0, tb_clause_eval);
                fail_cnt = 0;

                check({L}'b0,             {L}'b0,             1'b0, 0); // empty → inactive
                check({L}'b1,             {L}'b1,             1'b1, 1); // single inc, lit=1 → active
                check({L}'b0,             {L}'b1,             1'b0, 2); // single inc, lit=0 → inactive
                check({{{L}{{1'b1}}}},    {{{L}{{1'b1}}}},   1'b1, 3); // all inc, all lit=1 → active
                check({{{L}{{1'b1}}}} & ~{L}'b1, {{{L}{{1'b1}}}}, 1'b0, 4); // one miss → inactive
                // Non-included bits are don't-care: lit=0 but inc=0 should still pass
                check({{{L}{{1'b0}}}},    {{{L}{{1'b0}}}},   1'b0, 5); // empty stays inactive

                if (fail_cnt == 0) $display("tb_clause_eval: ALL PASSED");
                else               $display("tb_clause_eval: FAILED (%0d errors)", fail_cnt);
                $finish;
            end
        endmodule
        """)

    def _gen_tb_score_acc(self) -> str:
        C  = self.n_classes
        T  = self.threshold
        SW = self.score_width
        CW = max(1, int(math.ceil(math.log2(max(C, 2)))))
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_score_acc — unit test: accumulation, clamping, and clear
        module tb_score_acc;
            parameter N_CLASSES   = {C};
            parameter THRESHOLD   = {T};
            parameter SCORE_WIDTH = {SW};

            reg                              clk, rst_n;
            reg                              clear, valid, polarity, active;
            reg  [{CW}-1:0]                  cls;
            wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat;

            score_acc #(
                .N_CLASSES(N_CLASSES), .THRESHOLD(THRESHOLD), .SCORE_WIDTH(SCORE_WIDTH)
            ) dut (
                .clk(clk), .rst_n(rst_n), .clear(clear),
                .valid(valid), .cls(cls), .polarity(polarity),
                .active(active), .scores_flat(scores_flat)
            );

            initial clk = 0;
            always #5 clk = ~clk;

            function signed [SCORE_WIDTH-1:0] score_of;
                input integer c;
                begin score_of = scores_flat[c*SCORE_WIDTH +: SCORE_WIDTH]; end
            endfunction

            integer fail_cnt, i;

            initial begin
                $dumpfile("tb_score_acc.vcd");
                $dumpvars(0, tb_score_acc);
                fail_cnt = 0;
                clear = 0; valid = 0; polarity = 0; active = 0; cls = 0;
                rst_n = 0; repeat(4) @(posedge clk);
                rst_n = 1; @(posedge clk);

                // Accumulate THRESHOLD+2 positive votes; score must clamp at +T
                for (i = 0; i < THRESHOLD + 2; i = i + 1) begin
                    valid = 1; cls = 0; polarity = 1; active = 1; @(posedge clk);
                end
                valid = 0; @(posedge clk);
                if ($signed(score_of(0)) !== $signed({SW}'d{T})) begin
                    $display("FAIL clamp+: exp=%0d got=%0d", {T}, $signed(score_of(0)));
                    fail_cnt = fail_cnt + 1;
                end

                // Accumulate THRESHOLD+2 negative votes; score must clamp at -T
                for (i = 0; i < THRESHOLD + 2; i = i + 1) begin
                    valid = 1; cls = 0; polarity = 0; active = 1; @(posedge clk);
                end
                valid = 0; @(posedge clk);
                if ($signed(score_of(0)) !== -$signed({SW}'d{T})) begin
                    $display("FAIL clamp-: exp=%0d got=%0d", -{T}, $signed(score_of(0)));
                    fail_cnt = fail_cnt + 1;
                end

                // Inactive clause must not change score
                valid = 1; cls = 0; polarity = 1; active = 0; @(posedge clk);
                valid = 0; @(posedge clk);
                if ($signed(score_of(0)) !== -$signed({SW}'d{T})) begin
                    $display("FAIL inactive: score changed unexpectedly");
                    fail_cnt = fail_cnt + 1;
                end

                // Clear resets to 0
                clear = 1; @(posedge clk); clear = 0; @(posedge clk);
                if ($signed(score_of(0)) !== 0) begin
                    $display("FAIL clear: exp=0 got=%0d", $signed(score_of(0)));
                    fail_cnt = fail_cnt + 1;
                end

                if (fail_cnt == 0) $display("tb_score_acc: ALL PASSED");
                else               $display("tb_score_acc: FAILED (%0d errors)", fail_cnt);
                $finish;
            end
        endmodule
        """)

    def _gen_tb_argmax(self) -> str:
        C  = self.n_classes
        SW = self.score_width
        CW = self.class_width
        cases = ""
        for winner in range(C):
            scores_hex = 0
            for c in range(C):
                val = 2 if c == winner else 0
                scores_hex |= (val & ((1 << SW) - 1)) << (c * SW)
            hex_w = (SW * C + 3) // 4
            cases += (
                f"                scores_flat = {SW * C}'h{scores_hex:0{hex_w}X}; #1;\n"
                f"                if (pred_class !== {winner}) begin\n"
                f"                    $display(\"FAIL case{winner}: exp={winner} got=%0d\", pred_class);\n"
                f"                    fail_cnt = fail_cnt + 1;\n"
                f"                end\n"
            )
        return textwrap.dedent(f"""\
        `timescale 1ns/1ps
        // tb_argmax — unit test: one winner per class position
        module tb_argmax;
            parameter N_CLASSES   = {C};
            parameter SCORE_WIDTH = {SW};
            parameter CLASS_WIDTH = {CW};

            reg  [SCORE_WIDTH*N_CLASSES-1:0] scores_flat;
            wire [CLASS_WIDTH-1:0]            pred_class;

            argmax #(.N_CLASSES(N_CLASSES), .SCORE_WIDTH(SCORE_WIDTH),
                     .CLASS_WIDTH(CLASS_WIDTH)) dut (
                .scores_flat(scores_flat), .pred_class(pred_class)
            );

            integer fail_cnt;

            initial begin
                $dumpfile("tb_argmax.vcd");
                $dumpvars(0, tb_argmax);
                fail_cnt = 0;
        {cases}
                if (fail_cnt == 0) $display("tb_argmax: ALL PASSED");
                else               $display("tb_argmax: FAILED (%0d errors)", fail_cnt);
                $finish;
            end
        endmodule
        """)

    def _gen_tb_system(self) -> str:
        N   = self.n_features
        C   = self.n_classes
        K   = self.n_clauses_pc
        CT  = self.n_clauses_total
        T   = self.threshold
        NB  = self.n_beats
        AW  = self.axis_dw
        SW  = self.score_width
        CW  = self.class_width
        FD  = self.fifo_depth
        FS  = self.feat_slice
        CS  = self.clause_slice
        NFS = self.n_feat_slices
        NCS = self.n_clause_slices
        NLP = self.n_feat_padded
        TW  = self.tile_width
        NT  = self.n_tiles
        CCW = self.clause_cnt_w
        BCW = self.beat_cnt_w
        FCW = self.feat_cnt_w
        CSCW = self.cs_cnt_w
        latency = NFS * NCS + CT + 10

        tvecs   = self._get_test_vectors()
        n_tests = len(tvecs)

        tv_init = ""
        if tvecs:
            for idx, tv in enumerate(tvecs):
                beats = self._pack_beats(tv.input)
                for b, w in enumerate(beats):
                    is_final_beat = (idx == n_tests - 1 and b == NB - 1)
                    tv_init += (
                        f"        tv_data[{idx * NB + b}] = {AW}'h{w:0{(AW + 3) // 4}X};\n"
                        f"        tv_last[{idx * NB + b}] = 1'b{'1' if is_final_beat else '0'};\n"
                    )
                tv_init += f"        tv_exp[{idx}] = {CW}'d{tv.expected_class};\n"
        else:
            tv_init = "        // No embedded test vectors — regenerate after training.\n"

        return (
            f"`timescale 1ns/1ps\n"
            f"// tb_system — full end-to-end test against TMIR embedded test vectors\n"
            f"// Tiled: feat_slice={FS}  clause_slice={CS}\n"
            f"// Compute cycles per inference: {NFS}×{NCS}={NFS*NCS} (tile) + {CT} (score)\n"
            f"module tb_system;\n"
            f"\n"
            f"    parameter N_FEATURES      = {N};\n"
            f"    parameter N_CLASSES       = {C};\n"
            f"    parameter N_CLAUSES_PC    = {K};\n"
            f"    parameter N_CLAUSES_TOTAL = {CT};\n"
            f"    parameter THRESHOLD       = {T};\n"
            f"    parameter N_BEATS         = {NB};\n"
            f"    parameter FEAT_SLICE      = {FS};\n"
            f"    parameter CLAUSE_SLICE    = {CS};\n"
            f"    parameter N_FEAT_SLICES   = {NFS};\n"
            f"    parameter N_CLAUSE_SLICES = {NCS};\n"
            f"    parameter N_FEAT_PADDED   = {NLP};\n"
            f"    parameter TILE_WIDTH      = {TW};\n"
            f"    parameter N_TILES         = {NT};\n"
            f"    parameter AXIS_DATA_WIDTH = {AW};\n"
            f"    parameter SCORE_WIDTH     = {SW};\n"
            f"    parameter CLASS_WIDTH     = {CW};\n"
            f"    parameter FIFO_DEPTH      = {FD};\n"
            f"    parameter N_TESTS         = {n_tests};\n"
            f"    parameter MAX_TIMEOUT     = {latency + 100};\n"
            f"\n"
            f"    reg                        clk, rst_n;\n"
            f"    reg                        s_tvalid;\n"
            f"    wire                       s_tready;\n"
            f"    reg  [AXIS_DATA_WIDTH-1:0] s_tdata;\n"
            f"    reg                        s_tlast;\n"
            f"    wire                       m_tvalid;\n"
            f"    reg                        m_tready;\n"
            f"    wire [AXIS_DATA_WIDTH-1:0] m_tdata;\n"
            f"    wire                       m_tlast;\n"
            f"    wire                       busy;\n"
            f"\n"
            f"    tm_accelerator #(\n"
            f"        .N_FEATURES(N_FEATURES), .N_CLASSES(N_CLASSES),\n"
            f"        .N_CLAUSES_PC(N_CLAUSES_PC), .N_CLAUSES_TOTAL(N_CLAUSES_TOTAL),\n"
            f"        .THRESHOLD(THRESHOLD), .N_BEATS(N_BEATS),\n"
            f"        .FEAT_SLICE(FEAT_SLICE), .CLAUSE_SLICE(CLAUSE_SLICE),\n"
            f"        .N_FEAT_SLICES(N_FEAT_SLICES), .N_CLAUSE_SLICES(N_CLAUSE_SLICES),\n"
            f"        .N_FEAT_PADDED(N_FEAT_PADDED), .TILE_WIDTH(TILE_WIDTH),\n"
            f"        .N_TILES(N_TILES), .AXIS_DATA_WIDTH(AXIS_DATA_WIDTH),\n"
            f"        .SCORE_WIDTH(SCORE_WIDTH), .CLASS_WIDTH(CLASS_WIDTH),\n"
            f"        .FIFO_DEPTH(FIFO_DEPTH)\n"
            f"    ) dut (\n"
            f"        .clk(clk), .rst_n(rst_n),\n"
            f"        .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),\n"
            f"        .s_axis_tdata(s_tdata),   .s_axis_tlast(s_tlast),\n"
            f"        .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),\n"
            f"        .m_axis_tdata(m_tdata),   .m_axis_tlast(m_tlast),  .busy(busy)\n"
            f"    );\n"
            f"\n"
            f"    initial clk = 0;\n"
            f"    always #5 clk = ~clk;\n"
            f"\n"
            f"    reg [AXIS_DATA_WIDTH-1:0] tv_data [0:N_TESTS*N_BEATS-1];\n"
            f"    reg                       tv_last [0:N_TESTS*N_BEATS-1];\n"
            f"    reg [CLASS_WIDTH-1:0]     tv_exp  [0:N_TESTS-1];\n"
            f"\n"
            f"    integer t, g, fail_cnt, timeout;\n"
            f"\n"
            f"    initial begin\n"
            f"        $dumpfile(\"tb_system.vcd\");\n"
            f"        $dumpvars(0, tb_system);\n"
            f"{tv_init}"
            f"\n"
            f"        rst_n = 0; s_tvalid = 0; s_tdata = 0; s_tlast = 0; m_tready = 1;\n"
            f"        repeat(4) @(posedge clk);\n"
            f"        rst_n = 1;\n"
            f"        repeat(2) @(posedge clk);\n"
            f"\n"
            f"        fail_cnt = 0;\n"
            f"\n"
            f"        if (N_TESTS == 0) begin\n"
            f"            $display(\"tb_system: no test vectors embedded — skipping\");\n"
            f"            $finish;\n"
            f"        end\n"
            f"\n"
            f"        // Streaming: sender and receiver run in parallel.\n"
            f"        // Sender streams all N_TESTS*N_BEATS beats back-to-back, stalling only\n"
            f"        // when the input FIFO is full (s_tready=0 back-pressure).\n"
            f"        // Receiver collects N_TESTS outputs independently.\n"
            f"        fork\n"
            f"            // SENDER\n"
            f"            begin\n"
            f"                for (g = 0; g < N_TESTS * N_BEATS; g = g + 1) begin\n"
            f"                    s_tvalid = 1'b1;\n"
            f"                    s_tdata  = tv_data[g];\n"
            f"                    s_tlast  = tv_last[g];\n"
            f"                    @(posedge clk);\n"
            f"                    while (!s_tready) @(posedge clk);\n"
            f"                end\n"
            f"                s_tvalid = 1'b0; s_tlast = 1'b0;\n"
            f"            end\n"
            f"            // RECEIVER\n"
            f"            begin\n"
            f"                for (t = 0; t < N_TESTS; t = t + 1) begin\n"
            f"                    timeout = 0;\n"
            f"                    while (!m_tvalid) begin\n"
            f"                        @(posedge clk);\n"
            f"                        timeout = timeout + 1;\n"
            f"                        if (timeout >= MAX_TIMEOUT) begin\n"
            f"                            $display(\"TIMEOUT test %0d\", t);\n"
            f"                            fail_cnt = fail_cnt + 1 + (N_TESTS - t - 1);\n"
            f"                            $finish;\n"
            f"                        end\n"
            f"                    end\n"
            f"                    if (m_tdata[CLASS_WIDTH-1:0] !== tv_exp[t]) begin\n"
            f"                        $display(\"FAIL test %0d: expected=%0d got=%0d\",\n"
            f"                                 t, tv_exp[t], m_tdata[CLASS_WIDTH-1:0]);\n"
            f"                        fail_cnt = fail_cnt + 1;\n"
            f"                    end else\n"
            f"                        $display(\"PASS test %0d: class=%0d\", t, m_tdata[CLASS_WIDTH-1:0]);\n"
            f"                    if (t == N_TESTS - 1 && !m_tlast) begin\n"
            f"                        $display(\"FAIL: m_axis_tlast not asserted on final output\");\n"
            f"                        fail_cnt = fail_cnt + 1;\n"
            f"                    end\n"
            f"                    if (t != N_TESTS - 1 && m_tlast) begin\n"
            f"                        $display(\"FAIL test %0d: m_axis_tlast asserted mid-batch\", t);\n"
            f"                        fail_cnt = fail_cnt + 1;\n"
            f"                    end\n"
            f"                    @(posedge clk);  // consume output beat\n"
            f"                end\n"
            f"            end\n"
            f"        join\n"
            f"\n"
            f"        if (fail_cnt == 0)\n"
            f"            $display(\"tb_system: ALL %0d TESTS PASSED\", N_TESTS);\n"
            f"        else\n"
            f"            $display(\"tb_system: FAILED %0d/%0d\", fail_cnt, N_TESTS);\n"
            f"        $finish;\n"
            f"    end\n"
            f"endmodule\n"
        )

    # ------------------------------------------------------------------
    # Simulation script generators
    # ------------------------------------------------------------------

    def _gen_waves_sh(self) -> str:
        return textwrap.dedent("""\
        #!/usr/bin/env bash
        # Usage: bash waves.sh [tb_name]
        # Opens GTKWave with the matching .gtkw for the given testbench.
        # Prefers FST waveforms (Verilator), falls back to VCD (iverilog).
        set -euo pipefail
        TB="${1:-tb_system}"
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

        GTKW="$SCRIPT_DIR/$TB.gtkw"
        if [ ! -f "$GTKW" ]; then
            echo "No .gtkw file for $TB at $GTKW"
            exit 1
        fi

        # Prefer FST (Verilator in obj_dir or verilator/) → VCD (iverilog)
        WF="$SCRIPT_DIR/verilator/$TB.fst"
        if [ ! -f "$WF" ]; then
            WF="$SCRIPT_DIR/$TB.vcd"
        fi
        if [ ! -f "$WF" ]; then
            echo "No waveform file found for $TB. Run simulation first:"
            echo "  bash run_iverilog.sh         (produces .vcd)"
            echo "  make -C verilator run         (produces .fst, requires Verilator)"
            exit 1
        fi

        echo "Opening GTKWave: $WF with $GTKW"
        gtkwave "$WF" "$GTKW" &
        """)

    def _gen_verilator_makefile(self) -> str:
        return textwrap.dedent("""\
        # Auto-generated by matador rtl generate — Verilator build for tb_system
        #
        # Requires Verilator ≥5.0.  Override with:
        #   make VERILATOR=/path/to/verilator5 run
        #
        # Portable: SCRIPT_DIR is computed from the Makefile's own location at
        # make-time, so this directory can be copied or moved without editing paths.
        #
        # Portability note: Verilator's internal Vtm_accelerator.mk bakes in the
        # absolute path of tb_top.cpp from when Verilator was last invoked.  To
        # avoid stale paths from a previous build environment (e.g. a build done
        # on a different machine), obj_dir is always deleted before Verilator runs.
        # This guarantees a clean, self-consistent build regardless of where the
        # RTL directory was originally generated.
        #
        # VERILATOR_ROOT conflict: if VERILATOR_ROOT is set in the environment
        # (e.g. pointing to an older Verilator install), it can confuse a newer
        # binary.  unexport clears it for all child processes spawned from make.
        unexport VERILATOR_ROOT

        VERILATOR  ?= verilator
        SCRIPT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
        SRC_DIR    := $(SCRIPT_DIR)../../src
        OBJ_DIR    := $(SCRIPT_DIR)obj_dir
        BIN        := $(OBJ_DIR)/Vtm_accelerator

        # On macOS: if the default verilator is <5.0, try Homebrew's copy.
        # env -u VERILATOR_ROOT is used because $(shell) inherits make's environment
        # and an old VERILATOR_ROOT in the shell env will confuse newer Verilator binaries.
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

        # Fail fast with a readable message if Verilator is still too old.
        _VER_FINAL := $(shell env -u VERILATOR_ROOT $(VERILATOR) --version 2>/dev/null | awk '{print $$2}' | cut -d. -f1)
        ifeq ($(filter 5 6 7 8 9,$(_VER_FINAL)),)
          $(error Verilator ≥5.0 required (found: $(shell env -u VERILATOR_ROOT $(VERILATOR) --version 2>/dev/null | head -1 || echo not found)). Set VERILATOR=/path/to/verilator5)
        endif

        # Pass Verilator a clean environment: strip VERILATOR_ROOT so it always
        # uses the include/share paths from its own install tree, not a stale one.
        VERILATOR := env -u VERILATOR_ROOT $(VERILATOR)

        VSRCS := \\
            $(SRC_DIR)/axis_fifo.v     \\
            $(SRC_DIR)/clause_eval.v   \\
            $(SRC_DIR)/score_acc.v     \\
            $(SRC_DIR)/argmax.v        \\
            $(SRC_DIR)/tm_accelerator.v

        # --build: Verilator compiles and links in one invocation.
        # --CFLAGS "-std=c++17": required on macOS (Apple Clang defaults to C++98).
        VFLAGS := \\
            --cc --exe --build --trace-fst \\
            --language 1800-2005           \\
            --top-module tm_accelerator    \\
            --Mdir $(OBJ_DIR)              \\
            --CFLAGS "-std=c++17"          \\
            -Wall --Wno-TIMESCALEMOD --Wno-WIDTH --Wno-BLKSEQ

        .PHONY: all run clean

        all: $(BIN)

        # Delete obj_dir before invoking Verilator so the generated
        # Vtm_accelerator.mk always uses paths from the current environment.
        $(BIN): $(VSRCS) $(SCRIPT_DIR)tb_top.cpp
        \trm -rf $(OBJ_DIR)
        \t$(VERILATOR) $(VFLAGS) $(VSRCS) $(SCRIPT_DIR)tb_top.cpp

        # run always rebuilds cleanly to prevent stale-binary issues across
        # environments (e.g. macOS build running on Linux container).
        run: clean all
        \t@cd $(SCRIPT_DIR) && $(BIN)

        clean:
        \trm -rf $(OBJ_DIR)
        """)

    def _gen_verilator_tb_cpp(self) -> str:
        """C++ testbench for Verilator — mirrors tb_system.v logic with FST waveform dump."""
        N   = self.n_features
        C   = self.n_classes
        NB  = self.n_beats
        AW  = self.axis_dw
        CW  = self.class_width
        # Compute max latency: tile cycles + score cycles + slack
        latency = self.n_feat_slices * self.n_clause_slices + self.n_clauses_total + 50

        tvecs   = self._get_test_vectors()
        n_tests = len(tvecs)

        # Embed beat data, tlast flags, expected classes
        # Arrays need at least 1 element even if n_tests == 0
        arr_size_tv  = max(n_tests * NB, 1)
        arr_size_exp = max(n_tests, 1)

        beat_lines, last_lines, exp_lines = [], [], []
        for idx, tv in enumerate(tvecs):
            beats = self._pack_beats(tv.input)
            for b, word in enumerate(beats):
                beat_lines.append(f"    {word}ULL,  // test {idx} beat {b}")
                is_final_beat = (idx == n_tests - 1 and b == NB - 1)
                last_lines.append(f"    {1 if is_final_beat else 0},  // test {idx} beat {b} tlast")
            exp_lines.append(f"    {tv.expected_class},  // test {idx}")

        beat_data = "\n".join(beat_lines) if beat_lines else "    0"
        last_data = "\n".join(last_lines) if last_lines else "    0"
        exp_data  = "\n".join(exp_lines)  if exp_lines  else "    0"

        tdata_t = "uint64_t" if AW == 64 else "uint32_t"

        return textwrap.dedent(f"""\
        // tb_top.cpp — Verilator C++ testbench for tm_accelerator
        // Auto-generated by matador rtl generate. Do not edit.
        // Produces: tb_system.fst (FST waveform, requires Verilator --trace-fst)

        #include "Vtm_accelerator.h"
        #include "verilated.h"
        #include "verilated_fst_c.h"

        #include <cstdint>
        #include <cstdio>
        #include <cstdlib>

        // ── Embedded parameters ───────────────────────────────────────────────────────
        static const int N_FEATURES  = {N};
        static const int N_CLASSES   = {C};
        static const int N_BEATS     = {NB};
        static const int CLASS_WIDTH = {CW};
        static const int N_TESTS     = {n_tests};
        static const int MAX_TIMEOUT = {latency + 200};

        // ── Embedded test vectors ─────────────────────────────────────────────────────
        static const uint64_t tv_data[{arr_size_tv}] = {{
        {beat_data}
        }};
        static const int tv_last[{arr_size_tv}] = {{
        {last_data}
        }};
        static const int tv_exp[{arr_size_exp}] = {{
        {exp_data}
        }};

        // ── Simulation state ──────────────────────────────────────────────────────────
        static vluint64_t sim_time = 0;
        static Vtm_accelerator *dut = nullptr;
        static VerilatedFstC   *tfp = nullptr;

        static void tick() {{
            dut->clk = 0;
            dut->eval();
            if (tfp) tfp->dump(sim_time++);
            dut->clk = 1;
            dut->eval();
            if (tfp) tfp->dump(sim_time++);
        }}

        int main(int argc, char **argv) {{
            Verilated::commandArgs(argc, argv);
            Verilated::traceEverOn(true);

            dut = new Vtm_accelerator;
            tfp = new VerilatedFstC;
            dut->trace(tfp, 99);
            tfp->open("tb_system.fst");

            // Reset
            dut->clk = 0;
            dut->rst_n = 0;
            dut->s_axis_tvalid = 0;
            dut->s_axis_tdata  = 0;
            dut->s_axis_tlast  = 0;
            dut->m_axis_tready = 1;  // always ready to accept output
            for (int i = 0; i < 8; i++) tick();
            dut->rst_n = 1;
            tick(); tick();

            int fail_cnt = 0;
            int pass_cnt = 0;

            if (N_TESTS == 0) {{
                printf("tb_system: no test vectors embedded — skipping\\n");
            }} else {{
                // Streaming: send all N_TESTS*N_BEATS beats back-to-back.
                // s_axis_tready (= ~FIFO full) provides the only back-pressure.
                // Outputs are captured as m_axis_tvalid fires, independently.
                int input_beat   = 0;   // next beat to send (0..N_TESTS*N_BEATS-1)
                int output_count = 0;   // outputs collected so far
                const int total_beats = N_TESTS * N_BEATS;
                const int max_cycles  = total_beats + N_TESTS * MAX_TIMEOUT;
                int cycle = 0;

                while (output_count < N_TESTS && cycle < max_cycles) {{
                    // Present next input beat (or deassert when all beats sent)
                    if (input_beat < total_beats) {{
                        dut->s_axis_tvalid = 1;
                        dut->s_axis_tdata  = ({tdata_t})tv_data[input_beat];
                        dut->s_axis_tlast  = tv_last[input_beat];
                    }} else {{
                        dut->s_axis_tvalid = 0;
                        dut->s_axis_tlast  = 0;
                    }}

                    // Sample combinatorial tready before the clock edge
                    dut->eval();
                    bool beat_accepted = dut->s_axis_tvalid && dut->s_axis_tready;

                    tick();
                    cycle++;

                    // Advance input pointer if FIFO accepted this beat
                    if (beat_accepted) input_beat++;

                    // Capture output if the DUT asserted m_axis_tvalid this cycle
                    // (m_axis_tready=1 always, so the handshake is complete)
                    if (dut->m_axis_tvalid) {{
                        int got = (int)(dut->m_axis_tdata) & ((1 << CLASS_WIDTH) - 1);
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

            if (fail_cnt == 0)
                printf("tb_system: ALL %d TESTS PASSED\\n", N_TESTS);
            else
                printf("tb_system: FAILED %d/%d\\n", fail_cnt, N_TESTS);

            if (tfp) {{ tfp->close(); delete tfp; }}
            dut->final();
            delete dut;
            return (fail_cnt > 0) ? 1 : 0;
        }}
        """)

    # ------------------------------------------------------------------
    # Documentation generator
    # ------------------------------------------------------------------

    def _gen_readme(self) -> str:
        """Generate RTL/README.md — standalone doc for all three audiences."""
        N    = self.n_features
        C    = self.n_classes
        K    = self.n_clauses_pc
        CT   = self.n_clauses_total
        T    = self.threshold
        NB   = self.n_beats
        AW   = self.axis_dw
        FS   = self.feat_slice
        CS   = self.clause_slice
        NFS  = self.n_feat_slices
        NCS  = self.n_clause_slices
        TW   = self.tile_width
        NT   = self.n_tiles
        SW   = self.score_width
        CW   = self.class_width
        FD   = self.fifo_depth
        CCW  = self.clause_cnt_w
        FCW  = self.feat_cnt_w
        CSCW = self.cs_cnt_w
        BCW  = self.beat_cnt_w
        latency = NFS * NCS + CT + 6
        half_k  = K // 2

        return f"""\
# TM Accelerator RTL

Synthesisable Verilog-2001 implementation of a trained Vanilla Tsetlin Machine
(TM) classifier. Extracted TA Include-action outputs from the trained model are
compiled into an on-chip ROM at generation time. The module streams in a binary
feature vector over AXI-Stream input and returns the predicted class as a
single-beat AXI-Stream output (`m_axis_tvalid`/`m_axis_tdata`/`m_axis_tlast`).

This directory is self-contained. It does not depend on the Python toolchain
that produced it; any Verilog simulator or synthesis tool can consume it
directly.

---

## Frozen model parameters

These constants are burned into the generated Verilog and cannot be changed
without regenerating the RTL.

| Parameter             | Value  | Meaning |
|-----------------------|--------|---------|
| `N_FEATURES`          | {N}    | Binary input features per inference |
| `N_CLASSES`           | {C}    | Output classes |
| `N_CLAUSES_PER_CLASS` | {K}    | Clauses per class |
| `N_CLAUSES_TOTAL`     | {CT}   | {C} × {K} |
| `THRESHOLD`           | {T}    | Score clamping bound |
| `FEAT_SLICE`          | {FS}   | Features evaluated per tile |
| `CLAUSE_SLICE`        | {CS}   | Clauses evaluated in parallel |
| `N_FEAT_SLICES`       | {NFS}  | Tile rows  (⌈{N}/{FS}⌉) |
| `N_CLAUSE_SLICES`     | {NCS}  | Tile columns (⌈{CT}/{CS}⌉) |
| `N_TILES`             | {NT}   | Total ROM entries ({NFS}×{NCS}) |
| `TILE_WIDTH`          | {TW}   | Bits per ROM entry ({CS}×2×{FS}) |
| `AXIS_DATA_WIDTH`     | {AW}   | AXI-Stream TDATA width (bits) |
| `N_BEATS`             | {NB}   | AXI beats per feature vector |
| `FIFO_DEPTH`          | {FD}   | Input FIFO capacity (beats) |
| `SCORE_WIDTH`         | {SW}   | Signed score register width |

---

## Quick start

```bash
# ── Verilator (preferred — FST waveforms) ────────────────────────────────────
cd sim/verilator
make clean && make               # compiles RTL + C++ testbench natively
./obj_dir/Vtm_accelerator        # runs all embedded test vectors

# If you copied this directory from a different machine (e.g. out of a Docker
# container), run `make clean` first — obj_dir/ may contain stale binaries
# compiled for a different OS/architecture.

# ── iverilog (fallback — VCD waveforms) ──────────────────────────────────────
bash sim/run_iverilog.sh

# ── GTKWave (open pre-configured signal layout) ───────────────────────────────
bash sim/waves.sh                # default: tb_system
bash sim/waves.sh tb_clause_eval
bash sim/waves.sh tb_score_acc
bash sim/waves.sh tb_argmax
bash sim/waves.sh tb_axis_fifo
```

---

## Tool setup

### Verilator (recommended, ≥ 5.0)

Verilator compiles RTL to C++ for fast cycle-accurate simulation and produces
FST waveforms. Version 5.0 or later is required for full SystemVerilog
compatibility flags used by the generated Makefile.

```bash
# Ubuntu / Debian
sudo apt-get install verilator          # may be older; check version
verilator --version                     # need ≥ 5.0

# Build from source (latest stable — recommended)
git clone https://github.com/verilator/verilator
cd verilator && git checkout stable
autoconf && ./configure && make -j$(nproc)
sudo make install

# macOS (Homebrew)
brew install verilator
```

### iverilog (fallback, ≥ 11)

iverilog is a lightweight Verilog-2001 simulator that produces VCD waveforms.
Required if Verilator is not available; also used by the unit testbenches.

```bash
# Ubuntu / Debian
sudo apt-get install iverilog            # typically installs ≥ 11

# macOS (Homebrew)
brew install icarus-verilog

# Verify
iverilog -V                              # need ≥ 11.0
```

### GTKWave (any recent version)

GTKWave opens `.vcd` (iverilog) and `.fst` (Verilator) waveform files with the
pre-configured `.gtkw` signal layouts shipped in `sim/`.

```bash
# Ubuntu / Debian
sudo apt-get install gtkwave

# macOS (Homebrew)
brew install --cask gtkwave

# Verify
gtkwave --version
```

### Vivado xsim (optional)

The generated `sim/run_xsim.sh` targets Xilinx Vivado's built-in simulator.
Any Vivado installation that includes xsim (2019.1 or later recommended) will
work. No additional setup is required beyond having Vivado on `PATH`.

---

## Directory structure

```
RTL/
├── README.md               ← this file
│
├── src/                    Synthesisable RTL (no timing constructs)
│   ├── axis_fifo.v         AXI-Stream synchronous FIFO
│   ├── clause_eval.v       Single-clause combinatorial evaluator
│   ├── score_acc.v         Per-class vote accumulator with clamping
│   ├── argmax.v            Combinatorial tournament argmax
│   └── tm_accelerator.v    Top module: FSM + ROM + all sub-modules
│
├── tb/                     Testbenches — one per src/ module
│   ├── tb_axis_fifo.v
│   ├── tb_clause_eval.v
│   ├── tb_score_acc.v
│   ├── tb_argmax.v
│   └── tb_system.v         Full-system integration test (iverilog)
│
└── sim/                    Simulation scripts and waveform layouts
    ├── run_iverilog.sh     Compile + run all tb/ under iverilog
    ├── run_xsim.sh         Vivado xsim wrapper
    ├── lint_verilator.sh   Lint-only pass (no simulation)
    ├── waves.sh            Open GTKWave with matching .gtkw layout
    ├── tb_system.gtkw      Pre-configured signal groups for tb_system
    ├── tb_clause_eval.gtkw
    ├── tb_score_acc.gtkw
    ├── tb_argmax.gtkw
    ├── tb_axis_fifo.gtkw
    └── verilator/
        ├── Makefile        Build system for Verilator compilation (portable)
        ├── tb_top.cpp      C++ testbench (embeds test vectors, writes FST)
        └── obj_dir/        Verilator build artefacts (created by `make`)
            ├── Vtm_accelerator   simulation binary (native to build host)
            └── tb_system.fst     FST waveform (written on each run)
            # Run `make clean` if copied from another OS/machine
```

---

## The Tsetlin Machine algorithm

*For RTL engineers unfamiliar with TM-based machine learning.*

### What a Tsetlin Machine is

A TM is a rule-based classifier. It learns a Boolean decision rule for each
output class. Each rule is a set of **clauses**, where a clause is an AND of
selected binary literals.

A **literal** is either a raw feature bit (positive literal) or its complement
(negated literal). For {N} features there are {2*N} literals total.

### Tsetlin Automata and Include actions

During training, a **Tsetlin Automaton (TA)** runs for every (clause, literal)
pair — {CT} × {2*N} = {CT*2*N} automata in total for this model.

Each TA maintains a state integer on the number line [1, {self.tmir.hyperparameters.n_states}].
The midpoint is {self.tmir.hyperparameters.n_states // 2}. When the state exceeds the midpoint
the TA issues an **Include action**: that literal is part of this clause.
Below the midpoint it issues **Exclude**: that literal is ignored.

After training, the states are thresholded once to produce one bit per
(clause, literal) pair:

```
Include[clause, literal] = (ta_state[clause, literal] > n_states / 2)
```

**This is the only information the hardware uses.** The state integers are
not stored. The Include bits are packed into the on-chip tile ROM.

### Four-step inference

```
Step 1  Literal generation
        For each feature f (0-indexed):
          positive literal [f]       = feature_bit[f]
          negated  literal [N + f]   = NOT feature_bit[f]

Step 2  Clause evaluation  (CLAUSE_SLICE clauses computed in parallel each cycle)
        For each clause c:
          has_actions[c] = OR  over l of Include[c, l]     (is the clause non-empty?)
          all_lit_pass[c]= AND over l of (literal[l] OR NOT Include[c, l])
          active[c]      = all_lit_pass[c] AND has_actions[c]

        Plain English: every literal that this clause Includes must be 1.
        An empty clause (zero Includes) is never active.

Step 3  Score accumulation  (one clause per cycle, {CT} cycles)
        The {K} clauses per class are split: first {half_k} are positive,
        last {half_k} are negative.
        For each class j:
          score[j] = clamp(
              SUM active[c] for c in positive_clauses[j]
            - SUM active[c] for c in negative_clauses[j],
            range [-{T}, +{T}])

Step 4  Argmax
        predicted_class = argmax(score[0..{C-1}])
        Ties are broken by lower class index.
```

### Why positive and negative clauses?

The split creates competition. Positive clauses learn conditions *for* the
class; negative clauses learn conditions *against* it. A high score means the
class's positive evidence fired and its negative evidence did not. Clamping
prevents any single class from running away purely because it has more active
clauses than others.

---

## Hardware architecture

*For ML engineers unfamiliar with RTL design and for RTL engineers needing
the dataflow picture.*

### Top-level interface

```
INPUTS
  clk                          rising-edge clock
  rst_n                        asynchronous reset, active-low
  s_axis_tvalid                producer has a beat ready
  s_axis_tdata [{AW}-1:0]      {AW}-bit data beat (features packed LSB-first)
  s_axis_tlast                 high on the last beat of a feature vector
  s_axis_tready  ◄─ OUTPUT     module can accept a beat (FIFO not full)

OUTPUTS  (AXI-Stream master — single beat per inference)
  m_axis_tvalid                asserted when result is ready; held until m_axis_tready
  m_axis_tdata [{AW}-1:0]      predicted class in bits [ceil(log2 {C})-1:0]; upper bits zero
  m_axis_tlast                 always 1 with m_axis_tvalid (single-beat packet)
  busy                         high during inference; low in S_IDLE
INPUT
  m_axis_tready                back-pressure: hold low to pause output handshake
```

The {N}-bit feature vector is split into {NB} AXI beats of {AW} bits each.
Beat 0 carries features[0..{min(AW,N)-1}], beat 1 carries
features[{min(AW,N)}..{min(2*AW,N)-1}], and so on, all LSB-first within
each beat.

### Five-state FSM

```
  ┌─────────────────────────────────────────────────────────────────┐
  │                           S_IDLE                                │
  │  Waits for fifo_m_tvalid. Captures beat 0 → feature_reg[AW-1:0]│
  └───────────┬──────────────────────────────────┬─────────────────┘
              │ tlast=0 (multi-beat)              │ tlast=1 (single-beat)
              ▼                                   │
  ┌───────────────────────┐                       │
  │        S_RECV         │                       │
  │  Captures beats 1..{NB-1} into feature_reg   │
  └───────────┬───────────┘                       │
              │ fifo_m_tlast                      │
              └──────────────────────┬────────────┘
                                     ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                         S_COMPUTE                                │
  │  {NFS}×{NCS} = {NFS*NCS} cycles. feat_cnt×clause_slice_cnt sweep│
  │  tile ROM → CLAUSE_SLICE clause_eval instances (combinatorial)  │
  │  → AND into clause_pass[]; OR into clause_has_actions[]          │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │ all tiles done
                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                          S_SCORE                                 │
  │  {CT} cycles. score_cnt = 0..{CT-1}                              │
  │  score_acc sees one clause per cycle; updates score[class]       │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │ score_cnt == {CT-1}
                                 ▼
  ┌──────────────────────────────────────────────────────────────────┐
  ┌──────────────────────────────────────────────────────────────────┐
  │                          S_DONE                                  │
  │  Asserts m_axis_tvalid/tdata/tlast with argmax result.          │
  │  Holds until m_axis_tready (back-pressure supported).           │
  │  Arms score_clear on handshake cycle.                           │
  └──────────────────────────────┬───────────────────────────────────┘
                                 │ m_axis_tready → S_IDLE
```

Total latency (last TLAST → m_axis_tvalid): ≈ **{latency} cycles**.

### Tiled evaluation — why tiles?

Evaluating all {CT} clauses against all {N} features simultaneously would
require {CT*N} wires and an impractically wide ROM port. Instead the
Include-bit matrix is divided into a grid of tiles.

```
         clause_slice_cnt ─────────────────────────►
         0          1          2       ...   {NCS-1}
feat  0  tile(0,0)  tile(0,1)  tile(0,2) ... tile(0,{NCS-1})
cnt   1  tile(1,0)  tile(1,1)  ...
 │    :
 ▼  {NFS-1} tile({NFS-1},0) ...              tile({NFS-1},{NCS-1})
```

Each tile covers {FS} features × {CS} clauses = {TW} bits. The FSM
evaluates one tile per clock cycle in S_COMPUTE, sweeping feat_cnt (outer)
and clause_slice_cnt (inner).

**Empty tile rule.** When a clause has no Include actions for a particular
feature slice, `tile_has_actions[k]=0` and `clause_eval` returns `active=0`.
The FSM treats this as a vacuous pass: `clause_pass[g] AND= (tile_pass[g] OR NOT tile_has_actions[g])`.
An empty tile does not kill the clause; it contributes nothing to the decision.

### ROM layout

Address: `feat_cnt × {NCS} + clause_slice_cnt`

Bit layout within one {TW}-bit ROM entry (k = clause index 0..{CS-1} within
the tile, l = local feature index 0..{FS-1}):

```
  bit [k×{2*FS} + l]        Include for positive literal l, clause k
  bit [k×{2*FS} + {FS} + l] Include for negated  literal l, clause k
```

This matches `partial_lits = {{~feat_pos, feat_pos}}` so that `clause_eval`
can compute `&(literals | ~ta_action_mask)` without any address translation.

---

## Module reference

### `axis_fifo.v` — AXI-Stream synchronous FIFO

Buffers incoming feature beats while the FSM is not yet in S_IDLE or S_RECV.
Registered (not fall-through) output: data appears on `m_tdata` one cycle
after the read pointer advances.

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `clk` / `rst_n` | in | 1 | Clock / async reset |
| `s_tvalid` | in | 1 | Producer has data |
| `s_tready` | out | 1 | FIFO not full (combinatorial) |
| `s_tdata` | in | {AW} | Data beat |
| `s_tlast` | in | 1 | Last beat marker |
| `m_tvalid` | out | 1 | Data available to consumer |
| `m_tready` | in | 1 | Consumer will accept |
| `m_tdata` | out | {AW} | Data to consumer |
| `m_tlast` | out | 1 | Last beat to consumer |

`s_tready = NOT full`. `m_tvalid` follows the read pointer with 1 cycle
latency. Both pointers are one bit wider than the address to distinguish
full from empty without a counter.

### `clause_eval.v` — combinatorial clause evaluator

Implements the TM clause-firing rule for one clause against one feature
slice (2×FEAT_SLICE literals).

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `literals` | in | `2×{FS}` | `{{~feat_pos, feat_pos}}` — positive then negated |
| `ta_action_mask` | in | `2×{FS}` | Include bits from ROM tile |
| `active` | out | 1 | 1 iff clause fires for this feature window |

```
no_actions = (ta_action_mask == 0)          // empty clause
all_active = AND over l: (literals[l] OR NOT ta_action_mask[l])
active     = all_active AND NOT no_actions
```

{CS} instances of this module run in parallel each cycle of S_COMPUTE.
All instances share the same `literals` (feature window); only
`ta_action_mask` differs (one row of the tile per instance).

### `score_acc.v` — per-class vote accumulator

Holds one signed score per class. Updated by one clause per valid cycle.

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `clk` / `rst_n` | in | 1 | Clock / reset |
| `clear` | in | 1 | Synchronous clear of all scores to 0 |
| `valid` | in | 1 | Update enable (asserted during S_SCORE) |
| `cls` | in | `⌈log₂{C}⌉` | Target class (0..{C-1}) |
| `polarity` | in | 1 | 1=positive clause (+1), 0=negative (−1) |
| `active` | in | 1 | Whether the clause fired |
| `scores_flat` | out | `{SW}×{C}` | Packed signed scores |

Update guard: `valid AND active`. Inactive clauses leave all scores unchanged.
Clamping: `if positive: score < T → score++`; `if negative: score > -T → score--`.
Score range after clamping: [-(T-1), T-1].

Unpack class j: `$signed(scores_flat[j×{SW} +: {SW}])`.

### `argmax.v` — combinatorial tournament argmax

Scans classes 0 → {C-1} in a for-loop. Ties go to the lower-indexed class.

| Port | Dir | Width | Description |
|------|-----|-------|-------------|
| `scores_flat` | in | `{SW}×{C}` | Packed signed scores from `score_acc` |
| `m_axis_tvalid` | out | 1 | Result valid; held until `m_axis_tready` |
| `m_axis_tready` | in | 1 | Consumer ready — back-pressure |
| `m_axis_tdata` | out | {AW} | Bits [{CW}-1:0] = winning class index; upper bits 0 |
| `m_axis_tlast` | out | 1 | Always 1 with `m_axis_tvalid` (single-beat packet) |

Purely combinatorial: output is stable within the same cycle that
`scores_flat` settles (i.e. by the time S_DONE samples it).

### `tm_accelerator.v` — top module

Contains the FSM, tile ROM, `clause_pass[]` / `clause_has_actions[]`
registers, decode functions, and wires together all sub-modules.

**Key internal signals**

| Signal | Width | Meaning |
|--------|-------|---------|
| `feature_reg` | {NFS*FS} | Padded feature vector captured from FIFO |
| `feat_pos` | {FS} | Current feature window: `feature_reg[feat_cnt×{FS} +: {FS}]` |
| `partial_lits` | {2*FS} | `{{~feat_pos, feat_pos}}` fed to all clause_eval instances |
| `tile_rom_data` | {TW} | ROM output for current `(feat_cnt, clause_slice_cnt)` |
| `tile_pass[k]` | 1 | clause_eval `active` output for clause k in current tile |
| `tile_has_actions[k]` | 1 | `OR(ta_actions_k)` — non-empty flag for clause k in tile |
| `clause_pass[g]` | 1-bit array | AND-accumulated fire result for each of {CT} clauses |
| `clause_has_actions[g]` | 1-bit array | OR-accumulated non-empty flag per clause |
| `score_cnt` | {CCW} | Clause counter during S_SCORE (0 → {CT-1}) |
| `score_class` | `⌈log₂{C}⌉` | `clause_to_class(score_cnt)` |
| `score_is_pos` | 1 | `clause_is_positive(score_cnt)` — true for first {half_k} of each class |
| `score_active` | 1 | `clause_pass[score_cnt] AND clause_has_actions[score_cnt]` |

**Decode functions**

`clause_to_class(c)` and `clause_is_positive(c)` are Verilog functions
implemented as `case` statements. They map global clause index → class and
polarity respectively. This avoids dividers in hardware; the mapping is
precomputed at generation time and inlined.

---

## Simulation guide

### Waveform reading order

The `.gtkw` files pre-configure GTKWave with named signal groups. Work through
them top-to-bottom — they follow the dataflow order.

**1. FSM State** (`state`, `busy`)
- Verify the sequence IDLE → RECV → COMPUTE → SCORE → DONE → IDLE.
- Time from last `s_axis_tlast` to `m_axis_tvalid` should be ≈ {latency} cycles.
- `busy` goes low on the same cycle as the `m_axis_tvalid`/`m_axis_tready` handshake.

**2. AXI-Stream Input** (`s_tvalid`, `s_tready`, `s_tdata`, `s_tlast`)
- A beat transfers on a rising edge where both valid and ready are high.
- `s_tready` should stay high for all {NB} beats (FIFO has {FD} entries of
  headroom).
- `s_tlast` must be high on beat {NB-1} and only on that beat.

**3. Tile Counters** (`feat_cnt`, `clause_slice_cnt`, `score_cnt`, `beat_cnt`)
- During S_COMPUTE: `feat_cnt` is the slow outer counter (0 → {NFS-1}),
  `clause_slice_cnt` is the fast inner counter (0 → {NCS-1}).
- During S_SCORE: `score_cnt` counts 0 → {CT-1}.
- Confirm these sweep their full ranges without gaps.

**4. TA Action Signals** (`tile_eval[k].ta_actions_k`, `tile_has_actions[k]`)
- Focus on k=0 (the first clause within each tile).
- `ta_actions_k` should change every cycle during S_COMPUTE as the ROM
  address changes.
- `tile_has_actions[k]=0` on empty tiles. The corresponding `clause_pass[k]`
  should NOT drop to 0 (vacuous-pass rule).

**5. Clause Accumulators** (`clause_pass[0]`, `clause_has_actions[0]`)
- Clause 0 updates once per `clause_slice_cnt==0` cycle (every {NCS} cycles).
- First update (feat_cnt=0): seeds the register.
- Subsequent updates: AND with tile result.
- Final value after S_COMPUTE ends is the clause's fire decision.

**6. Score Accumulation** (`scores_flat`, `score_class`, `score_is_pos`, `score_active`)
- `scores_flat` is a packed {SW}×{C}-bit bus; examine individual classes.
- Confirm `score_is_pos` alternates correctly for each class's {K} clauses
  (first {half_k} positive, next {half_k} negative).
- Total score range after all {CT} clauses: [−{T-1}, +{T-1}].

**7. AXI-Stream Output** (`m_tvalid`, `m_tready`, `m_tdata`, `m_tlast`)
- `m_tvalid` asserts when the result is ready and stays high until `m_tready`.
- `m_tdata[{CW}-1:0]` holds the winning class index.
- `m_tlast` is wired to `m_tvalid` — every output packet is exactly one beat.

### Unit testbenches

| Testbench | Key checks | Open waveform |
|-----------|-----------|---------------|
| `tb_clause_eval` | all-Include fires, partial Include, empty clause=0, negated literal | `waves.sh tb_clause_eval` |
| `tb_score_acc` | +1 accumulation to +T, -1 accumulation to -T, inactive=no change, clear | `waves.sh tb_score_acc` |
| `tb_argmax` | highest score wins, tie-break to lower index | `waves.sh tb_argmax` |
| `tb_axis_fifo` | beat ordering, TLAST position, full/empty | `waves.sh tb_axis_fifo` |

---

## Verilog concepts for ML engineers

*The RTL uses five Verilog constructs repeatedly. This section explains each
one without assuming prior hardware knowledge.*

### 1. `wire` vs `reg` — combinatorial vs registered signals

```verilog
wire foo = a & b;      // combinatorial: updated instantly when a or b changes
reg  bar;              // register: holds its value; only changes on a clock edge
```

A `wire` is like an expression that is evaluated eagerly at every moment. A
`reg` is like a variable that is frozen between clock ticks and only rewritten
when the always block fires.

### 2. Non-blocking assignment `<=` — scheduled register updates

```verilog
always @(posedge clk) begin
    clause_pass[g] <= clause_pass[g] & tile_pass[k];  // schedules the write
end
```

The right-hand side is evaluated with the *current* (pre-clock-edge) values.
The write happens *after* all always blocks in the design have evaluated.
This prevents the classic race condition where one register's new value
accidentally feeds into another's update in the same cycle.

Blocking `=` (used for loop variables like `clause_g` inside always blocks)
updates immediately within the procedural flow and is not visible to other
blocks — safe to use for temporary index computations.

### 3. Parameters — compile-time constants

```verilog
parameter CLAUSE_SLICE = {CS};   // can be overridden at instantiation
localparam LAST_CS = N_CLAUSE_SLICES - 1;  // derived, not overridable
```

Parameters are like Python constants or C `#define`. They exist only at
compile time; there are no arithmetic units in silicon for them.

### 4. Generate blocks — hardware-time for-loops

```verilog
genvar k;
generate
    for (k = 0; k < CLAUSE_SLICE; k = k + 1) begin : tile_eval
        wire [2*FEAT_SLICE-1:0] ta_actions_k;
        assign ta_actions_k = tile_rom_data[k * 2 * FEAT_SLICE +: 2 * FEAT_SLICE];
        clause_eval u_ce (.literals(partial_lits), .ta_action_mask(ta_actions_k), ...);
    end
endgenerate
```

This creates {CS} independent `clause_eval` hardware instances that all
operate in parallel every cycle. It is not a software loop — all {CS}
instances exist simultaneously as physical logic. In GTKWave you can
inspect each instance individually as `tile_eval[0].ta_actions_k`,
`tile_eval[1].ta_actions_k`, etc.

### 5. Dynamic part-select `[base +: width]` and concatenation `{{a, b}}`

```verilog
feature_reg[feat_cnt * {FS} +: {FS}]   // {FS} bits starting at feat_cnt×{FS}
{{~feat_pos, feat_pos}}                  // concatenate: negated bits above, positive below
scores_flat[j * {SW} +: {SW}]           // unpack class j's score from the flat bus
```

`[base +: width]` selects a fixed-width window at a run-time offset — useful
for indexing into an array without a multiplexer for each element.
`{{a, b}}` concatenates two signals into one wider bus. The left operand
occupies the most-significant bits.

---

## Integration guide

### Connecting to a larger system

```
  Your producer             tm_accelerator          Your consumer
  ─────────────             ──────────────          ─────────────
  s_axis_tvalid ──────────►                         m_axis_tvalid ──────────►
  s_axis_tdata  ──────────►                         m_axis_tdata  ──────────►
  s_axis_tlast  ──────────►                         m_axis_tlast  ──────────►
                ◄────────── s_axis_tready           ◄──────────── m_axis_tready
                                                    busy          ──────────►  (optional)
```

### AXI-Stream protocol

- A beat is accepted on a rising clock edge where **both** `s_axis_tvalid`
  and `s_axis_tready` are high. The producer must not change `s_tdata` or
  `s_tlast` after asserting `s_tvalid` until the beat is accepted.
- `s_axis_tready` may de-assert only when the internal FIFO is full. With
  a {FD}-entry FIFO and {NB}-beat vectors this should not occur in practice.
- `s_axis_tlast` must be high on the last beat (beat index {NB-1}) and low
  on all others.

### Back-to-back inferences

The module returns to S_IDLE on the cycle the `m_axis_tvalid`/`m_axis_tready`
handshake completes. The first beat of the next inference can be presented as
soon as `busy` goes low (equivalent to the handshake cycle).

Scores are cleared in S_IDLE (one cycle after the handshake asserts `score_clear`).
This is transparent to the external interface.

### Reset

Assert `rst_n=0` for at least 2 clock cycles before normal operation. All
internal registers clear to 0: `feature_reg`, `clause_pass`,
`clause_has_actions`, `scores`, `state`, `m_axis_tvalid`, `m_axis_tdata`.

### Output timing

`m_axis_tdata` is registered in S_DONE. It is stable from the cycle
`m_axis_tvalid` is asserted until the handshake completes. With `m_axis_tready`
permanently tied high, the handshake completes on the first cycle of S_DONE.

### Critical path

The deepest combinatorial path in S_COMPUTE is inside `clause_eval`:
`&(literals | ~ta_action_mask)` — a {2*FS}-bit AND reduction over ORed
pairs. This is a {2*FS}-input gate tree, well within a single cycle at any
standard-cell technology node.

The ROM read is a registered-array read: `tile_rom[feat_cnt × {NCS} + clause_slice_cnt]`.
The address is stable for the full cycle before the data is used by
`clause_eval`, so there is no read-address timing concern.

---

*All extracted TA Include-action outputs from the trained model are frozen into
this RTL at generation time and cannot be changed without regeneration. This
directory is fully standalone and requires only a Verilog simulator (iverilog,
Verilator, Xsim) or synthesis tool.*
"""

    def _gen_iverilog_script(self) -> str:
        tbs = ["tb_axis_fifo", "tb_clause_eval", "tb_score_acc", "tb_argmax", "tb_system"]
        lines = [
            "#!/usr/bin/env bash",
            "# Auto-generated by matador rtl generate",
            "set -euo pipefail",
            "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"",
            "SRC=\"$SCRIPT_DIR/../src\"",
            "TB=\"$SCRIPT_DIR/../tb\"",
            "OUT=\"$SCRIPT_DIR\"",
            "PASS=0; FAIL=0",
            "",
        ]
        for tb in tbs:
            lines += [
                f"echo \"=== {tb} ===\"",
                f"iverilog -g2001 -Wall -Wno-timescale -o \"$OUT/{tb}\" \\",
                f"    \"$SRC\"/axis_fifo.v  \"$SRC\"/clause_eval.v \\",
                f"    \"$SRC\"/score_acc.v  \"$SRC\"/argmax.v     \\",
                f"    \"$SRC\"/tm_accelerator.v \"$TB\"/{tb}.v",
                f"cd \"$OUT\"",
                f"VVP_OUT=$(vvp \"$OUT/{tb}\" 2>&1)",
                f"echo \"$VVP_OUT\"",
                f"if echo \"$VVP_OUT\" | grep -qE 'FAIL|TIMEOUT'; then",
                f"    FAIL=$((FAIL+1))",
                f"else",
                f"    PASS=$((PASS+1))",
                f"fi",
                "",
            ]
        lines += [
            "echo \"\"",
            "echo \"Results: $PASS passed, $FAIL failed\"",
            "[ \"$FAIL\" -eq 0 ]",
        ]
        return "\n".join(lines) + "\n"

    def _gen_verilator_lint(self) -> str:
        return textwrap.dedent("""\
        #!/usr/bin/env bash
        # Auto-generated by matador rtl generate
        # Lints RTL sources only (no timing in design sources).
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        SRC="$SCRIPT_DIR/../src"

        verilator --lint-only -Wall --Wno-TIMESCALEMOD --Wno-WIDTH \\
            --language 1800-2005 \\
            --top-module tm_accelerator \\
            "$SRC"/axis_fifo.v     \\
            "$SRC"/clause_eval.v   \\
            "$SRC"/score_acc.v     \\
            "$SRC"/argmax.v        \\
            "$SRC"/tm_accelerator.v

        echo "Verilator lint: PASSED"
        """)

    def _gen_xsim_script(self) -> str:
        tbs = ["tb_axis_fifo", "tb_clause_eval", "tb_score_acc", "tb_argmax", "tb_system"]
        lines = [
            "#!/usr/bin/env bash",
            "# Auto-generated by matador rtl generate",
            "set -euo pipefail",
            "SCRIPT_DIR=\"$(cd \"$(dirname \"$0\")\" && pwd)\"",
            "SRC=\"$SCRIPT_DIR/../src\"",
            "TB=\"$SCRIPT_DIR/../tb\"",
            "",
            "xvlog --nolog -sv \"$SRC\"/*.v",
            "",
        ]
        for tb in tbs:
            lines += [
                f"echo \"=== {tb} ===\"",
                f"xvlog --nolog \"$TB\"/{tb}.v",
                f"xelab --nolog --debug off --top {tb} --snapshot {tb}_snap",
                f"xsim  --nolog {tb}_snap --runall",
                "",
            ]
        return "\n".join(lines) + "\n"
