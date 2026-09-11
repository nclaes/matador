`timescale 1ns/1ps
// =============================================================================
// score_acc_rt — per-class vote accumulator, unclamped
// =============================================================================
// Same unclamped accumulation as score_acc.v (see its header comment for the
// full rationale): scores are the raw, unclamped sum of clause votes, never
// saturated. SCORE_WIDTH is sized (see the backend's _capacity_params()) to
// the true worst-case vote magnitude any model within this core's
// compile-time capacity could ever produce -- half of MAX_CLAUSES_TOTAL, all
// one polarity, all firing -- so the register can never overflow.
//
// This module used to carry a vestigial `threshold` input port (CMD_LOAD's
// header used to reserve a byte for it purely for wire-protocol compatibility
// with an even older runtime-clamp design; it never affected accumulation
// here). That header byte was reclaimed to widen `clauses_per_class` from
// 8 to 16 bits (see tm_accel_gp.v's header comment), which retired the field
// from the wire protocol entirely -- so the port is gone too, not just unused.
// (An earlier version of this module used `threshold` as a RUNTIME clamp
// bound, guarding each increment/decrement (`if (score < threshold) ...`).
// That was order-dependent -- see score_acc.v's header -- and removed in
// favor of the exact, unclamped sum, matching vanilla_tiled's score_acc.v.)
//
// N_CLASSES here is the compile-time CAPACITY. With a model using fewer
// classes, unused score registers simply stay 0 (clear covers all of them);
// argmax_rt masks them out of the decision.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module score_acc_rt #(
    parameter N_CLASSES   = 16,
    parameter SCORE_WIDTH = 6
)(
    input  wire                             clk,
    input  wire                             rst_n,
    input  wire                             clear,
    input  wire                             valid,
    input  wire [$clog2(N_CLASSES)-1:0]     cls,
    input  wire                             polarity,
    input  wire                             active,
    output wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat
);

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
                scores[i] = {SCORE_WIDTH{1'b0}};
        end else if (clear) begin
            for (i = 0; i < N_CLASSES; i = i + 1)
                scores[i] = {SCORE_WIDTH{1'b0}};
        end else if (valid & active) begin
            if (polarity) begin
                scores[cls] <= scores[cls] + 1;
            end else begin
                scores[cls] <= scores[cls] - 1;
            end
        end
    end

endmodule
