`timescale 1ns/1ps
// =============================================================================
// score_acc_rt — per-class vote accumulator with RUNTIME threshold
// =============================================================================
// Identical semantics to score_acc.v, except THRESHOLD is an input port
// instead of a compile-time parameter, so a newly loaded model can carry its
// own clamping bound (CMD_LOAD header word 1, byte [23:16]).
//
// CLAMPING (same guarded update as score_acc.v)
//   positive: update only if score <  threshold   -> ceiling  = +threshold
//   negative: update only if score > -threshold   -> floor    = -threshold
//   i.e. the reachable range is [-threshold, +threshold] inclusive.
//   (Note: the original README states [-(T-1), T-1]; the RTL and
//   tb_score_acc both realise +/-T. This module matches the RTL.)
//
// CONSTRAINT ON threshold
//   Callers must keep threshold in 1 .. 2^(SCORE_WIDTH-1)-1 so that
//   $signed(threshold) is positive and +/-threshold fit SCORE_WIDTH bits.
//   tm_accel_gp.v validates this range before accepting a CMD_LOAD header.
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
    input  wire [SCORE_WIDTH-1:0]           threshold,   // runtime clamp bound
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
                if ($signed(scores[cls]) < $signed(threshold))
                    scores[cls] <= scores[cls] + 1;
            end else begin
                if ($signed(scores[cls]) > -$signed(threshold))
                    scores[cls] <= scores[cls] - 1;
            end
        end
    end

endmodule
