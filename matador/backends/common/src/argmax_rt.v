`timescale 1ns/1ps
// =============================================================================
// argmax_rt — combinatorial argmax over a RUNTIME number of classes
// =============================================================================
// Shared across three backends (vanilla_gp_tiled, redress, and
// vanilla_hardwired's "templated" scoring mode) — one tested module, copied
// verbatim by each backend's generate(), rather than three independent
// copies. See matador/backends/common/README.md for the shared-module
// convention this directory follows.
//
// Identical algorithm to argmax.v (linear scan, strict >, ties to the lower
// index) with one addition: only classes with index < n_classes participate.
//
// WHY MASKING IS REQUIRED
//   A caller that clears ALL compile-time class registers/scores up front
//   (gp_tiled's score_acc_rt, redress's score_acc_rt) leaves unused scores
//   at 0 for a model with fewer classes than the compile-time ceiling.
//   Without masking, a model whose real classes all score negative would
//   lose to a phantom class sitting at 0. The (k < n_classes) guard removes
//   phantom classes from the tournament entirely. A caller with a
//   compile-time-fixed class count (vanilla_hardwired) simply ties
//   n_classes to N_CLASSES itself — the guard is then always true, a no-op.
//
// n_classes is CLASS_WIDTH+1 bits wide so the value N_CLASSES itself
// (e.g. 16 with CLASS_WIDTH=4) is representable. Callers keep it in
// 1..N_CLASSES; the caller's own load-time validation enforces this.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module argmax_rt #(
    parameter N_CLASSES   = 16,
    parameter SCORE_WIDTH = 6,
    parameter CLASS_WIDTH = 4
)(
    input  wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat,
    input  wire [CLASS_WIDTH:0]             n_classes,   // runtime, 1..N_CLASSES
    output reg  [CLASS_WIDTH-1:0]           pred_class
);
    integer k;
    reg signed [SCORE_WIDTH-1:0] cur_max;
    reg signed [SCORE_WIDTH-1:0] score_k;

    always @(*) begin
        cur_max    = $signed(scores_flat[SCORE_WIDTH-1:0]);
        pred_class = {CLASS_WIDTH{1'b0}};
        for (k = 1; k < N_CLASSES; k = k + 1) begin
            score_k = $signed(scores_flat[k*SCORE_WIDTH +: SCORE_WIDTH]);
            if ((k[CLASS_WIDTH:0] < n_classes) && (score_k > cur_max)) begin
                cur_max    = score_k;
                pred_class = k[CLASS_WIDTH-1:0];
            end
        end
    end
endmodule
