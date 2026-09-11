`timescale 1ns/1ps
// =============================================================================
// argmax_rt — combinatorial argmax over a RUNTIME number of classes
// =============================================================================
// Identical algorithm to argmax.v (linear scan, strict >, ties to the lower
// index) with one addition: only classes with index < n_classes participate.
//
// WHY MASKING IS REQUIRED
//   score_acc_rt clears ALL compile-time class registers, so a model with
//   fewer classes leaves the unused scores at 0. Without masking, a model
//   whose real classes all score negative would lose to a phantom class
//   sitting at 0. The (k < n_classes) guard removes phantom classes from the
//   tournament entirely.
//
// n_classes is CLASS_WIDTH+1 bits wide so the value N_CLASSES itself
// (e.g. 16 with CLASS_WIDTH=4) is representable. Callers keep it in
// 1..N_CLASSES; tm_accel_gp.v validates this at CMD_LOAD time.
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
