`timescale 1ns/1ps
// =============================================================================
// score_acc_rt — per-class vote accumulator, unclamped, with a legacy
// runtime threshold input retained for wire-protocol compatibility only
// =============================================================================
// Same unclamped accumulation as score_acc.v (see its header comment for the
// full rationale): scores are the raw, unclamped sum of clause votes, never
// saturated. SCORE_WIDTH is sized (see the backend's _capacity_params()) to
// the true worst-case vote magnitude any model within this core's
// compile-time capacity could ever produce -- half of MAX_CLAUSES_TOTAL, all
// one polarity, all firing -- so the register can never overflow.
//
// `threshold` IS VESTIGIAL. This core is reprogrammable: CMD_LOAD's header
// (word 1, byte [23:16]) still carries a threshold field so existing host
// packers (tm_emulator.py::encode_load_packet()) and the wire protocol keep
// working unchanged, and tm_accel_gp.v still range-checks and latches it
// into cfg_threshold. It is wired into this module's `threshold` port for
// that reason alone -- it has no effect on accumulation.
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
    input  wire [SCORE_WIDTH-1:0]           threshold,   // vestigial, see header
    output wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat
);

    /* verilator lint_off UNUSEDSIGNAL */
    wire [SCORE_WIDTH-1:0] _unused_threshold = threshold;
    /* verilator lint_on UNUSEDSIGNAL */

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
