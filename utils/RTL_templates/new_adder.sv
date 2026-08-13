`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: Adder_new
//
// Weighted-sum adder for the Coalesced TM: for each class, sums the signed
// per-clause weight of every clause that fired.
//
// Rewritten from the original staged/tiled adder tree (weight_adder_tile /
// weighted_adder_new), which wired the weight array into a sub-module port
// via an indexed part-select (`weights[i*CLAUSE_PER_STAGE +: CLAUSE_PER_STAGE]`)
// -- valid SystemVerilog that Verilator accepts but Icarus Verilog cannot
// elaborate ("Array slices are not yet supported for continuous
// assignment"). This version indexes the weight array directly inside one
// module instead of slicing it across a port, so the same generated design
// runs under both simulators.
//
// STAGE_NUM is accepted for interface compatibility but unused: this
// version computes every class's full sum in one combinational reduction,
// registered on the next clock edge, rather than genuinely pipelining
// across STAGE_NUM stages. That's a scope reduction (simpler, portable,
// one cycle of latency instead of STAGE_NUM), not a behavioral change to
// what gets computed -- revisit if timing closure on real hardware needs
// the pipelining back.
//////////////////////////////////////////////////////////////////////////////////

module Adder_new #(
    parameter CLAUSE_NUM,
    parameter CLASS_NUM,
    parameter WEIGHT_LENGTH,
    parameter STAGE_NUM = 1
)
(
    input  logic clk,
    input  logic rst,
    input  logic [CLAUSE_NUM - 1:0] clauses,
    output logic signed [WEIGHT_LENGTH - 1:0] class_sums [CLASS_NUM],
    input  logic valid,
    output logic adder_done
);

    logic signed [WEIGHT_LENGTH - 1:0] weights [CLASS_NUM][CLAUSE_NUM];

    hard_coded_weight #(
        .CLAUSE_NUM(CLAUSE_NUM)
    )
    HCW (
        .weights(weights)
    );

    integer c, j;
    logic signed [WEIGHT_LENGTH - 1:0] sum;

    initial begin
        adder_done = 1'b0;
    end

    always @(posedge clk) begin
        if (rst) begin
            adder_done <= 1'b0;
        end
        else if (valid) begin
            for (c = 0; c < CLASS_NUM; c = c + 1) begin
                sum = {WEIGHT_LENGTH{1'b0}};
                for (j = 0; j < CLAUSE_NUM; j = j + 1) begin
                    if (clauses[j]) begin
                        sum = sum + weights[c][j];
                    end
                end
                class_sums[c] <= sum;
            end
            adder_done <= 1'b1;
        end
        else begin
            adder_done <= 1'b0;
        end
    end

endmodule
