`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/27/2023 03:56:34 AM
// Design Name:  Tousif Rahman, Gang Mao
// Module Name: new_adder
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module weight_adder_tile #(
    parameter STAGE_NUM,
    parameter CLAUSE_NUM,
    parameter WEIGHT_LENGTH,
    parameter CLAUSE_PER_STAGE = CLAUSE_NUM/STAGE_NUM
)
(
    input clk,
    input rst,
    input valid,
    input logic signed [CLAUSE_PER_STAGE - 1:0] clauses,
    input logic signed [WEIGHT_LENGTH - 1:0] weights[CLAUSE_PER_STAGE - 1:0],
    input [WEIGHT_LENGTH - 1:0] p_c_sum_in,
    output reg [WEIGHT_LENGTH - 1:0]p_c_sum
);
    reg[WEIGHT_LENGTH - 1:0] c_sum[CLAUSE_PER_STAGE - 1:0];
    reg [WEIGHT_LENGTH - 1:0]weight[CLAUSE_PER_STAGE - 1:0];
    generate 
        genvar k;
        for (k = 0 ; k < CLAUSE_PER_STAGE; k = k + 1) begin
            assign weight[CLAUSE_PER_STAGE - 1 - k] = weights[k];
        end
    endgenerate 
    
    generate 
        genvar j;  
        assign c_sum[0] = p_c_sum_in + clauses[0] * weight[0];
        for (j = 1 ; j < CLAUSE_PER_STAGE; j = j + 1) begin
            assign c_sum[j] = c_sum[j - 1] + clauses[j] * weight[j];
        end
    endgenerate
    always@ (posedge  clk) begin
        if (valid) begin
            p_c_sum = c_sum[CLAUSE_PER_STAGE - 1];
        end
    end
endmodule 

module weighted_adder_new #(
    parameter STAGE_NUM,
    parameter CLAUSE_NUM,
    parameter WEIGHT_LENGTH,
    parameter CLAUSE_PER_STAGE = CLAUSE_NUM/STAGE_NUM
)
(clauses, valid, weights, c_sum_out, clk, finished);
    input logic signed clk; 
    input logic signed valid;
	input logic signed [CLAUSE_NUM - 1:0] clauses;
	input logic signed [WEIGHT_LENGTH - 1:0] weights [CLAUSE_NUM];
	output reg [WEIGHT_LENGTH - 1:0] c_sum_out;
	output reg finished;
	logic half_complete;
	reg[WEIGHT_LENGTH - 1:0] c_sum[STAGE_NUM:0];
	reg [STAGE_NUM - 1:0] done;
	//reg [13:0] c_sum_reg[STAGE_NUM:0];
	
	generate 
        genvar i;
        assign c_sum[0] = 0;   
        for (i = 0; i < STAGE_NUM; i = i + 1) begin
            weight_adder_tile #(
                .STAGE_NUM(STAGE_NUM),
                .CLAUSE_NUM(CLAUSE_NUM),
                .WEIGHT_LENGTH(WEIGHT_LENGTH),
                .CLAUSE_PER_STAGE(CLAUSE_PER_STAGE)
            )
            w_tile (
                .clk(clk),
                .rst(rst),
                .valid(done[i]),
                .clauses(clauses[i * CLAUSE_PER_STAGE+:CLAUSE_PER_STAGE]),
                .weights(weights[i * CLAUSE_PER_STAGE+:CLAUSE_PER_STAGE]),
                .p_c_sum_in(c_sum[i]),
                .p_c_sum(c_sum[i + 1])
            );
        end
    endgenerate
	
    initial begin
        done = {STAGE_NUM{1'b0}};
        for (int n = 1; n < STAGE_NUM + 1; n= n + 1) begin
            c_sum[n] = 0;
        end
    end
    
	always @(posedge clk) begin
        if(valid && done == {STAGE_NUM{1'b0}}) begin 
		    done[0] = 1'b1;
        end 
        else if (valid && done != {STAGE_NUM{1'b0}} && done[STAGE_NUM - 1] != 1'b1) begin
            done = done << 1;
        end	
        else if (!valid && done[STAGE_NUM - 1]) begin
            done[STAGE_NUM - 1] <= 0;
//            for (int m = 1; m < STAGE_NUM; m = m + 1) begin
//                c_sum[m] = 0;
//            end
        end
	end
	assign c_sum_out = c_sum[STAGE_NUM];
	assign finished = done[STAGE_NUM - 1];
endmodule



module Adder_new #(
    parameter CLAUSE_NUM,
    parameter CLASS_NUM,
    parameter WEIGHT_LENGTH,
    parameter STAGE_NUM
)
(clk, rst, clauses, class_sums, valid,adder_done);
	input logic clk;
	input logic rst;
	input logic valid;
	input logic [CLAUSE_NUM - 1:0] clauses;
	output logic signed [WEIGHT_LENGTH - 1:0] class_sums [CLASS_NUM];
	output reg adder_done;
	logic signed [WEIGHT_LENGTH - 1:0] class_sums_local [CLASS_NUM];
	logic signed [WEIGHT_LENGTH - 1:0] weights [CLASS_NUM][CLAUSE_NUM];


    hard_coded_weight #(
        .CLAUSE_NUM(CLAUSE_NUM)
    )
    HCW(
        .weights(weights)
    );
    
    reg [CLASS_NUM - 1:0] weight_done;
	generate
		genvar i;
		for(i = 0; i < CLASS_NUM; i= i+1) begin 
			weighted_adder_new #(
            .STAGE_NUM(STAGE_NUM),
            .CLAUSE_NUM(CLAUSE_NUM),
            .WEIGHT_LENGTH(WEIGHT_LENGTH)
			)
			adder_w(
			    .valid(valid),
			    .finished(weight_done[i]),
			    .clk(clk),
				.clauses(clauses),
				.weights(weights[i]),
				.c_sum_out(class_sums_local[i])
			);
		end
	endgenerate
    
    initial begin
        adder_done = 0;
    end
    
	always @(posedge clk) begin 
	   if (rst) begin
	       adder_done = 0;
	   end
        if (weight_done == {CLASS_NUM{1'b1}}) begin
		  class_sums = class_sums_local;
		  adder_done = 1;
		end
		else if (weight_done == {CLASS_NUM{1'b0}}) begin
		  adder_done = 0;
		end
	end
endmodule

