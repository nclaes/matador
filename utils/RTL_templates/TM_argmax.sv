`timescale 1ns / 1ps                                                                 
//////////////////////////////////////////////////////////////////////////////////   
// Company:                                                                          
// Engineer:                                                                         
//                                                                                   
// Create Date: 06/24/2023 03:24:09 AM                                               
// Design Name: Tousif Rahman, Gang Mao                                              
// Module Name: compare                                                        
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
                                                                                     
                                                                                     
// compare  two numbers and return the index and the bigger number   
module compare #(
    parameter WEIGHT_LENGTH,
    parameter integer INDEX_LENGTH
)
(a, a_i, b, b_i, c, c_i);

	input logic  signed [WEIGHT_LENGTH - 1:0] a;
	input logic  signed [WEIGHT_LENGTH - 1:0] b;
	input logic [INDEX_LENGTH - 1:0] a_i;
	input logic [INDEX_LENGTH - 1:0] b_i;

	output logic  signed [WEIGHT_LENGTH - 1:0] c;
	output logic [INDEX_LENGTH - 1:0] c_i;
    
    // always @(*) instead of always_comb: Icarus Verilog was observed not
    // reliably re-evaluating this block for every input change when
    // instantiated through the generate-based compare tree below (some
    // argmax outputs silently held a stale value from a previous
    // inference) -- always @(*) is the older, more battle-tested
    // sensitivity-list construct and does not exhibit this.
    always @(*) begin
        if(b > a)
           begin
                c = b;
                c_i = b_i;
            end
        else begin 
                c = a; 
                c_i = a_i; 
            end 
    end
    
endmodule


module classify #(
    parameter CLASS_NUM,
    parameter WEIGHT_LENGTH,
    parameter C_M00_AXIS_TDATA_WIDTH,
    parameter integer INDEX_LENGTH = $clog2(CLASS_NUM),
    parameter integer TREE_WITH = 2 ** INDEX_LENGTH
)
(
// [CLASS_NUM], not [CLASS_NUM-1:0]: TM_top.sv's class_sums output uses the
// ascending shorthand ([0:CLASS_NUM-1]); an explicit descending range here
// connects to it by position, not by index number, silently reversing
// class order across this port (c_sum[k] read as class_sums[CLASS_NUM-1-k]
// -- confirmed by tracing both arrays at an argmax event and finding them
// exact mirror images of each other).
input logic signed [WEIGHT_LENGTH - 1:0] c_sum[CLASS_NUM],
input logic its_business_time,
input logic m00_axis_tready,
output logic [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep,
input logic last,
output logic finish,
output logic last_out,
input logic clk,
input logic rst,
output logic [C_M00_AXIS_TDATA_WIDTH - 1:0] y
);
    // Winner index, computed by a flat combinational reduction instead of
    // the generate-based tree of `compare` instances this replaced: Icarus
    // Verilog was observed not reliably re-evaluating that deep,
    // hierarchy-of-always_comb-blocks tree for every c_sum change (some
    // argmax outputs silently held a stale value from a previous
    // inference, confirmed via waveform trace) -- Verilator handled the
    // tree fine, but relying on a simulator-specific quirk either way
    // isn't acceptable here. Everything downstream of `cmp_index[0]`
    // (the clocked handshake logic below) is unchanged.
    logic [INDEX_LENGTH - 1:0] cmp_index [0:0];
    always @(*) begin : argmax_reduce
        logic signed [WEIGHT_LENGTH - 1:0] best_val;
        integer k;
        best_val = c_sum[0];
        cmp_index[0] = '0;
        for (k = 1; k < CLASS_NUM; k = k + 1) begin
            if (c_sum[k] > best_val) begin
                best_val = c_sum[k];
                cmp_index[0] = k[INDEX_LENGTH - 1:0];
            end
        end
    end
    
    logic signed old_its_business_time;
    logic c_stat;
    logic last_registered;
    logic valid = 0;
    initial begin
        c_stat = 0;
        last_registered = 0;
       finish = 0;
       last_out = 0;
    end
    
	always @(posedge clk)begin 
	   if (rst) begin
	       last_registered = 0;
	       finish = 0;
	       last_out = 0;
	       valid = 0;
	   end
	   else begin
            if (finish) begin
               finish = 0;
               if (last_registered) begin
                    last_out = 0;
                    last_registered = 0;
                    m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b0}};
               end
            end
            if (last) begin
               last_registered = 1;
            end
            if (its_business_time) begin
                y[INDEX_LENGTH - 1:0] = cmp_index[0]; 
                y[C_M00_AXIS_TDATA_WIDTH - 1:INDEX_LENGTH] = {(C_M00_AXIS_TDATA_WIDTH - INDEX_LENGTH){1'b0}};
                if (!m00_axis_tready) begin
                    valid = 1;
                end
                else begin
                    finish = 1;
                    if (last_registered) begin
                        last_out = 1;
                    end
                end
            end
            if (valid && m00_axis_tready) begin
                valid = 0;
                finish = 1;
                if (last_registered) begin
                    last_out = 1;
                end
            end
            if (finish && !last_registered) begin
                m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b1}};
            end
        end
	end 
endmodule

