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
    
    always_comb begin 
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
input logic signed [WEIGHT_LENGTH - 1:0] c_sum[CLASS_NUM - 1:0],
input logic its_business_time,
input logic clk,
output logic [C_M00_AXIS_TDATA_WIDTH - 1:0] y
);
    logic signed [INDEX_LENGTH - 1:0] index [TREE_WITH - 1:0]; 
    logic signed [INDEX_LENGTH - 1:0] cmp_index [2 * TREE_WITH - 2:0]; 
    logic signed [WEIGHT_LENGTH - 1:0] cmp_result [2 * TREE_WITH - 2:0]; 
    integer sum = 0;
    generate
        genvar i,j;
        for (i = 0; i < TREE_WITH; i = i + 1) begin
            assign cmp_index[2 ** (INDEX_LENGTH + 1) - 2 - i] = i;
            if (i < CLASS_NUM) begin
                assign cmp_result[2 ** (INDEX_LENGTH + 1) - 2 - i] = c_sum[CLASS_NUM - 1 - i];
            end
            else begin
                assign cmp_result[2 ** (INDEX_LENGTH + 1) - 2 - i] = {1'b1,{(WEIGHT_LENGTH - 1){1'b0}}};
            end
        end
        for (i = 0; i < INDEX_LENGTH; i = i + 1) begin
            for (j = 0; j < 2 ** i; j = j + 1) begin
                compare #(
                .WEIGHT_LENGTH(WEIGHT_LENGTH),
                .INDEX_LENGTH(INDEX_LENGTH)
                )
                c(
                .a(cmp_result[2 * (2 ** (i) + j - 1) + 1]), 
                .a_i(cmp_index[2 * (2 ** (i) + j - 1) + 1]), 
                .b(cmp_result[2 * (2 ** (i) + j - 1) + 2]), 
                .b_i(cmp_index[2 * (2 ** (i) + j - 1) + 2]), 
                .c(cmp_result[2 ** (i) + j - 1]), 
                .c_i(cmp_index[2 ** (i) + j - 1])
                );
            end
        end
    endgenerate
    
    logic signed old_its_business_time;
    logic c_stat;
    
    initial begin
        c_stat = 0;
    end
    
	always @(posedge clk)begin 
        old_its_business_time <= its_business_time;
        if(its_business_time && !old_its_business_time)begin
            c_stat = 1;
        end
        else if (c_stat) begin
            c_stat = 0;
            y[INDEX_LENGTH - 1:0] = cmp_index[0]; 
            y[C_M00_AXIS_TDATA_WIDTH - 1:INDEX_LENGTH] = {(C_M00_AXIS_TDATA_WIDTH - INDEX_LENGTH){1'b0}};
        end
	end 
endmodule

