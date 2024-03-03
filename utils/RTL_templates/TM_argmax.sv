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
    parameter integer INDEX_LENGTH = $clog2(CLASS_NUM - 1),
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
        end
	end 
endmodule


//module classify(c_sum, y, its_business_time, clk);

//    input logic its_business_time;
//    input logic clk; 
//	(*DONT_TOUCH = "TRUE"*)  input logic  signed [13:0] c_sum [10];
//	output logic [63:0] y;

//	logic signed [$clog2(10):0] a_0 = 0; 
//	logic signed [$clog2(10):0] a_1 = 1; 
//	logic signed [$clog2(10):0] a_2 = 2; 
//	logic signed [$clog2(10):0] a_3 = 3; 
//	logic signed [$clog2(10):0] a_4 = 4; 
//	logic signed [$clog2(10):0] a_5 = 5; 
//	logic signed [$clog2(10):0] a_6 = 6; 
//	logic signed [$clog2(10):0] a_7 = 7; 
//	logic signed [$clog2(10):0] a_8 = 8; 
//	logic signed [$clog2(10):0] a_9 = 9; 

//	// first level 
//	logic signed [13:0] a_0_1;
//	logic signed [$clog2(10):0] a_0_1_i; 

//	logic signed [13:0] a_2_3;
//	logic signed [$clog2(10):0] a_2_3_i;

//	logic signed [13:0] a_4_5; 
//	logic signed [$clog2(10):0] a_4_5_i; 

//	logic signed [13:0] a_6_7; 
//	logic signed [$clog2(10):0] a_6_7_i; 

//	logic signed [13:0] a_8_9; 
//	logic signed [$clog2(10):0] a_8_9_i; 

//	compare a_0_1_(.a(c_sum[0]), .a_i(a_0), .b(c_sum[1]), .b_i(a_1), .c(a_0_1), .c_i(a_0_1_i));
//	compare a_2_3_(.a(c_sum[2]), .a_i(a_2), .b(c_sum[3]), .b_i(a_3), .c(a_2_3), .c_i(a_2_3_i));
//	compare a_4_5_(.a(c_sum[4]), .a_i(a_4), .b(c_sum[5]), .b_i(a_5), .c(a_4_5), .c_i(a_4_5_i));
//	compare a_6_7_(.a(c_sum[6]), .a_i(a_6), .b(c_sum[7]), .b_i(a_7), .c(a_6_7), .c_i(a_6_7_i));
//	compare a_8_9_(.a(c_sum[8]), .a_i(a_8), .b(c_sum[9]), .b_i(a_9), .c(a_8_9), .c_i(a_8_9_i));
	
//	// second level 
//	logic signed [13:0] a_0_1_2_3; 
//	logic signed [$clog2(10):0] a_0_1_2_3_i; 

//	logic signed [13:0] a_4_5_6_7; 
//	logic signed [$clog2(10):0] a_4_5_6_7_i; 

//	compare a_0_1_2_3_(.a(a_0_1), .a_i(a_0_1_i), .b(a_2_3), .b_i(a_2_3_i), .c(a_0_1_2_3), .c_i(a_0_1_2_3_i));
//	compare a_4_5_6_7_(.a(a_4_5), .a_i(a_4_5_i), .b(a_6_7), .b_i(a_6_7_i), .c(a_4_5_6_7), .c_i(a_4_5_6_7_i));

//	// third level 
//	logic signed [13:0] a_0_1_2_3_8_9; 
//	(*DONT_TOUCH = "TRUE"*) logic signed [$clog2(10):0] a_0_1_2_3_8_9_i; 

//	compare a_0_1_2_3_8_9_(.a(a_0_1_2_3), .a_i(a_0_1_2_3_i), .b(a_8_9), .b_i(a_8_9_i), .c(a_0_1_2_3_8_9), .c_i(a_0_1_2_3_8_9_i));

//	// fourth level 

//	logic signed [13:0] a_0_1_2_3_8_9_4_5_6_7; 
//	logic signed [$clog2(10):0] a_0_1_2_3_8_9_4_5_6_7_i; 

//	compare a_0_1_2_3_8_9_4_5_6_7_(.a(a_0_1_2_3_8_9), .a_i(a_0_1_2_3_8_9_i), .b(a_4_5_6_7), .b_i(a_4_5_6_7_i), .c(a_0_1_2_3_8_9_4_5_6_7), .c_i(a_0_1_2_3_8_9_4_5_6_7_i));
    
//    logic signed old_its_business_time;
//    logic c_stat;
    
//    initial begin
//        c_stat = 0;
//    end
    
//	always @(posedge clk)begin 
//        old_its_business_time <= its_business_time;
//        if(its_business_time && !old_its_business_time)begin
//            c_stat = 1;
//        end
//        else if (c_stat) begin
//            c_stat = 0;
//            y =  a_0_1_2_3_8_9_4_5_6_7_i; 
//        end
//	end 
	
//endmodule


