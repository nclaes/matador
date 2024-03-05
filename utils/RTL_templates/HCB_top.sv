`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/30/2023 01:27:25 PM
// Design Name: 
// Module Name: HCB_top
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


module HCB_top #(
    parameter PACKETS_NUM,
    parameter CLAUSE_NUM,
    parameter C_S00_AXIS_TDATA_WIDTH
    )
    (
    input clk,
    input [C_S00_AXIS_TDATA_WIDTH - 1:0] x,
    input [PACKETS_NUM - 1:0] valid,
    output [CLAUSE_NUM - 1:0] partial_clause
    );
    logic [CLAUSE_NUM - 1:0] partial_clause_reg_0;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_1;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_2;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_3;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_4;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_5;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_6;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_7;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_8;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_9;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_10;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_11;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_12;
       initial begin
           partial_clause_reg_0  = {200'b0};
           partial_clause_reg_1  = {200'b0};
           partial_clause_reg_2  = {200'b0};
           partial_clause_reg_3  = {200'b0};
           partial_clause_reg_4  = {200'b0};
           partial_clause_reg_5  = {200'b0};
           partial_clause_reg_6  = {200'b0}; 
           partial_clause_reg_7  = {200'b0};
           partial_clause_reg_8  = {200'b0};
           partial_clause_reg_9  = {200'b0};
           partial_clause_reg_10 = {200'b0};
           partial_clause_reg_11 = {200'b0};
           partial_clause_reg_12 = {200'b0}; 
       end
    
    assign partial_clause = partial_clause_reg_12;
    
    HCB_0 HCB_inst_0(
		.clk(clk),
		.x(x),
		.valid(valid[0]),
		.partial_clause(partial_clause_reg_0)
	);

	HCB_1 HCB_inst_1(
		.clk(clk),
		.x(x),
		.valid(valid[1]),
		.partial_clause_prev(partial_clause_reg_0),
		.partial_clause(partial_clause_reg_1)
	);

	HCB_2 HCB_inst_2(
		.clk(clk),
		.x(x),
		.valid(valid[2]),
		.partial_clause_prev(partial_clause_reg_1),
		.partial_clause(partial_clause_reg_2)
	);

	HCB_3 HCB_inst_3(
		.clk(clk),
		.x(x),
		.valid(valid[3]),
		.partial_clause_prev(partial_clause_reg_2),
		.partial_clause(partial_clause_reg_3)
	);

	HCB_4 HCB_inst_4(
		.clk(clk),
		.x(x),
		.valid(valid[4]),
		.partial_clause_prev(partial_clause_reg_3),
		.partial_clause(partial_clause_reg_4)
	);

	HCB_5 HCB_inst_5(
		.clk(clk),
		.x(x),
		.valid(valid[5]),
		.partial_clause_prev(partial_clause_reg_4),
		.partial_clause(partial_clause_reg_5)
	);

	HCB_6 HCB_inst_6(
		.clk(clk),
		.x(x),
		.valid(valid[6]),
		.partial_clause_prev(partial_clause_reg_5),
		.partial_clause(partial_clause_reg_6)
	);

	HCB_7 HCB_inst_7(
		.clk(clk),
		.x(x),
		.valid(valid[7]),
		.partial_clause_prev(partial_clause_reg_6),
		.partial_clause(partial_clause_reg_7)
	);

	HCB_8 HCB_inst_8(
		.clk(clk),
		.x(x),
		.valid(valid[8]),
		.partial_clause_prev(partial_clause_reg_7),
		.partial_clause(partial_clause_reg_8)
	);

	HCB_9 HCB_inst_9(
		.clk(clk),
		.x(x),
		.valid(valid[9]),
		.partial_clause_prev(partial_clause_reg_8),
		.partial_clause(partial_clause_reg_9)
	);

	HCB_10 HCB_inst_10(
		.clk(clk),
		.x(x),
		.valid(valid[10]),
		.partial_clause_prev(partial_clause_reg_9),
		.partial_clause(partial_clause_reg_10)
	);

	HCB_11 HCB_inst_11(
		.clk(clk),
		.x(x),
		.valid(valid[11]),
		.partial_clause_prev(partial_clause_reg_10),
		.partial_clause(partial_clause_reg_11)
	);

	HCB_12 HCB_inst_12(
		.clk(clk),
		.x(x),
		.valid(valid[12]),
		.partial_clause_prev(partial_clause_reg_11),
		.partial_clause(partial_clause_reg_12)
	);
endmodule
