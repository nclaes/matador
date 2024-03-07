`timescale 1ns / 1ps


module HCB_top #(
	parameter CLASS_NUM,
	parameter PACKETS_NUM,
	parameter CLAUSE_NUM,
	parameter C_S00_AXIS_TDATA_WIDTH
	)
	(
	input clk,
	input [C_S00_AXIS_TDATA_WIDTH - 1:0] x,
	input [PACKETS_NUM - 1:0] valid,
	output [CLAUSE_NUM - 1:0] partial_clause [CLASS_NUM]
	);
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_0 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_1 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_2 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_3 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_4 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_5 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_6 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_7 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_8 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_9 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_10 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_11 [CLASS_NUM];
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_12 [CLASS_NUM];
	assign partial_clause = partial_clause_reg_12;
	integer i;
	initial begin
		for (i = 0; i < CLASS_NUM; i = i+1) begin
			partial_clause_reg_0[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_1[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_2[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_3[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_4[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_5[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_6[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_7[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_8[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_9[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_10[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_11[i] = {CLAUSE_NUM{1'b0}};
			partial_clause_reg_12[i] = {CLAUSE_NUM{1'b0}};
		end
	end
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
