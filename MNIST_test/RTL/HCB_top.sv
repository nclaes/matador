`timescale 1ns / 1ps


module HCB_top #(
	parameter CLASS_NUM,
	parameter PACKETS_NUM,
	parameter CLAUSE_NUM,
	parameter C_S00_AXIS_TDATA_WIDTH
	)
	(
	input clk,
	input rst,
	input [C_S00_AXIS_TDATA_WIDTH - 1:0] x,
	input valid,
	input HCB_done,
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


	    //shift register
    logic [PACKETS_NUM - 1:0] HT_en;
    logic [PACKETS_NUM - 1:0] HT_en_ctrl;
    
    initial begin
        HT_en = '0;
        HT_en[0] = 1'b1;
        HT_en_ctrl = '0;
    end
    
    always@(posedge clk) begin
        if (rst) begin
            HT_en = '0;
            HT_en[0] = 1'b1;
        end
        else begin
            if (valid) begin
                if (PACKETS_NUM == 1) begin
                    
                end
                else begin
                    HT_en = {{HT_en[PACKETS_NUM - 2:0]},{HT_en[PACKETS_NUM - 1]}};
                end
            end
        end
    end
    
    assign HT_en_ctrl = HT_en & {PACKETS_NUM{valid}};
    assign HCB_done = HT_en_ctrl[PACKETS_NUM - 1];

	HCB_0 HCB_inst_0(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[0]),
		.partial_clause(partial_clause_reg_0)
	);

	HCB_1 HCB_inst_1(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[1]),
		.partial_clause_prev(partial_clause_reg_0),
		.partial_clause(partial_clause_reg_1)
	);

	HCB_2 HCB_inst_2(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[2]),
		.partial_clause_prev(partial_clause_reg_1),
		.partial_clause(partial_clause_reg_2)
	);

	HCB_3 HCB_inst_3(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[3]),
		.partial_clause_prev(partial_clause_reg_2),
		.partial_clause(partial_clause_reg_3)
	);

	HCB_4 HCB_inst_4(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[4]),
		.partial_clause_prev(partial_clause_reg_3),
		.partial_clause(partial_clause_reg_4)
	);

	HCB_5 HCB_inst_5(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[5]),
		.partial_clause_prev(partial_clause_reg_4),
		.partial_clause(partial_clause_reg_5)
	);

	HCB_6 HCB_inst_6(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[6]),
		.partial_clause_prev(partial_clause_reg_5),
		.partial_clause(partial_clause_reg_6)
	);

	HCB_7 HCB_inst_7(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[7]),
		.partial_clause_prev(partial_clause_reg_6),
		.partial_clause(partial_clause_reg_7)
	);

	HCB_8 HCB_inst_8(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[8]),
		.partial_clause_prev(partial_clause_reg_7),
		.partial_clause(partial_clause_reg_8)
	);

	HCB_9 HCB_inst_9(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[9]),
		.partial_clause_prev(partial_clause_reg_8),
		.partial_clause(partial_clause_reg_9)
	);

	HCB_10 HCB_inst_10(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[10]),
		.partial_clause_prev(partial_clause_reg_9),
		.partial_clause(partial_clause_reg_10)
	);

	HCB_11 HCB_inst_11(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[11]),
		.partial_clause_prev(partial_clause_reg_10),
		.partial_clause(partial_clause_reg_11)
	);

	HCB_12 HCB_inst_12(
		.clk(clk),
		.x(x),
		.valid(HT_en_ctrl[12]),
		.partial_clause_prev(partial_clause_reg_11),
		.partial_clause(partial_clause_reg_12)
	);

endmodule
