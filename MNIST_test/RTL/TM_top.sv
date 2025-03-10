`timescale 1ns / 1ps

module Hard_Coded_Inference_Top #(

		parameter STAGE_NUM,
	parameter CLAUSE_NUM,
	parameter CLASS_NUM,
	parameter WEIGHT_LENGTH,
	parameter C_S00_AXIS_TDATA_WIDTH,
	parameter C_M00_AXIS_TDATA_WIDTH,
	parameter PACKETS_NUM
)
(x, y, packet_counter, valid, s_axis_tready, clk, rst, adder_en, adder_done, argmax, finish, last, last_out,
clauses, class_sums, m00_axis_tready, m00_axis_tkeep);
	input logic clk;
	input  logic rst;
	output logic finish;
	output logic [CLAUSE_NUM - 1:0] clauses [CLASS_NUM - 1:0];
	output logic signed [WEIGHT_LENGTH - 1:0] class_sums [CLASS_NUM];
	input logic s_axis_tready;
	input logic m00_axis_tready;
	output logic [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep;
	input logic [C_S00_AXIS_TDATA_WIDTH - 1:0] x;
	input logic valid;
	input logic last;
	output logic last_out;
	output logic adder_en;
	output logic adder_done;
	output logic argmax;
	input logic [C_M00_AXIS_TDATA_WIDTH-1:0] packet_counter;
	output logic [C_M00_AXIS_TDATA_WIDTH-1:0] y;
		
		

	logic [CLAUSE_NUM - 1:0] partial_clause_reg_0;
	logic valid_reg_0;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_1;
	logic valid_reg_1;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_2;
	logic valid_reg_2;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_3;
	logic valid_reg_3;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_4;
	logic valid_reg_4;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_5;
	logic valid_reg_5;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_6;
	logic valid_reg_6;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_7;
	logic valid_reg_7;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_8;
	logic valid_reg_8;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_9;
	logic valid_reg_9;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_10;
	logic valid_reg_10;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_11;
	logic valid_reg_11;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg_12;
	logic valid_reg_12;
	logic valid_reg_13;
	logic adder;
	logic adder_2;
	logic argmax;
	logic argmax_reset;
	logic delay_1;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg [CLASS_NUM - 1:0];

	assign clauses = partial_clause_reg;
	initial begin
		valid_reg_0 = 1'b0;
		valid_reg_1 = 1'b0;
		valid_reg_2 = 1'b0;
		valid_reg_3 = 1'b0;
		valid_reg_4 = 1'b0;
		valid_reg_5 = 1'b0;
		valid_reg_6 = 1'b0;
		valid_reg_7 = 1'b0;
		valid_reg_8 = 1'b0;
		valid_reg_9 = 1'b0;
		valid_reg_10 = 1'b0;
		valid_reg_11 = 1'b0;
		valid_reg_12 = 1'b0;
		valid_reg_13 = 1'b0;
		(*DONT_TOUCH = "TRUE"*) argmax = 1'b0;
		adder_2 = 1'b0;
		(*DONT_TOUCH = "TRUE"*) delay_1 = 1'b0;
		y = {64'b0};
		partial_clause_reg_0 = {200'b0};
		partial_clause_reg_1 = {200'b0};
		partial_clause_reg_2 = {200'b0};
		partial_clause_reg_3 = {200'b0};
		partial_clause_reg_4 = {200'b0};
		partial_clause_reg_5 = {200'b0};
		partial_clause_reg_6 = {200'b0};
		partial_clause_reg_7 = {200'b0};
		partial_clause_reg_8 = {200'b0};
		partial_clause_reg_9 = {200'b0};
		partial_clause_reg_10 = {200'b0};
		partial_clause_reg_11 = {200'b0};
		partial_clause_reg_12 = {200'b0};
	end

	HCB_top #(
		.CLASS_NUM(CLASS_NUM),
		.PACKETS_NUM(PACKETS_NUM),
		.CLAUSE_NUM(CLAUSE_NUM),
		.C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH)
	)
	HT(.clk(clk),
	    .rst(rst),
		.x(x),
		.valid(valid),
		.HCB_done(HCB_done),
		.partial_clause(partial_clause_reg));

	logic adder_done;
	logic adder_en;

    //always @(posedge clk) begin 
    //end
    reg old_adder_valid;
	always @(posedge clk) begin
//	   if (m00_axis_tready) begin
	       adder_en <= HCB_done;
	       argmax <= adder_done && m00_axis_tready;
//	   end
	end
	
    logic signed [WEIGHT_LENGTH - 1:0] class_sums_co [CLASS_NUM];
    logic signed [WEIGHT_LENGTH - 1:0] class_sums_org [CLASS_NUM];
    
    logic adder_done_co,adder_done_org;

 	Adder_new #(
 	.STAGE_NUM(STAGE_NUM),
    .CLAUSE_NUM(CLAUSE_NUM),
    .CLASS_NUM(CLASS_NUM),
    .WEIGHT_LENGTH(WEIGHT_LENGTH)
 	)add_inst
 	(
 		.clk(clk),
 		.clauses(partial_clause_reg),
 		.class_sums(class_sums),
 		.valid(adder),
 		.adder_done(adder_done)
 	);
    
	classify #(
    .CLASS_NUM(CLASS_NUM),    
    .WEIGHT_LENGTH(WEIGHT_LENGTH),
    .C_M00_AXIS_TDATA_WIDTH(C_M00_AXIS_TDATA_WIDTH)
	)
	classify_inst (
		.c_sum(class_sums),
		.y(y),
		.last(last),
		.m00_axis_tready(m00_axis_tready),
		.m00_axis_tkeep(m00_axis_tkeep),
		.its_business_time(argmax),
		.finish(finish),
		.last_out(last_out),
		.clk(clk),
		.rst(rst)
	);
	
endmodule

