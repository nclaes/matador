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
(x, y, packet_counter, valid, s_axis_tready, clk, finish, last, last_out,
clauses, class_sums, m00_axis_tready);
	input logic clk;
	output logic finish;
	output logic [CLAUSE_NUM - 1:0] clauses;
	output logic signed [WEIGHT_LENGTH - 1:0] class_sums [10];
	input logic s_axis_tready;
	input logic m00_axis_tready;
	input logic [C_M00_AXIS_TDATA_WIDTH:0] x;
	input logic [PACKETS_NUM - 1:0] valid;
	input logic last;
	output logic last_out;
	input logic [C_M00_AXIS_TDATA_WIDTH:0] packet_counter;
	output logic [C_M00_AXIS_TDATA_WIDTH:0] y;

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
	logic finished;
	logic delay_1;
	logic last_registered;
	assign finish = finished;
	logic [CLAUSE_NUM - 1:0] partial_clause_reg;

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
		last_registered = 1'b0;
		last_out = 1'b0;
		(*DONT_TOUCH = "TRUE"*) argmax = 1'b0;
		(*DONT_TOUCH = "TRUE"*) adder  = 1'b0;
		adder_2 = 1'b0;
		(*DONT_TOUCH = "TRUE"*) delay_1 = 1'b0;
		(*DONT_TOUCH = "TRUE"*) delay_1 = 1'b0;
		y = {64'b0};
		partial_clause_reg_0 = {100'b0};
		partial_clause_reg_1 = {100'b0};
		partial_clause_reg_2 = {100'b0};
		partial_clause_reg_3 = {100'b0};
		partial_clause_reg_4 = {100'b0};
		partial_clause_reg_5 = {100'b0};
		partial_clause_reg_6 = {100'b0};
		partial_clause_reg_7 = {100'b0};
		partial_clause_reg_8 = {100'b0};
		partial_clause_reg_9 = {100'b0};
		partial_clause_reg_10 = {100'b0};
		partial_clause_reg_11 = {100'b0};
		partial_clause_reg_12 = {100'b0};
	end

	HCB_top #(
		.CLASS_NUM(CLASS_NUM),
		.PACKETS_NUM(PACKETS_NUM),
		.CLAUSE_NUM(CLAUSE_NUM),
		.C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH)
	)
	HT(.clk(clk),
		.x(x),
		.valid(valid), 
		.partial_clause(partial_clause_reg));

	logic adder_done;


    reg old_adder_valid;
	always @(posedge clk) begin
	   old_adder_valid <= valid[12];
	   if(last) begin 
            last_registered = 1'b1;
       end
	   if(valid[12] && !old_adder_valid) begin
	       adder = 1; 
	   end
	   else if(adder_done && adder) begin
	       adder = 0; 
	       argmax = 1; 
	   end 
	   else if(argmax && m00_axis_tready) begin 
	       argmax = 0;
	       finished = 1;
	       if(last_registered) begin 
	           last_out = 1'b1;
	           last_registered =1'b0;
	       end 
	   end
	   else if(finished) begin
	       finished = 0;
	       if(last_out) begin 
	           last_out = 1'b0;
	       end
	   end
	end
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
	//.CLAUSE_NUM(CLAUSE_NUM),
    .CLASS_NUM(CLASS_NUM),    
    .WEIGHT_LENGTH(WEIGHT_LENGTH),
    .C_M00_AXIS_TDATA_WIDTH(C_M00_AXIS_TDATA_WIDTH)
	)
	classify_inst (
		.c_sum(class_sums),
		.y(y),
		.its_business_time(argmax),
		.clk(clk)
	);
	
endmodule
