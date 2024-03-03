`timescale 1ns / 1ps                                                                 
//////////////////////////////////////////////////////////////////////////////////   
// Company:                                                                          
// Engineer:                                                                         
//                                                                                   
// Create Date: 06/24/2023 03:24:09 AM                                               
// Design Name: Tousif Rahman, Gang Mao                                              
// Module Name: Hard_Coded_Inference_Top                                                        
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
                                                                                     
                                                                                     

module Hard_Coded_Inference_Top #(
    parameter STAGE_NUM,
    parameter CLAUSE_NUM,
    parameter CLASS_NUM,
    parameter WEIGHT_LENGTH,
    parameter C_M00_AXIS_TDATA_WIDTH
)
(x, y, packet_counter,valid,s_axis_tready, clk, finish, last, last_out, clauses, class_sums, m00_axis_tready);
	input logic clk;
	output logic finish; 
	output logic [CLAUSE_NUM - 1:0] clauses;
	output logic signed [WEIGHT_LENGTH-1:0] class_sums [CLASS_NUM];
	input logic s_axis_tready;
	input logic m00_axis_tready;
	input logic [C_M00_AXIS_TDATA_WIDTH-1:0] x;
	input logic [12:0] valid;
	input logic last;
	output logic last_out;
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
    logic finished; 
    logic delay_1;
    logic last_registered;
    assign finish = finished;
    assign clauses = partial_clause_reg_12;
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
       (*DONT_TOUCH = "TRUE"*) finished = 1'b0;
  
       y = {64'b0};
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
	
    wire adder_done;
    
    always @(posedge clk) begin
//        if(last) begin 
//            last_registered = 1'b1;
//        end
    end
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
//       else if (adder_done) begin
//           adder = 0;
//           argmax = 1;
//       end
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

//		//if (s_axis_tready) begin
//		  case(packet_counter)
//			'd0 : begin 
//				valid_reg_0 = 1;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd1 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 1;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd2 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 1;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd3 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 1;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd4 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 1;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd5 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 1;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd6 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 1;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd7 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 1;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd8 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 1;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd9 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 1;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd10 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 1;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd11 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 1;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			'd12 : begin 
//				valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 1;
//				valid_reg_13 = 0;
//			end
//			default: begin
//			    valid_reg_0 = 0;
//				valid_reg_1 = 0;
//				valid_reg_2 = 0;
//				valid_reg_3 = 0;
//				valid_reg_4 = 0;
//				valid_reg_5 = 0;
//				valid_reg_6 = 0;
//				valid_reg_7 = 0;
//				valid_reg_8 = 0;
//				valid_reg_9 = 0;
//				valid_reg_10 = 0;
//				valid_reg_11 = 0;
//				valid_reg_12 = 0;
//				valid_reg_13 = 0;
//			end
//			endcase
//		//end
////		else begin
////		  valid_reg_0 = 0;
////				valid_reg_1 = 0;
////				valid_reg_2 = 0;
////				valid_reg_3 = 0;
////				valid_reg_4 = 0;
////				valid_reg_5 = 0;
////				valid_reg_6 = 0;
////				valid_reg_7 = 0;
////				valid_reg_8 = 0;
////				valid_reg_9 = 0;
////				valid_reg_10 = 0;
////				valid_reg_11 = 0;
////				valid_reg_12 = 0;
////				valid_reg_13 = 0;
////		end
			
		

	end
	Adder_new #(
	.STAGE_NUM(STAGE_NUM),
    .CLAUSE_NUM(CLAUSE_NUM),
    .CLASS_NUM(CLASS_NUM)
	)add_inst
	(
		.clk(clk),
		.clauses(partial_clause_reg_12),
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
