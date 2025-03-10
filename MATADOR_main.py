import json
import os
import numpy as np
from math import ceil 
from math import log
from utils.train import prep_data, checkconfig
from utils.tmu.models.classification.vanilla_classifier import TMClassifier
from utils.tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from utils.tmu.tools import BenchmarkTimer
# from utils.rtl import coalesced_tm_write_hard_coded_blocks, coalesced_tm_hard_coded_blocks_top

def to_bin(val, bits):
    s = bin(val & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def write_weights(filename, bits_required, weights, classes, clauses): 

	with open(filename, "w") as f: 
		# top level module for feeding in the weights and class_sums per class
		print("module hard_coded_weight #(", file=f)
		print("\tparameter CLAUSE_NUM", file=f)
		print("\t)", file=f)
		print("\t(", file=f)
		print("\toutput logic signed [%d:0] weights [%d][CLAUSE_NUM]); " %(bits_required-1, classes), file=f)
		print("", file=f)
		for i in range(classes):
			for j in range(clauses):
				print("\tassign weights[%d][%d] \t=\t%d'b%s;" %(i, j, bits_required,to_bin(weights[i][j], bits_required)), file=f)

def get_bits_required(Weights_file, clauses, classes):
    Weights = []
    W_file = open(Weights_file, "r")
    data = W_file.read()
    data_into_list = data.split("\n")
    Weights = [list(map(int, i.split())) for i in data_into_list if i]  # Split each line into individual integers
    Weights = np.array(Weights)
    Weights = np.reshape(Weights, (classes, clauses))
    print(Weights.shape)

    max_positive = 0 
    max_negative = 0 
    max_positive_current = 0 
    max_negative_current = 0  

    for i in range(Weights.shape[0]):
        # classes
        for j in range(Weights.shape[1]):
            #clauses
            if Weights[i][j] > 0: 
                max_positive_current += Weights[i][j]
            else:
                max_negative_current += Weights[i][j]

        if max_positive_current > max_positive: 
            max_positive = max_positive_current

        if max_negative_current < max_negative: 
            max_negative = max_negative_current

        max_negative_current = 0 
        max_positive_current = 0 

    print("Max Positive Weight: " , max_positive)
    print("Max Negative Weight: ", max_negative)

    max_pos = abs(max_positive)
    max_neg = abs(max_negative) 
    bits = 0 

    if max_pos > max_neg: 
        abs_w = max_pos
        bits = ceil(log(abs_w, 2))
    else: 
        abs_w = max_neg 
        bits = ceil(log(abs_w, 2)) + 1

    print("bits required: ",  bits)
    return bits, Weights 

def coalesced_tm_write_axis_wrapper(axis_wrapper_f, AXI_data_width, number_of_blocks, adder_stages, clauses, classes, features):
	with open(axis_wrapper_f, "w") as f:
		print("module axis_wrapper_top #", file=f)
		print("\t(", file=f)
		print("\t\tparameter integer DEPTH = 1,", file=f)
		print("\t\tparameter integer WIDTH = 1,", file=f)
		print("\t\tparameter integer PACKETS = %d," %(number_of_blocks), file=f)
		print("\t\tparameter integer C_S00_AXIS_TDATA_WIDTH = %d," %(AXI_data_width), file=f)
		print("\t\tparameter integer C_M00_AXIS_TDATA_WIDTH = %d," %(AXI_data_width), file=f)
		print("\t\t// design configurations", file=f)
		print("\t\tparameter STAGE_NUM = %d," %(adder_stages), file=f)
		print("\t\tparameter CLAUSE_NUM = %d," %(clauses), file=f)
		print("\t\tparameter CLASS_NUM = %d," %(classes), file=f)
		# in this clase the weight length is not so useful - its not for the weights themselves
		# this is used as how many bits to represent the class sum 
		bits_required = int(ceil(log(clauses, 2)) + 1)
		print("\t\tparameter WEIGHT_LENGTH = %d," %(bits_required), file=f)
		print("\t\tparameter FEATURE_NUM = %d," %(features), file=f)
		print("\t\tparameter PACKETS_NUM = (FEATURE_NUM - 1)/C_S00_AXIS_TDATA_WIDTH + 1,", file=f)
		print("\t\tparameter VANILLA = 0,", file=f)
		print("\t\tparameter COALESCED = 1", file=f)
		print("\t)", file=f)
		print("\t(", file=f)
		print("""

		// Ports of Axi Slave Bus Interface S00_AXIS
		input  wire    s00_axis_aclk,
		input  wire    s00_axis_aresetn,
		input  wire    [C_S00_AXIS_TDATA_WIDTH-1 : 0] s00_axis_tdata,
		input  wire    [(C_S00_AXIS_TDATA_WIDTH/8)-1 : 0] s00_axis_tstrb,
		input  wire    s00_axis_tlast,
		input  wire    s00_axis_tvalid,
		output wire    s00_axis_tready,
		//test ports
		output wire [PACKETS-1:0] valid_reg,
	    output wire [C_S00_AXIS_TDATA_WIDTH-1:0] axis2pipe_data,
        output wire [CLAUSE_NUM - 1:0] clauses [CLASS_NUM - 1:0],
	    output logic signed [WEIGHT_LENGTH-1:0] class_sums [CLASS_NUM],
	    output logic adder_en,
        output logic adder_done,
        output logic argmax,
		output reg 	[31:0] flag_out,
		// Ports of Axi Master Bus Interface M00_AXIS
		input  wire  m00_axis_aclk,
		input  wire  m00_axis_aresetn,
		input  wire  m00_axis_tready,
		output wire  m00_axis_tvalid,
		output reg   [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep,
		output wire  [C_M00_AXIS_TDATA_WIDTH-1 : 0] m00_axis_tdata,
		output wire  [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tstrb,
		output wire  m00_axis_tlast
    );
    
   	wire axis2pipe_tvalid, axis2pipe_tready, axis2pipe_tlast;
	wire pipe2axis_tvalid, pipe2axis_tready, pipe2axis_tlast;
	wire [C_M00_AXIS_TDATA_WIDTH-1:0] pipe2axis_data;
		
	
		logic [12:0] packet_counter;


	logic full;
	logic inference_complete;
	logic last_registered; 
	logic last_complete;

    reg old_inference_complete,old_old_inference_complete;
    reg old_s00_axis_tlast,old_last_complete,old_old_last_complete;	

	assign m00_axis_tvalid = inference_complete;
	assign m00_axis_tlast = last_complete;	
	
	reg old_s00_axis_tready;
	reg plus_stat;
	initial begin
//	   m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b0}};
	   packet_counter = 0;
	end

	always @(posedge m00_axis_aclk) begin
//	   old_inference_complete <= inference_complete;
//	   old_old_inference_complete <= old_inference_complete;
//	   old_s00_axis_tready <= s00_axis_tready;
//	   old_s00_axis_tlast <= s00_axis_tlast;
//	   old_last_complete <= last_complete;
//	   old_old_last_complete <= old_last_complete;
//	   if(!s00_axis_aresetn)begin 
//	       packet_counter = 0;
//	       plus_stat = 0;
//	   end
//        if (inference_complete) begin
//	       packet_counter = 0;
//	   end
//	   if(packet_counter == PACKETS - 1) begin 
//	       packet_counter = 0; 
//	   end
//	   else if (s00_axis_tlast && !old_s00_axis_tlast) begin
//	       plus_stat = 0;
//	   end
//	   else if(axis2pipe_tready && s00_axis_tready && old_s00_axis_tready) begin 
//	       if(plus_stat) begin 
//	           packet_counter = packet_counter + 1;
//	       end
//	       else begin
//	       end
//	   end
	   
//	   if (inference_complete && !last_complete) begin
//	       m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b1}};
//	   end
//	   else if (!old_old_last_complete && old_last_complete) begin
//	       m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b0}};
//	   end
	end
    assign axis2pipe_tready = m00_axis_tready;

	Hard_Coded_Inference_Top #(
	.STAGE_NUM(STAGE_NUM),
    .CLAUSE_NUM(CLAUSE_NUM),
    .CLASS_NUM(CLASS_NUM),
    .WEIGHT_LENGTH(WEIGHT_LENGTH),
    .C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH),
    .C_M00_AXIS_TDATA_WIDTH(C_M00_AXIS_TDATA_WIDTH),
    .PACKETS_NUM(PACKETS_NUM)
	)
	tm(
	   .x(axis2pipe_data),
	   .clk(m00_axis_aclk),
	   .rst(~s00_axis_aresetn),
	   .valid(valid),
	   .s_axis_tready(s00_axis_tready),
	   .m00_axis_tready(axis2pipe_tready),
	   .m00_axis_tkeep(m00_axis_tkeep),
	   .packet_counter(packet_counter),
	   .y(m00_axis_tdata),
	   //test ports 
	   .clauses(clauses),
	   .class_sums(class_sums),
	   .adder_en(adder_en),
	   .adder_done(adder_done),
	   .argmax(argmax),
	   .finish(inference_complete),
	   .last(s00_axis_tlast),
	   .last_out(last_complete)
	);
	
	// Instantiation of Axi Bus Interface S00_AXIS
	axis_adder_v1_0_S00_AXIS #(
	   .DATA_WIDTH(C_S00_AXIS_TDATA_WIDTH),
	   .PACKETS_NUM(PACKETS_NUM)
	)
	 axis_adder_v1_0_S00_AXIS_inst (
		.clk(s00_axis_aclk),
		.rst(~s00_axis_aresetn),
		.s_axis_tdata(s00_axis_tdata),
		.s_axis_tvalid(s00_axis_tvalid),
		.s_axis_tready(s00_axis_tready),
		.s_axis_tlast(s00_axis_tlast),
		.valid(valid),
		//.ptr_reg(ptr_reg),
		.full(full),
		.m_axis_tdata(axis2pipe_data),
		.m_axis_tvalid(axis2pipe_tvalid),
		.m_axis_tready(axis2pipe_tready)
	); 
	
endmodule

""", file=f)


def coalesced_tm_write_top(filename, number_of_blocks, AXI_data_width, clauses):
	with open(filename, "w") as f:
		print("`timescale 1ns / 1ps", file=f)
		print("", file=f)
		print("module Hard_Coded_Inference_Top #(", file=f)
		print("""
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
		
		""", file=f)
		# OLD VERISON ----------------------------
		# print("\tparameter STAGE_NUM,", file=f)
		# print("\tparameter CLAUSE_NUM,", file=f)
		# print("\tparameter CLASS_NUM,", file=f)
		# print("\tparameter WEIGHT_LENGTH,", file=f)
		# print("\tparameter C_S00_AXIS_TDATA_WIDTH,", file=f)
		# print("\tparameter C_M00_AXIS_TDATA_WIDTH,", file=f)
		# print("\tparameter PACKETS_NUM", file=f)
		# print(")", file=f)

		# print("(x, y, packet_counter, valid, s_axis_tready, clk, rst, finish, last, last_out,", file=f)
		# print("clauses, class_sums, m00_axis_tready, m00_axis_tkeep);", file=f)
		# print("\tinput logic clk;", file=f)
		# print("\tinput logic rst;", file=f)
		# print("\toutput logic finish;", file=f)
		# print("\toutput logic [CLAUSE_NUM - 1:0] clauses [CLASS_NUM - 1:0];", file=f)
		# print("\toutput logic signed [WEIGHT_LENGTH - 1:0] class_sums [%d];" %(classes), file=f)
		# print("\tinput logic s_axis_tready;", file=f)
		# print("\tinput logic m00_axis_tready;", file=f)
		# print("\toutput logic [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep;")
		# print("\tinput logic [C_M00_AXIS_TDATA_WIDTH-1:0] x;", file=f)
		# print("\tinput logic [PACKETS_NUM - 1:0] valid;", file=f)
		# print("\tinput logic last;", file=f)
		# print("\toutput logic last_out;", file=f)
		# print("\toutput logic adder_en;", file=f)
		# print("\tadder_done;", file=f)
		# print("\targmax")
		# print("\tinput logic [C_M00_AXIS_TDATA_WIDTH-1:0] packet_counter;", file=f)
		# print("\toutput logic [C_M00_AXIS_TDATA_WIDTH-1:0] y;", file=f)
		# OLD VERISON ----------------------------

		print("", file=f)
		for i in range(number_of_blocks):
			print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg_%d;" %(i), file=f)
			print("\tlogic valid_reg_%d;" %(i), file=f)
		print("\tlogic valid_reg_%d;" %(number_of_blocks), file=f)
		
		print("\tlogic adder;", file=f)
		print("\tlogic adder_2;", file=f)
		print("\tlogic argmax;", file=f)
		print("\tlogic argmax_reset;", file=f)
		# print("\tlogic finished;", file=f)
		# print("\tlogic finished_reset;", file=f)
		print("\tlogic delay_1;", file=f)
		# print("\tlogic last_registered;", file=f)
		# print("\tassign finish = finished;")

		print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg [CLASS_NUM - 1:0];", file=f)
		print("", file=f)

		print("\tassign clauses = partial_clause_reg;", file=f)
		print("\tinitial begin", file=f)
		for i in range(number_of_blocks):
			print("\t\tvalid_reg_%d = 1'b0;" %(i), file=f)

		print("\t\tvalid_reg_%d = 1'b0;" %(number_of_blocks), file=f)
		# print("\t\tlast_registered = 1'b0;", file=f)
		# print("\t\tlast_out = 1'b0;", file=f)

		print("\t\t(*DONT_TOUCH = \"TRUE\"*) argmax = 1'b0;", file=f)
		# print("\t\t(*DONT_TOUCH = \"TRUE\"*) adder = 1'b0;", file=f)
		print("\t\tadder_2 = 1'b0;", file=f)
		# print("\t\tadder  = 1'b0;", file=f)
		print("\t\t(*DONT_TOUCH = \"TRUE\"*) delay_1 = 1'b0;", file=f)
		# print("\t\t(*DONT_TOUCH = \"TRUE\"*) finished = 1'b0;", file=f)

		print("\t\ty = {%d'b0};" %(AXI_data_width), file=f)
		for i in range(number_of_blocks):
			print("\t\tpartial_clause_reg_%d = {%d'b0};" %(i, clauses), file=f)
		# print("\t\tfinished = 0;", file=f)
		# print("\t\tfinish_reset = 0;", file=f)
		print("\tend", file=f)

		print("""
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
""", file=f)



def coalesced_tm_hard_coded_blocks_top(filename, number_of_blocks):
	with open(filename, "w") as f:
		print("`timescale 1ns / 1ps", file=f)
		print("", file=f)
		print("", file=f)
		print("module HCB_top #(", file=f)
		print("\tparameter CLASS_NUM,", file=f)
		print("\tparameter PACKETS_NUM,", file=f)
		print("\tparameter CLAUSE_NUM,", file=f)
		print("\tparameter C_S00_AXIS_TDATA_WIDTH", file=f)
		print("\t)", file=f)
		print("\t(", file=f)
		print("\tinput clk,", file=f)
		print("\tinput rst,", file=f)
		print("\tinput [C_S00_AXIS_TDATA_WIDTH - 1:0] x,", file=f)
		print("\tinput valid,", file=f)
		print("\tinput HCB_done,", file=f)
		print("\toutput [CLAUSE_NUM - 1:0] partial_clause [CLASS_NUM]", file=f)
		print("\t);", file=f)

		for i in range(number_of_blocks):
			print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg_%d [CLASS_NUM];"%(i), file=f)

		print("\tassign partial_clause = partial_clause_reg_%d;" %(number_of_blocks-1), file=f)
		print("\tinteger i;", file=f)

		print("\tinitial begin", file=f)
		print("\t\tfor (i = 0; i < CLASS_NUM; i = i+1) begin", file=f)
		for i in range(number_of_blocks):
			print("\t\t\tpartial_clause_reg_%d[i] = {CLAUSE_NUM{1'b0}};"%(i), file=f)
		print("\t\tend", file=f)
		print("\tend", file=f)
		# print("\tassign partial_clause = partial_clause_reg_%d;"%(i) ,file=f)
		print("""

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
    assign HCB_done = HT_en_ctrl[PACKETS_NUM - 1];\n""", file=f)

		for i in range(number_of_blocks):
			print("\tHCB_%d HCB_inst_%d(" %(i, i), file=f)
			print("\t\t.clk(clk),", file=f)
			print("\t\t.x(x),", file=f)
			print("\t\t.valid(HT_en_ctrl[%d]),"%(i), file=f)
			if(i == 0):
				print("\t\t.partial_clause(partial_clause_reg_%d)" %(i), file=f)
			else:
				print("\t\t.partial_clause_prev(partial_clause_reg_%d),"%(i-1), file=f)
				print("\t\t.partial_clause(partial_clause_reg_%d)"%(i), file=f)
			print("\t);", file=f)
			print("", file=f)
		print("endmodule", file=f)

def coalesced_tm_write_hard_coded_blocks(number_of_blocks, TAs, Weights, file_name, bus_width, output_directory, clauses, features):
    starting_block = 0 
    # because the data is x and ~x so we can fit both
    finish_block = bus_width*2

    # The raw clause expressions can be written for visual
    clause_expressions = output_directory + "/raw_clause_expressions.txt"
    print(" [RTL_gen][d]    Raw Clause Expressions have been written")
    with open(clause_expressions, "w") as clause_expressions_fp:
        for j in range(clauses):
            l = " & ".join(["x[%d]" % (k/2) if k %2 == 0 else "~x[%d]" % (int(k/2))
                        for k in range(features*2) if TAs[j][k] == 1])
            print("clause %d: %s" % (j, l), file=clause_expressions_fp)

	# we must also deal with clauses that have no includes - these are all exclude clauses 
	# in the code below we are taking the indexes for these clauses - we will set these as zero.
    all_exclude_indexes = []
    all_exc_count = 0	

    for j in range(TAs.shape[0]):
        if(any(v == 1 for v in TAs[j])):
            pass
        else:
            all_exc_count += 1
            all_exclude_indexes.append(j)
    print(" [RTL_gen][d]    No. All Exclude Clauses: ", all_exc_count)

    TAs = TAs.reshape(1, clauses, features*2)

    with open(file_name, "w") as f:
        for i in range(number_of_blocks):
            TA_slice = []
            for c in range(1):
                TA_slice.append(TAs[c][:, starting_block:finish_block])
            if finish_block >= features*2:    
                lit_range = (features*2 - starting_block)
            else:
                lit_range = bus_width*2
            starting_block += bus_width*2
            if finish_block + bus_width*2 > features*2:    
                finish_block = starting_block + (features*2 - starting_block)
            else:
                finish_block += bus_width*2
            if i == 0:
                print("module HCB_%d (x, partial_clause, clk, valid);" % (i), file=f)
                print("\toutput\tlogic[%d:0] partial_clause [%d];" % (clauses-1, 0), file=f)
            else:
                print("module HCB_%d (x, partial_clause, partial_clause_prev, clk, valid);" % (i), file=f)
                print("\tinput\tlogic [%d:0] partial_clause_prev [%d];" % (clauses-1, 0), file=f)
                print("\toutput\tlogic[%d:0] partial_clause [%d];" % (clauses-1, 0), file=f)
            
            print("\tinput\tlogic clk;", file=f)
            print("\tinput\tlogic [%d:0] x;" % (bus_width-1), file=f)    
            print("\tinput\tlogic valid;", file=f)

            print("\talways @(posedge clk) begin", file=f)
            print("\t\tif(valid) begin", file=f)
            for c in range(1):
                TAs_ = TA_slice[c]
                print("\t\t\t// Class %d" % (c), file=f)
                for j in range(clauses):
                    l = " & ".join(["x[%d]" % (k/2) if k %2 == 0 else "~x[%d]" % (int(k/2))
                                    for k in range(lit_range) if TAs_[j][k] == 1])
                    if l == '': 
                        if i != (number_of_blocks):
                            if i == 0: 
                                if j in all_exclude_indexes:
                                    print("\t\t\tpartial_clause[%d][%d] \t= 1'b0;" % (c, j), file=f)
                                else:
                                    print("\t\t\tpartial_clause[%d][%d] \t= 1'b1;" % (c, j), file=f)
                            else:
                                print("\t\t\tpartial_clause[%d][%d] \t= partial_clause_prev[%d][%d] & 1'b1;" % (c, j, c, j), file=f)
                    else:
                        if i == 0:
                            print("\t\t\tpartial_clause[%d][%d] \t= %s;" % (c, j, l), file=f)
                        else:
                            print("\t\t\tpartial_clause[%d][%d] \t= partial_clause_prev[%d][%d] & %s;" % (c, j, c, j, l), file=f)
            print("\t\tend", file=f)
            print("\tend", file=f)
            print("endmodule\n\n", file=f)
        

def parse_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def train_model(config):
    # NOTE - I haven't checked the training code thorougly here - this is just a placeholder
    # Using the standard TMU impl (see utis/tmu/models/classification)
    print("Training model with config:", config)
    
    # Check the configuration
    config_err = 0
    checkconfig(config, config_err)
    if config_err:
        print("There were errors in the training config json")
        return

    # Prepare the data
    data = prep_data(config)
    
    # Determine the type of Tsetlin Machine to use
    if config["TM"] == "Vanilla":
        tm = TMClassifier(
            type_iii_feedback=False,
            number_of_clauses=int(config["Clauses"]),
            T=int(config["T_value"]),
            s=float(config["s_value"]),
            max_included_literals=int(config["max_included_literals"]),
            weighted_clauses=False,
            seed=42,
        )
    elif config["TM"] == "Coalesced":
        tm = TMCoalescedClassifier(
            type_iii_feedback=False,
            number_of_clauses=int(config["Clauses"]),
            T=int(config["T_value"]),
            s=float(config["s_value"]),
            max_included_literals=int(config["max_included_literals"]),
            weighted_clauses=True,
            seed=42,
        )
    else:
        print("Unknown Tsetlin Machine type:", config["TM"])
        return

    # Train the model
    for epoch in range(int(config["epochs"])):
        benchmark_total = BenchmarkTimer(logger=None, text="Epoch Time")
        with benchmark_total:
            tm.fit(data["X_train"], data["Y_train"])
            result = 100 * (tm.predict(data["X_test"]) == data["Y_test"]).mean()
            print(f"Epoch {epoch+1} Accuracy: {result:.2f}%")

def generate_rtl(config):
    output_directory = config.get("Output_Directory")
    tm_type = config.get("TM")
    tas = config.get("TAs")
    weights = config.get("Weights")
    classes = int(config.get("Classes"))
    clauses = int(config.get("Clauses"))
    bus_width = int(config.get("BusWidth"))
    features = int(config.get("Features"))
    test_data = config.get("Test_Data")

    number_of_blocks   	= ceil(features/bus_width)  

    print(f"Generating RTL with config: Output Directory: {output_directory}, TM: {tm_type}, TAs: {tas}, Weights: {weights}, Classes: {classes}, Clauses: {clauses}, BusWidth: {bus_width}, Features: {features}, Test Data: {test_data}")

    path = output_directory+"/RTL"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
    else:
        print("Directory exists")
    
    # Only doing the Coalesced TM for now
    if tm_type == "Coalesced":
        print("	----------------------------------------")
        print("	    	Starting RTL Generation")
        print("	----------------------------------------")
        print("                        ")
        print("	[W] = warning  [e] = error  [d] = debug ")
        print("                        ")
        # Read in the TA_file and generate the HCB (Hard Coded Clause Blocks - per class)
        TAs = np.loadtxt(tas, dtype=int)
        print("")
        print(" [RTL_gen][d]    Number of TAs: ", TAs.shape[0])
        print(" [RTL_gen][d]    Raw Values: ", TAs)

        for i in range(TAs.shape[0]):
            if TAs[i] <= 128:
                TAs[i] = 0
            else:
                TAs[i] = 1

        print(" [RTL_gen][d]    Number of Includes: ", np.count_nonzero(TAs))
        TAs = TAs.reshape(clauses, features*2)
        print(" [RTL_gen][d]    TA new shape: ", TAs.shape)
        print(" [RTL_gen][W]    Incorrect TA profiles produce incorrect hardware ;)")
        print("")

        Weights = np.loadtxt(weights, dtype=int)
        print(" [RTL_gen][d]    Weights of TAs: ", Weights.shape[0])
        # write the hard coded clause block code - this one is for vanilla TM
        hard_coded_blocks 	= output_directory + "/RTL/TM_Hard_Coded_Clause_Blocks.sv"
        coalesced_tm_write_hard_coded_blocks(number_of_blocks, TAs, Weights, hard_coded_blocks, bus_width, output_directory, clauses, features)

        # write the hard coded clause block top - this one is for vanilla TM
        hard_coded_top      = output_directory + "/RTL/HCB_top.sv"
        coalesced_tm_hard_coded_blocks_top(hard_coded_top, number_of_blocks)

        TM_top      = output_directory + "/RTL/TM_top.sv"
        coalesced_tm_write_top(TM_top, number_of_blocks, bus_width, clauses)

        axis_wrapper_f		= output_directory + "/RTL/axis_wrapper.sv"
        coalesced_tm_write_axis_wrapper(axis_wrapper_f, bus_width, number_of_blocks, 1, clauses, classes, features)

        # write the weights into a hard coded weights file
        bits_required = 0
        bits_required, Weights = get_bits_required(weights, clauses, classes)

        weights_file		= output_directory + "/RTL/hard_coded_weight.sv"
        write_weights(weights_file, Weights, bits_required, classes, clauses)

        print("	----------------------------------------")
        print("")

        print("	----------------------------------------")
        print("		    Adjusting templates")
        print("	----------------------------------------")

        os.system("cp utils/RTL_templates/TM_argmax.sv "+ output_directory+"/RTL")
        print("	[RTL_gen]	TM_argmax.sv \t\tis added")

        os.system("cp utils/RTL_templates/new_adder.sv "+ output_directory+"/RTL")
        print("	[RTL_gen]	new_adder.sv \t\tis added")

        os.system("cp utils/RTL_templates/AXI_Interface.sv "+ output_directory+"/RTL")
        print("	[RTL_gen]	AXI_Interface.sv \tis added")




def synthesize_and_implement(config):
    print("Synthesizing and implementing with config:", config)

def deploy(config):
    print("Deploying with config:", config)

if __name__ == "__main__":
    file_path = '/home/tousif/Desktop/MATADOR_09_03_2025/matador/MATADOR_NO_GUI.json'
    data = parse_json(file_path)
    
    flow_control = data.get("Flow Control", {})
    
    if flow_control.get("Train_Model") == "Y":
        train_model(data.get("Model_Training_Config", {}))
    
    if flow_control.get("Generate_RTL") == "Y":
        generate_rtl(data.get("Generate_RTL_Config", {}))
    
    if flow_control.get("Synth + Impl") == "Y":
        synthesize_and_implement(data.get("Generate_RTL_Config", {}))
    
    if flow_control.get("Deploy") == "Y":
        deploy(data.get("Generate_RTL_Config", {}))