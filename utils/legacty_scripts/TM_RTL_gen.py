import numpy as np 
from math import ceil 
from math import log
import os
import json

def hard_coded_blocks_top(number_of_blocks, clauses,filename):
	with open(filename, "w") as f:
		print("`timescale 1ns / 1ps", file=f)
		print("", file=f)
		print("", file=f)
		print("module HCB_top #(", file=f)
		print("\tparameter PACKETS_NUM,", file=f)
		print("\tparameter CLAUSE_NUM,", file=f)
		print("\tparameter C_S00_AXIS_TDATA_WIDTH", file=f)
		print("\t)", file=f)
		print("\t(", file=f)
		print("\tinput clk,", file=f)
		print("\tinput [C_S00_AXIS_TDATA_WIDTH - 1:0] x,", file=f)
		print("\tinput [PACKETS_NUM - 1:0] valid,", file=f)
		print("\toutput [CLAUSE_NUM - 1:0] partial_clause", file=f)
		print("\t);", file=f)

		for i in range(number_of_blocks):
			print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg_%d;"%(i), file=f)

		print("\tinitial begin", file=f)
		for i in range(number_of_blocks):
			print("\t\tpartial_clause_reg_%d = {%d'b0};"%(i,clauses), file=f)
		print("\tend", file=f)
		print("\tassign partial_clause = partial_clause_reg_%d;"%(i) ,file=f)

		for i in range(number_of_blocks):
			print("\tHCB_%d HCB_inst_%d(" %(i, i), file=f)
			print("\t\t.clk(clk),", file=f)
			print("\t\t.x(x),", file=f)
			print("\t\t.valid(valid[%d]),"%(i), file=f)
			if(i == 0):
				print("\t\t.partial_clause(partial_clause_reg_%d)" %(i), file=f)
			else:
				print("\t\t.partial_clause_prev(partial_clause_reg_%d),"%(i-1), file=f)
				print("\t\t.partial_clause(partial_clause_reg_%d)"%(i), file=f)
			print("\t);", file=f)
			print("", file=f)
		print("endmodule", file=f)


def write_hard_coded_blocks(number_of_blocks, TAs, filename):
	starting_block = 0 
	finish_block = AXI_data_width*2
	all_exclude_indexes = []
	# Find the indexes with all excludes and zero them at the start
	# TA shape = clauses x literals	
	print(TAs.shape)
	for i in range(TAs.shape[0]):
		# print(TAs[i])
		if(any(v == 1 for v in TAs[i])):
			all_exclude_indexes.append(i)
	print(all_exclude_indexes)

	with open(filename, "w") as f:
		for i in range(number_of_blocks):
			TA_slice = TAs[:, starting_block:finish_block]
			if finish_block >= number_of_features*2:	
				lit_range = (number_of_features*2 - starting_block)
			else:
				lit_range = AXI_data_width*2
			starting_block += AXI_data_width*2
			if finish_block + AXI_data_width*2 > number_of_features*2:	
				finish_block = starting_block + (number_of_features*2 - starting_block)
			else:
				finish_block += AXI_data_width*2
			print(lit_range)
			print(TA_slice.shape)
			if(i == 0):
				print("module HCB_%d (x, partial_clause, clk, valid);" %(i), file=f)
				print("\toutput\tlogic[%d:0] partial_clause;" % (clauses-1), file=f)
			else:
				print("module HCB_%d (x, partial_clause, partial_clause_prev, clk, valid);" %(i), file=f)
				print("\tinput\tlogic [%d:0] partial_clause_prev;" % (clauses-1), file=f)
				print("\toutput\tlogic[%d:0] partial_clause;" % (clauses-1), file=f)
			
			print("\tinput\tlogic clk;", file=f)
			print("\tinput\tlogic [%d:0] x;" % (AXI_data_width-1), file =f)	
			print("\tinput\tlogic valid;", file=f)

			print("\talways @(posedge clk) begin", file=f)
			print("\t\tif(valid) begin", file=f)
			for j in range(clauses):
				l = " & ".join(["x[%d]" % (k/2) if k %2 == 0 else "~x[%d]" % (int(k/2))
					for k in range(lit_range) if TA_slice[j][k] == 1])
				if l == '': 
					if i != (number_of_blocks):
						if(i == 0): 
							print("\t\t\tpartial_clause[%d] \t= 1'b1;" % (j), file=f)
						else:
							print("\t\t\tpartial_clause[%d] \t= partial_clause_prev[%d] & 1'b1;" % (j, j), file=f)
				else:
					if(i==0):
						print("\t\t\tpartial_clause[%d] \t= %s;" % (j, l), file=f)
					else:
						print("\t\t\tpartial_clause[%d] \t= partial_clause_prev[%d] & %s;" % (j, j, l), file=f)
			print("\t\tend",file=f)
			print("\tend", file=f)
			print("endmodule\n\n", file=f)

def write_top(number_of_blocks, filename, bits_required, classes, bus_width, clauses):
	with open(filename, "w") as f:
		print("`timescale 1ns / 1ps", file=f)
		print("", file=f)
		print("module Hard_Coded_Inference_Top #(", file=f)
		print("\tparameter STAGE_NUM,", file=f)
		print("\tparameter CLAUSE_NUM,", file=f)
		print("\tparameter CLASS_NUM,", file=f)
		print("\tparameter WEIGHT_LENGTH,", file=f)
		print("\tparameter C_S00_AXIS_TDATA_WIDTH,", file=f)
		print("\tparameter C_M00_AXIS_TDATA_WIDTH,", file=f)
		print("\tparameter PACKETS_NUM", file=f)
		print(")", file=f)

		print("(x, y, packet_counter, valid, s_axis_ready, clk, finish, last, last_out,", file=f)
		print("clauses, class_sums, m00_axis_tready);", file=f)
		print("\tinput logic clk;", file=f)
		print("\toutput logic finish;", file=f)
		print("\toutput logic [CLAUSE_NUM - 1:0] clauses;", file=f)
		print("\toutput logic signed [WEIGHT_LENGTH - 1:0] class_sums [%d];" %(classes), file=f)
		print("\tinput logic s_axis_ready;", file=f)
		print("\tinput logic m00_axis_tready;", file=f)
		print("\tinput logic [C_M00_AXIS_TDATA_WIDTH:0] x;", file=f)
		print("\tinput logic [PACKETS_NUM - 1:0] valid;", file=f)
		print("\tinput logic last;", file=f)
		print("\toutput logic last_out;", file=f)
		print("\tinput logic [C_M00_AXIS_TDATA_WIDTH:0] packet_counter;", file=f)
		print("\toutput logic [C_M00_AXIS_TDATA_WIDTH:0] y;", file=f)


		print("", file=f)
		for i in range(number_of_blocks):
			print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg_%d;" %(i), file=f)
			print("\tlogic valid_reg_%d;" %(i), file=f)
		print("\tlogic valid_reg_%d;" %(number_of_blocks), file=f)
		print("\tlogic adder;", file=f)
		print("\tlogic adder_2;", file=f)
		print("\tlogic argmax;", file=f)
		print("\tlogic finished;", file=f)
		print("\tlogic delay_1;", file=f)
		print("\tlogic last_registered;", file=f)
		print("\tassign finish = finished;", file=f)

		print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg;", file=f)
		print("", file=f)

		print("\tassign clauses = partial_clause_reg;", file=f)
		print("\tinitial begin", file=f)
		for i in range(number_of_blocks):
			print("\t\tvalid_reg_%d = 1'b0;" %(i), file=f)

		print("\t\tvalid_reg_%d = 1'b0;" %(number_of_blocks), file=f)
		print("\t\tlast_registered = 1'b0;", file=f)
		print("\t\tlast_out = 1'b0;", file=f)

		print("\t\t(*DONT_TOUCH = \"TRUE\"*) argmax = 1'b0;", file=f)
		print("\t\t(*DONT_TOUCH = \"TRUE\"*) adder  = 1'b0;", file=f)
		print("\t\tadder_2 = 1'b0;", file=f)
		print("\t\t(*DONT_TOUCH = \"TRUE\"*) delay_1 = 1'b0;", file=f)
		print("\t\t(*DONT_TOUCH = \"TRUE\"*) delay_1 = 1'b0;", file=f)

		print("\t\ty = {%d'b0};" %(bus_width), file=f)
		print("\tend", file=f)
		print("", file=f)
		print("\tHCB_top #(", file=f)
		print("\t\t.PACKETS_NUM(PACKETS_NUM),", file=f)
		print("\t\t.CLAUSE_NUM(CLAUSE_NUM),", file=f)
		print("\t\t.C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH)", file=f)
		print("\t)", file=f)
		print("\tHT(.clk(clk),", file=f)
		print("\t\t.x(x),", file=f)
		print("\t\t.valid(valid), ", file=f)
		print("\t\t.partial_clause(partial_clause_reg));", file=f)


		print("", file=f)
		print("\twire adder_done;", file=f)

		print("""
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
	
endmodule""", file=f)

def write_axis_wrapper(number_of_blocks, axis_wrapper_f, AXI_data_width, clauses, classes, bits_required, features, adder_stages):
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
		print("\t\tparameter WEIGHT_LENGTH = %d," %(bits_required), file=f)
		print("\t\tparameter FEATURE_NUM = %d," %(features), file=f)
		print("\t\tparameter PACKETS_NUM = (FEATURE_NUM - 1)/C_S00_AXIS_TDATA_WIDTH + 1", file=f)
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
        output wire [CLAUSE_NUM - 1:0] clauses,
	    output logic signed [WEIGHT_LENGTH-1:0] class_sums [CLASS_NUM],
		// Ports of Axi Master Bus Interface M00_AXIS
		input  wire  m00_axis_aclk,
		input  wire  m00_axis_aresetn,
		input  wire  m00_axis_tready,
		output wire  m00_axis_tvalid,
		output reg    [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tkeep,
		output wire  [C_M00_AXIS_TDATA_WIDTH-1 : 0] m00_axis_tdata,
		output wire  [(C_M00_AXIS_TDATA_WIDTH/8)-1 : 0] m00_axis_tstrb,
		output wire  m00_axis_tlast
    );
    
   	wire axis2pipe_tvalid, axis2pipe_tready, axis2pipe_tlast;
	wire pipe2axis_tvalid, pipe2axis_tready, pipe2axis_tlast;
	wire [C_M00_AXIS_TDATA_WIDTH-1:0] pipe2axis_data;
	
	logic [63:0] packet_counter; 
	logic full;
	logic inference_complete;
	logic last_registered; 
	logic last_complete;

    reg old_inference_complete,old_old_inference_complete;
    reg old_s00_axis_tlast,old_last_complete,old_old_last_complete;	

	assign m00_axis_tvalid = old_inference_complete;
	assign m00_axis_tlast = old_last_complete;	
	
	reg old_s00_axis_tready;
	reg plus_stat;
	initial begin
	   m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b0}};
	   packet_counter = 0;
	end

	always @(posedge m00_axis_aclk) begin
	   old_inference_complete <= inference_complete;
	   old_old_inference_complete <= old_inference_complete;
	   old_s00_axis_tready <= s00_axis_tready;
	   old_s00_axis_tlast <= s00_axis_tlast;
	   old_last_complete <= last_complete;
	   old_old_last_complete <= old_last_complete;
	   if(!s00_axis_aresetn)begin 
	       packet_counter = 0;
	       plus_stat = 0;
	   end
        if (inference_complete) begin
	       packet_counter = 0;
	   end
	   if(packet_counter == PACKETS - 1) begin 
	       packet_counter = 0; 
	   end
	   else if (s00_axis_tlast && !old_s00_axis_tlast) begin
	       plus_stat = 0;
	   end
	   else if(axis2pipe_tready && s00_axis_tready && old_s00_axis_tready) begin 
	       if(plus_stat) begin 
	           packet_counter = packet_counter + 1;
	       end
	       else begin
	       end
	   end
	   
	   if (inference_complete && !last_complete) begin
	       m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b1}};
	   end
	   else if (!old_old_last_complete && old_last_complete) begin
	       m00_axis_tkeep = {(C_M00_AXIS_TDATA_WIDTH/8){1'b0}};
	   end
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
	   .valid(valid_reg),
	   .s_axis_tready(s00_axis_tready),
	   .m00_axis_tready(axis2pipe_tready),
	   .packet_counter(packet_counter),
	   .y(m00_axis_tdata),
	   //test ports 
	   .clauses(clauses),
	   .class_sums(class_sums),
	   .finish(inference_complete),
	   .last(s00_axis_tlast),
	   .last_out(last_complete)
	);
	
	// Instantiation of Axi Bus Interface S00_AXIS
	axis_adder_v1_0_S00_AXIS #(
	   .PACKETS_NUM(PACKETS_NUM)
	)
	 axis_adder_v1_0_S00_AXIS_inst (
		.clk(s00_axis_aclk),
		.rst(~s00_axis_aresetn),
		.s_axis_tdata(s00_axis_tdata),
		.s_axis_tvalid(s00_axis_tvalid),
		.s_axis_tready(s00_axis_tready),
		.s_axis_tlast(s00_axis_tlast),
		.valid(valid_reg),
		//.ptr_reg(ptr_reg),
		.full(full),
		.m_axis_tdata(axis2pipe_data),
		.m_axis_tvalid(axis2pipe_tvalid),
		.m_axis_tready(axis2pipe_tready)
	); 
	
endmodule""", file=f)



def to_bin(val, bits):
    s = bin(val & int("1"*bits, 2))[2:]
    return ("{0:0>%s}" % (bits)).format(s)

def write_adder(number_of_blocks, filename, bits_required, weights, classes, clauses): 

	with open(filename, "w") as f: 
		# generate the individual class sums 
		# print("module weighted_adder(clauses, weights, c_sum);", file=f)
		# print("\tinput wire [%d:0] clauses;" %(clauses-1), file=f)
		# print("\tinput wire [%d:0] weights [%d:0];" %(bits_required,clauses-1), file=f)
		# print("\toutput logic [%d:0] c_sum;" %(bits_required-1), file=f)
		# print("", file=f)
		# print("\talways_comb begin", file=f)
		# print("\t\tc_sum = 0;", file=f)
		# print("\t\tfor(int i = 0; i < %d; i= i+1) begin " %(clauses), file=f)
		# print("\t\t\tc_sum += clauses[i]*weights[i];", file=f)
		# print("\t\tend", file=f)
		# print("\tend", file=f)
		# print("endmodule", file=f)
		# print("", file=f)

		# top level module for feeding in the weights and class_sums per class
		print("module hard_coded_weight #(", file=f)
		print("\tparameter CLAUSE_NUM", file=f)
		print("\t)", file=f)
		print("\t(", file=f)
		print("\toutput logic signed [%d:0] weights [%d][CLAUSE_NUM]); " %(bits_required-1, classes), file=f)
		print("", file=f)
		# 	lk, clauses, class_sums, valid);", file=f)
		# print("\tinput logic clk;", file=f)
		# print("\tinput logic valid;", file=f)
		# print("\tinput logic [%d:0] clauses;" %(clauses-1), file=f)
		# print("\toutput logic [%d:0] class_sums [%d];" %(bits_required-1, classes-1), file=f)
		# print("\tlogic [%d:0] class_sums_local [%d:0];" %(bits_required-1, classes-1), file=f)
		# print("\tlogic [%d:0][%d:0] weights [%d:0];" %(bits_required-1, clauses-1, classes-1), file=f)
		# print("", file=f)
		# iterorate and assign the weights 
		for i in range(classes):
			for j in range(clauses):
				print("\tassign weights[%d][%d] \t=\t%d'b%s;" %(i, j, bits_required,to_bin(weights[i][j], bits_required)), file=f)
		# generate adders for each class
		# print("", file=f) 
		# print("\tgenerate", file=f)
		# print("\t\tgenvar i;", file=f)
		# print("\t\tfor(i = 0; i < %d; i= i+1) begin " %(classes), file=f)
		# print("\t\t\tweighted_adder adder_w(", file=f)
		# print("\t\t\t\t.clauses(clauses),", file=f)
		# print("\t\t\t\t.weights(weights[i]),", file=f)
		# print("\t\t\t\t.c_sum(class_sums_local[i])", file=f)
		# print("\t\t\t);", file=f)
		# print("\t\tend", file=f)
		# print("\tendgenerate", file=f)
		# print("", file=f)
		# print("\talways @(posedge clk) begin ", file=f)
		# print("\t\tclass_sums = class_sums_local;", file=f)
		# print("\tend", file=f)
		print("endmodule", file=f)

def get_bits_required(Weights_file):
	Weights = []
	W_file = open(Weights_file, "r")
	data = W_file.read()
	data_into_list = data.split("\n")
	Weights = [int(i) for i in data_into_list]
	Weights = np.array(Weights)
	Weights = np.reshape(Weights, (classes,clauses))
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

	bits 	= 0 

	if max_pos > max_neg: 
		abs_w = max_pos
		bits = ceil(log(abs_w, 2))
	else: 
		abs_w = max_neg 
		bits = ceil(log(abs_w, 2)) + 1

	print("bits required: ",  bits)

	return bits, Weights 


# Globals
# number_of_features = 784
# AXI_data_width 	   = 64

# classes 	 	   = 10 
# clauses 		   = 500	

config_file = open("gen_RTL.json")
config = json.load(config_file)

output_dir = config["Output_Directory"]
TA_file = config["TAs"]
Weights_file = config["Weights"]
number_of_features = int(config["Features"])
AXI_data_width = int(config["BusWidth"])
classes = int(config["Classes"])
clauses = int(config["Clauses"])
number_of_blocks   = ceil(number_of_features/AXI_data_width)  
adder_stages = int(config["Adder_Stages"])
features = int(config["Features"])

# make directory for the RTL 
path = output_dir+"/RTL"

isExist = os.path.exists(path)
if not isExist:
	print("Creating directory for RTL")
	os.makedirs(path)	

# move the RTL templates to the RTL directory 
print("Number of Hard Coded Blocks required: ", number_of_blocks)

TA_file = output_dir+"/Hard_Coded_TAs.txt"
TAs = np.loadtxt(TA_file, dtype=int)
TAs = TAs.transpose()
print(TAs.shape)

hard_coded_blocks 	= output_dir + "/RTL/TM_Hard_Coded_Clause_Blocks.sv"
hard_coded_top      = output_dir + "/RTL/HCB_top.sv"
top 				= output_dir + "/RTL/TM_top.sv"	
weight 				= output_dir + "/RTL/hard_coded_weight.sv" 
axis_wrapper_f		= output_dir + "/RTL/axis_wrapper.sv"

bits_required = 0
bits_required, Weights = get_bits_required(Weights_file)


hard_coded_blocks_top(number_of_blocks, clauses, hard_coded_top)
write_hard_coded_blocks(number_of_blocks, TAs, hard_coded_blocks)
write_top(number_of_blocks, top, bits_required, classes, AXI_data_width, clauses)
write_adder(number_of_blocks, weight, bits_required, Weights, classes, clauses)
write_axis_wrapper(number_of_blocks, axis_wrapper_f, AXI_data_width, clauses, classes, bits_required, features, adder_stages)

# cp the files from RTL_templates
os.system("cp /home/tousif/MATADOR/RTL_templates/AXI_Interface.sv "+ output_dir+"/RTL")
os.system("cp /home/tousif/MATADOR/RTL_templates/new_adder.sv "+ output_dir+"/RTL")
os.system("cp /home/tousif/MATADOR/RTL_templates/TM_argmax.sv "+ output_dir+"/RTL")
# write_argmax(bits_required, classes)


