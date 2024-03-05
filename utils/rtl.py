# Tousif Rahman, Gang Mao 2023
# 
# This file is used for generating RTL hardware from trained
# TM models. Currently it supports Vanilla and Coalesced Models.
# 
# Last Update 08/09/2023
# 
# Matador 2023

  # Permission to use, copy, modify, and/or distribute this software for any  
  # purpose with or without fee is hereby granted, provided that the above    
  # copyright notice and this permission notice appear in all copies.         
                                                                            
  # THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES  
  # WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF          
  # MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR   
  # ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES    
  # WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN     
  # ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF   
  # OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


import numpy as np #
import math
from math import ceil 
from math import log
import os
import json
import argparse

# Read in the config file and examine the inputs 
parser = argparse.ArgumentParser(description="Matador Training")

# to train the output directory for the results must be specified. 
parser.add_argument("-output_dir", help="results directory to store the training data", type=str)
args = parser.parse_args()

output_dir = args.output_dir
config_file = output_dir + "/gen_RTL.json"
config_file = open(config_file)
config = json.load(config_file)

TM_type				= config["TM"]

output_dir 			= config["Output_Directory"]
TA_file 			= config["TAs"]
Weights_file 		= config["Weights"]
number_of_features 	= int(config["Features"])
AXI_data_width 		= int(config["BusWidth"])
classes 			= int(config["Classes"])
clauses 			= int(config["Clauses"])
number_of_blocks   	= ceil(number_of_features/AXI_data_width)  
adder_stages 		= int(config["Adder_Stages"])
features 			= int(config["Features"])
test_data 			= config["Test_Data"]

project_config_file = open(output_dir+"/config.json")
project_config = json.load(project_config_file)

Matador_home_path = project_config["Matador_home_path"]


def pack_data():

	mnist_file = test_data
	File_data = np.loadtxt(mnist_file, dtype=int)
	number_of_examples = 10
	# Remove the class cols 
	cols = File_data[:, -1]

	f = open(output_dir + "/RTL/expected_answers.txt", "w")
	for i in range(number_of_examples):
		f.write(str(cols[i])+"\n")
	f.close()

	File_data = File_data[:, :-1]
	AXI_bus_size =  AXI_data_width

	print("	[RTL_gen][d] Number of features:\t\t", File_data[0].shape[0])
	print(File_data[0])

	print("	[RTL_gen][d] Packets:\t\t\t", File_data[0].shape[0]/AXI_bus_size)
	print("	[RTL_gen][d] Number of Extra packet(s):\t", math.ceil(File_data[0].shape[0]/AXI_bus_size) - math.floor(File_data[0].shape[0]/AXI_bus_size))
	print("	[RTL_gen][d] Number of packets required:\t", math.ceil(File_data[0].shape[0]/AXI_bus_size))

	mnist_packets = []
	packet_32 = []
	packet_counter = 0 

	for j in range(number_of_examples):
		for i in range(File_data[j].shape[0]):
			if packet_counter <= AXI_bus_size-1: 
				packet_32.append(File_data[j][i])
			else: 
				mnist_packets.append(packet_32)
				packet_32 = []
				# print(i)
				packet_counter = 0
				packet_32.append(File_data[j][i])

			packet_counter += 1

		# If datapoint is complete and packet is not fully filled...
		# Fill the remainder with zeros 
		if(len(packet_32) != 0):
			# print("Half filled packet: ", len(packet_32))
			remainder_zero_fill = AXI_bus_size -len(packet_32)   
			for l in range(remainder_zero_fill):
				packet_32.append(0)
			
			mnist_packets.append(packet_32)
			packet_counter = 0
			packet_32 = []
	# convert the list of lists to a numpy array 
	mnist_packets_np_1 = np.array(mnist_packets)
	mnist_packets_np  = np.fliplr(mnist_packets_np_1)
	# print(mnist_packets_np)
	# now convert each 32 number array to its equivalent binary 
	# this will be the 32 bit packet sent each time 
	mnist_packets_np_bits =np.packbits(mnist_packets_np_1, axis=-1,  bitorder='little')
	mnist_packets_np_bits.dtype = np.uint64
	# print(mnist_packets_np_bits)
	# now the mnist data is in 32 bit packets -- we can write the testbench 
	
	tb_file_test = output_dir + "/RTL/test_data_copy_paste_examples.mem"
	f = open(tb_file_test, "w")

	for i in range(mnist_packets_np.shape[0]):
		# if i %13 == 0:
			# print("------")
		# print(mnist_packets_np_bits[i])
		print(''.join(map(str, mnist_packets_np[i])), file=f)

def write_testbench(tb_file):
	with open(tb_file, "w") as f:
		print("module axis_stream_tb();", file=f)
		print("\t\tparameter C_S00_AXIS_DATA_WIDTH = %d;"% AXI_data_width, file=f)
		print("\t\tparameter C_M00_AXIS_DATA_WIDTH = %d;"% AXI_data_width, file=f)
		print("\t\tparameter NUM_PACKETS = %d;"% int(number_of_blocks), file=f)
		print("\t\tparameter DATAPOINTS = %d;" % 10, file=f)
		print("""

    reg clock;
    reg areset;
    
    reg s00_axis_tready;
    reg m00_axis_tready;
    reg s00_last;
    reg m00_last; 
    reg m00_valid; 
    reg s00_valid; 
    reg [C_S00_AXIS_DATA_WIDTH-1:0] payload; 
    reg [C_M00_AXIS_DATA_WIDTH-1:0] recieved;
    reg [63:0] input_data [0:DATAPOINTS * NUM_PACKETS - 1]; 
    
    wire [7:0] bytes; 
    int input_counter = 0; 
    
    reg last_triggered; 
    
    axis_wrapper_top  
        DUT(
            .s00_axis_aclk(clock),
            .s00_axis_aresetn(areset),
            .s00_axis_tready(s00_axis_tready),                  
            .s00_axis_tdata(payload),
            .s00_axis_tstrb(bytes),
            .s00_axis_tlast(s00_last),
            .s00_axis_tvalid(s00_valid),
            .m00_axis_aclk(clock),
            .m00_axis_aresetn(areset),
            .m00_axis_tready(m00_axis_tready),
            .m00_axis_tdata(recieved),                                                     
            .m00_axis_tlast(m00_last),                          
            .m00_axis_tvalid(m00_valid)                        
        );
 
    initial begin    
        last_triggered = 1'b0;                                          
        #0 clock = 0;
        forever begin
            #20 clock = ~clock;
        end 
    end
    initial begin     
        $readmemb("test_data_copy_paste_100_new_examples.mem", input_data);
        areset <= 1'b1;
        #20 s00_last        <= 1'b0;
        #20 areset          <= 1'b0;
        #20 areset          <= 1'b1;
        #20 s00_valid       <= 1'b0;
        m00_axis_tready     <= 1'b1;
        #20 payload         <= input_data[0];
        s00_valid           <= 1'b1;
    end
    integer cnt = 0;
    always @(posedge clock) begin  
        if(s00_axis_tready && s00_valid) begin
            if(input_counter == DATAPOINTS*NUM_PACKETS-2) begin 
               s00_last     <= 1'b1;  
            end
			// Note you may have to change this line here 
            else if (input_counter == 160) begin
                s00_valid = 0;
            end
            
            if(s00_last == 1'b1) begin 
                s00_last    <= 1'b0;
                s00_valid   <= 1'b0;
            end
            input_counter   = input_counter + 1;
            payload         = input_data[input_counter];
        end 
        else if (!s00_valid) begin
            if (cnt < NUM_PACKETS-1) begin
                cnt = cnt + 1;
            end
            else if (cnt == NUM_PACKETS-1) begin
                s00_valid = 1;
                cnt = 0;
            end
        end
        
        if(m00_last) begin 
            last_triggered = 1'b1;
        end 
        else if(last_triggered) begin 
            input_counter = 0; 
            s00_last    <= 1'b0;
            s00_valid   <= 1'b1;
            last_triggered = 1'b0;
            payload         <= input_data[0];
        end
    end

endmodule
	""", file=f)

# View from the bottom of this file to see how everything fits together ;) 
def get_bits_required(Weights_file):
	Weights = []
	W_file = open(Weights_file, "r")
	data = W_file.read()
	data_into_list = data.split("\n")
	Weights = [int(i) for i in data_into_list]
	Weights = np.array(Weights)
	Weights = np.reshape(Weights, (classes,clauses))
	# print(Weights.shape)

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

	max_pos = abs(max_positive)
	max_neg = abs(max_negative) 

	bits 	= 0 

	if max_pos > max_neg: 
		abs_w = max_pos
		bits = ceil(log(abs_w, 2))
	else: 
		abs_w = max_neg 
		bits = ceil(log(abs_w, 2)) + 1
	# print("bits required: ",  bits)
	return bits, Weights 

def vanilla_tm_write_axis_wrapper(axis_wrapper_f):
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
		bits_required = int(ceil(log(clauses, 2)) - 1)
		print("\t\tparameter WEIGHT_LENGTH = %d," %(bits_required), file=f)
		print("\t\tparameter FEATURE_NUM = %d," %(features), file=f)
		print("\t\tparameter PACKETS_NUM = (FEATURE_NUM - 1)/C_S00_AXIS_TDATA_WIDTH + 1,", file=f)
		print("\t\tparameter VANILLA = 1,", file=f)
		print("\t\tparameter COALESCED = 0", file=f)
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

def write_top(filename):
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

		print("(x, y, packet_counter, valid, s_axis_tready, clk, finish, last, last_out,", file=f)
		print("clauses, class_sums, m00_axis_tready);", file=f)
		print("\tinput logic clk;", file=f)
		print("\toutput logic finish;", file=f)
		print("\toutput logic [CLAUSE_NUM - 1:0] clauses [CLASS_NUM - 1:0];", file=f)
		print("\toutput logic signed [WEIGHT_LENGTH - 1:0] class_sums [%d];" %(classes), file=f)
		print("\tinput logic s_axis_tready;", file=f)
		print("\tinput logic m00_axis_tready;", file=f)
		print("\tinput logic [C_M00_AXIS_TDATA_WIDTH-1:0] x;", file=f)
		print("\tinput logic [PACKETS_NUM - 1:0] valid;", file=f)
		print("\tinput logic last;", file=f)
		print("\toutput logic last_out;", file=f)
		print("\tinput logic [C_M00_AXIS_TDATA_WIDTH-1:0] packet_counter;", file=f)
		print("\toutput logic [C_M00_AXIS_TDATA_WIDTH-1:0] y;", file=f)


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

		print("\tlogic [CLAUSE_NUM - 1:0] partial_clause_reg [CLASS_NUM - 1:0];", file=f)
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

		print("\t\ty = {%d'b0};" %(AXI_data_width), file=f)
		for i in range(number_of_blocks):
			print("\t\tpartial_clause_reg_%d = {%d'b0};" %(i, clauses), file=f)
		print("\tend", file=f)

		print("", file=f)
		print("\tHCB_top #(", file=f)
		print("\t\t.CLASS_NUM(CLASS_NUM),", file=f)
		print("\t\t.PACKETS_NUM(PACKETS_NUM),", file=f)
		print("\t\t.CLAUSE_NUM(CLAUSE_NUM),", file=f)
		print("\t\t.C_S00_AXIS_TDATA_WIDTH(C_S00_AXIS_TDATA_WIDTH)", file=f)
		print("\t)", file=f)
		print("\tHT(.clk(clk),", file=f)
		print("\t\t.x(x),", file=f)
		print("\t\t.valid(valid), ", file=f)
		print("\t\t.partial_clause(partial_clause_reg));", file=f)

		print("", file=f)
		print("\tlogic adder_done;", file=f)

		print("""

    always @(posedge clk) begin
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
	
    logic signed [WEIGHT_LENGTH - 1:0] class_sums_co [CLASS_NUM];
    logic signed [WEIGHT_LENGTH - 1:0] class_sums_org [CLASS_NUM];
    
    logic adder_done_co,adder_done_org;

    org_adder#(
    .CLAUSE_NUM(CLAUSE_NUM),
    .CLASS_NUM(CLASS_NUM),
    .WEIGHT_LENGTH(WEIGHT_LENGTH)
    )
    org_add(
    .clk(clk),
    .valid(valid[PACKETS_NUM]),
    .clause(partial_clause_reg),
    .class_sums(class_sums),
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
		.its_business_time(argmax),
		.clk(clk)
	);
	
endmodule""", file=f)

def vanilla_tm_hard_coded_blocks_top(filename):
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
		print("\tinput [C_S00_AXIS_TDATA_WIDTH - 1:0] x,", file=f)
		print("\tinput [PACKETS_NUM - 1:0] valid,", file=f)
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


def vanilla_tm_write_hard_coded_blocks(number_of_blocks, TAs, filename):
	starting_block = 0 
	finish_block = AXI_data_width*2

	# The raw clause expressions can be written for visual
	clause_expressions = output_dir + "/raw_clause_expressions.txt"
	print("	[RTL_gen][d]	Raw Clause Expressions have been written")
	with open(clause_expressions, "w") as clause_expressions_fp:
		for i in range(classes):
			for j in range(clauses):
				# the TAs may be arranged differently according to the type of implementation used 
				# for TMU the bit packed TAs are arranged as "[FEATURES][COMPLEMENTS]" whereas for
				# other implementatios they will be arranged as "[FEATURE]["COMPLEMENT"][FEATURE]..."
				# This tool curruently uses TMU so we are using the former method - however we have
				# also written the other way if needed - just switch the codes where it says.

				## If you used TMU then you can leave this bit as is 
				# l = " & ".join(["x[%d]" % (k) if (k<features) else "~x[%d]" % int((k/2))
				# 			for k in range(features*2) if TAs[i][j][k] == 1])
				# print("class %d clause %d: %s" % (i, j, l), file=clause_expressions_fp)				

				## If you didn't use TMU then you need to uncomment this bit:
				l = " & ".join(["x[%d]" % (k/2) if k %2 == 0 else "~x[%d]" % (int(k/2))
							for k in range(features*2) if TAs[i][j][k] == 1])
				print("class %d clause %d: %s" % (i, j, l), file=clause_expressions_fp)

	# we must also deal with clauses that have no includes - these are all exclude clauses 
	# in the code below we are taking the indexes for these clauses - we will set these as zero.
	all_exclude_indexes = []
	all_exc_count = 0	

	for i in range(TAs.shape[0]): 
		all_exc_class = []
		for j in range(TAs.shape[1]):
			if(any(v == 1 for v in TAs[i][j])):
				pass
			else:
				all_exc_class.append(j)
				all_exc_count += 1
		all_exclude_indexes.append(all_exc_class)
	print("	[RTL_gen][d]	No. All Exclude Clauses: ", all_exc_count	)

	# now the clauses will be divided into there respective HCBs
	# the same idea as above applies there as well.
	# This tool curruently uses TMU so we are using the former method - however we have
	# also written the other way if needed - just switch the codes where it says. 
	with open(filename, "w") as f:
		for i in range(number_of_blocks):
			TA_slice = []
			for c in range(classes):
				TA_slice.append(TAs[c][:, starting_block:finish_block])
			if finish_block >= number_of_features*2:	
				lit_range = (number_of_features*2 - starting_block)
			else:
				lit_range = AXI_data_width*2
			starting_block += AXI_data_width*2
			if finish_block + AXI_data_width*2 > number_of_features*2:	
				finish_block = starting_block + (number_of_features*2 - starting_block)
			else:
				finish_block += AXI_data_width*2
			if(i == 0):
				print("module HCB_%d (x, partial_clause, clk, valid);" %(i), file=f)
				print("\toutput\tlogic[%d:0] partial_clause [%d];" % (clauses-1, classes), file=f)
			else:
				print("module HCB_%d (x, partial_clause, partial_clause_prev, clk, valid);" %(i), file=f)
				print("\tinput\tlogic [%d:0] partial_clause_prev [%d];" % (clauses-1, classes), file=f)
				print("\toutput\tlogic[%d:0] partial_clause [%d];" % (clauses-1 ,classes), file=f)
			
			print("\tinput\tlogic clk;", file=f)
			print("\tinput\tlogic [%d:0] x;" % (AXI_data_width-1), file =f)	
			print("\tinput\tlogic valid;", file=f)

			print("\talways @(posedge clk) begin", file=f)
			print("\t\tif(valid) begin", file=f)
			for c in range(classes):
				TAs_ = TA_slice[c]
				print("\t\t\t// Class %d" % (c), file=f)
				for j in range(clauses):

					## If you didn't use TMU then you need to uncomment this bit:
					l = " & ".join(["x[%d]" % (k/2) if k %2 == 0 else "~x[%d]" % (int(k/2))
						for k in range(lit_range) if TAs_[j][k] == 1])

					if l == '': 
						if i != (number_of_blocks):
							if(i == 0): 
								if j in all_exclude_indexes[c]:
									print("\t\t\tpartial_clause[%d][%d] \t= 1'b0;" % (c,j), file=f)
								else:
									print("\t\t\tpartial_clause[%d][%d] \t= 1'b1;" % (c,j), file=f)
							else:
								print("\t\t\tpartial_clause[%d][%d] \t= partial_clause_prev[%d][%d] & 1'b1;" % (c,j, c,j), file=f)
					else:
						if(i==0):
							print("\t\t\tpartial_clause[%d][%d] \t= %s;" % (c, j, l), file=f)
						else:
							print("\t\t\tpartial_clause[%d][%d] \t= partial_clause_prev[%d][%d] & %s;" % (c,j, c,j, l), file=f)
			print("\t\tend",file=f)
			print("\tend", file=f)
			print("endmodule\n\n", file=f)	

# make directory for the RTL 
path = output_dir+"/RTL"
isExist = os.path.exists(path)

print("	----------------------------------------")
print("	    	Starting RTL Generation")
print("	----------------------------------------")
print("                        ")
print("	[W] = warning  [e] = error  [d] = debug ")
print("                        ")

if not isExist:
	print("	[RTL_gen]	Creating directory for RTL")
	os.makedirs(path)
else:
	print("	[RTL_gen]	The RTL dir exists - will be overwritten")	

if(TM_type == "Tsetlin Machine: Vanilla "):
	print("	[RTL_gen]	Generating RTL for Vanilla TM")
	print("	[RTL_gen][W] 	Weights are not permitted - only TA_file")

	# Read in the TA_file and generate the HCB (Hard Coded Clause Blocks - per class)
	TAs = np.loadtxt(TA_file, dtype=int)
	print("")
	print("	[RTL_gen][d]	Number of TAs: ", TAs.shape[0])
	print("	[RTL_gen][d]	Raw Values: ", TAs)

	# Assuming now that all the TAs have already been converted from their raw states
	# print("	[RTL_gen][d]	Max TA state: ", max(TAs))
	# print("	[RTL_gen][d]	Number of States:", no_states )
	# print("	[RTL_gen]	Converting to Inc/Exc")

	# for i in range(TAs.shape[0]):
	# 	if(TAs[i] <= (no_states/2)):
	# 		TAs[i] = 0
	# 	else:
	# 		TAs[i] = 1

	# print("	[RTL_gen]	Inc/Exc TAs: ", TAs)
	print("	[RTL_gen][d]	Number of Includes: ", np.count_nonzero(TAs))
	TAs = TAs.reshape(classes, clauses, features*2)
	print("	[RTL_gen][d]	TA new shape: ", TAs.shape)
	print("")
	print("	[RTL_gen][W]	Incorrect TA profiles produce incorrect hardware ;)")
	print("")

	print("	----------------------------------------")
	print("		Generating Your Model RTL")
	print("	----------------------------------------")
	# write the hard coded clause block code - this one is for vanilla TM
	hard_coded_blocks 	= output_dir + "/RTL/TM_Hard_Coded_Clause_Blocks.sv"

	vanilla_tm_write_hard_coded_blocks(number_of_blocks, TAs, hard_coded_blocks)
	print("")
	print("	[RTL_gen]	TM_Hard_Coded_Clause_Blocks.sv \tis written")

	# write the hard coded clause block top - this one is for vanilla TM
	hard_coded_top      = output_dir + "/RTL/HCB_top.sv"
	vanilla_tm_hard_coded_blocks_top(hard_coded_top)
	print("	[RTL_gen]	HCB_top.sv \t\tis written")

	# write the hard coded top block - this one is generic
	hard_coded_top      = output_dir + "/RTL/TM_top.sv"
	write_top(hard_coded_top)
	print("	[RTL_gen]	TM_top.sv \t\tis written")

	axis_wrapper_f		= output_dir + "/RTL/axis_wrapper.sv"
	vanilla_tm_write_axis_wrapper(axis_wrapper_f)
	print("	[RTL_gen]	axis_wrapper.sv \tis written")

	print("	----------------------------------------")
	print("")

	print("	----------------------------------------")
	print("		    Adjusting templates")
	print("	----------------------------------------")

	os.system("cp "+Matador_home_path+"/utils/RTL_templates/TM_argmax.sv "+ output_dir+"/RTL")
	print("	[RTL_gen]	TM_argmax.sv \t\tis added")
	
	os.system("cp "+Matador_home_path+"/utils/RTL_templates/decoder.sv "+ output_dir+"/RTL")
	print("	[RTL_gen]	decoder.sv \t\tis added")

	# new adder now corresponds to the Coalesced TM implementation
	# os.system("cp "+Matador_home_path+"/utils/RTL_templates/new_adder.sv "+ output_dir+"/RTL")
	# print("	[RTL_gen]	new_adder.sv \t\tis added")

	os.system("cp "+Matador_home_path+"/utils/RTL_templates/AXI_Interface.sv "+ output_dir+"/RTL")
	print("	[RTL_gen]	AXI_Interface.sv \tis added")

	print("	----------------------------------------")

	print("	----------------------------------------")
	print("		    Testbench generation			")
	print("	----------------------------------------")

	tb 	= output_dir + "/RTL/testbench.sv"
	pack_data()
	write_testbench(tb)





	print("")
	print("	[RTL_gen][d] 	If there are issues - check debug [d]")

# Note I haven't pushed the changes for the coalesced TM just yet
elif(TM_type == "coal"):
	print("	[RTL_gen]	Generating RTL for coalesced TM")
	print("	[RTL_gen][W]	Please ensure weight file is correct")

	# Read in the TA_file and generate the HCB (Hard Coded Clause Blocks)
	TAs = np.loadtxt(TA_file, dtype=int)
	print("")
	print("	[RTL_gen][d]	Number of TAs: ", TAs.shape[0])
	print("	[RTL_gen][d]	Raw Values: ", TAs)
	# print("	[RTL_gen][d]	Max TA state: ", max(TAs))
	# print("	[RTL_gen][d]	Number of States:", no_states )
	# print("	[RTL_gen]	Converting to Inc/Exc")

	# for i in range(TAs.shape[0]):
	# 	if(TAs[i] <= (no_states/2)):
	# 		TAs[i] = 0
	# 	else:
	# 		TAs[i] = 1

	# print("	[RTL_gen]	Inc/Exc TAs: ", TAs)
	print("	[RTL_gen][d]	Number of Includes: ", np.count_nonzero(TAs))
	TAs = TAs.reshape(classes, clauses, features*2)
	print("	[RTL_gen][d]	TA new shape: ", TAs.shape)
	print("")
	
	if(Weights_file == ""):
		print("	[RTL_gen][e]	Weight file was not found")
		print("	Exiting...")

	print("	[RTL_gen][d] If there are issues - check debug [d]")

else:
	print("	[RTL_gen][e]	TM type was not recognised - available options [vanilla, coal]")
	print("	Exiting...")


