module axis_wrapper_top #
	(
		parameter integer DEPTH = 1,
		parameter integer WIDTH = 1,
		parameter integer WEIGHT_LENGTH = 16,
		parameter integer PACKETS = 13,
		parameter integer C_S00_AXIS_TDATA_WIDTH = 64,
		parameter integer C_M00_AXIS_TDATA_WIDTH = 64,
		// design configurations
		parameter STAGE_NUM = 4,
		parameter CLAUSE_NUM = 100,
		parameter CLASS_NUM = 10,
		parameter FEATURE_NUM = 784,
		parameter PACKETS_NUM = (FEATURE_NUM - 1)/C_S00_AXIS_TDATA_WIDTH + 1,
		parameter VANILLA = 1,
		parameter COALESCED = 0
	)
	(

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
	
endmodule
