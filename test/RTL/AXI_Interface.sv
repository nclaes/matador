`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 06/24/2023 03:24:09 AM
// Design Name: Tousif Rahman, Gang Mao
// Module Name: axis_adder_v1_0_S00_AXIS
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


`resetall
`timescale 1ns / 1ps
`default_nettype none

/*
 * AXI4-Stream SRL-based FIFO
 */
module axis_adder_v1_0_S00_AXIS #
(
    // Width of AXI stream interfaces in bits
    parameter DATA_WIDTH = 64,
    // Propagate tkeep signal
    parameter KEEP_ENABLE = (DATA_WIDTH>8),
    // tkeep signal width (words per cycle)
    parameter KEEP_WIDTH = ((DATA_WIDTH+7)/8),
    // Propagate tlast signal
    parameter LAST_ENABLE = 1,
    // Propagate tid signal
    parameter ID_ENABLE = 0,
    // tid signal width
    parameter ID_WIDTH = 8,
    // Propagate tdest signal
    parameter DEST_ENABLE = 0,
    // tdest signal width
    parameter DEST_WIDTH = 8,
    // Propagate tuser signal
    parameter USER_ENABLE = 1,
    // tuser signal width
    parameter USER_WIDTH = 1,
    // FIFO depth in cycles
    parameter DEPTH = 1,
    
    parameter PACKETS_NUM
)
(
    input  wire                       clk,
    input  wire                       rst,

    /*
     * AXI input
     */
    input  wire [DATA_WIDTH-1:0]      s_axis_tdata,
    input  wire [KEEP_WIDTH-1:0]      s_axis_tkeep,
    input  wire                       s_axis_tvalid,
    output wire                       s_axis_tready,
    input  wire                       s_axis_tlast,
    input  wire [ID_WIDTH-1:0]        s_axis_tid,
    input  wire [DEST_WIDTH-1:0]      s_axis_tdest,
    input  wire [USER_WIDTH-1:0]      s_axis_tuser,
    output wire full,
    /*
     * AXI output
     */
    output reg [DATA_WIDTH-1:0]      m_axis_tdata,
    output wire [KEEP_WIDTH-1:0]      m_axis_tkeep,
    output reg                       m_axis_tvalid,
    input  wire                       m_axis_tready,
    output wire                       m_axis_tlast,
    output wire [ID_WIDTH-1:0]        m_axis_tid,
    output wire [DEST_WIDTH-1:0]      m_axis_tdest,
    output wire [USER_WIDTH-1:0]      m_axis_tuser,
    output reg [PACKETS_NUM - 1:0] valid,
    /*
     * Status
     */
    output wire [$clog2(DEPTH+1)-1:0] count
);

localparam KEEP_OFFSET = DATA_WIDTH;
localparam LAST_OFFSET = KEEP_OFFSET + (KEEP_ENABLE ? KEEP_WIDTH : 0);
localparam ID_OFFSET   = LAST_OFFSET + (LAST_ENABLE ? 1          : 0);
localparam DEST_OFFSET = ID_OFFSET   + (ID_ENABLE   ? ID_WIDTH   : 0);
localparam USER_OFFSET = DEST_OFFSET + (DEST_ENABLE ? DEST_WIDTH : 0);
localparam WIDTH       = USER_OFFSET + (USER_ENABLE ? USER_WIDTH : 0);

reg [WIDTH-1:0] data_reg[DEPTH-1:0];
reg [$clog2(DEPTH+1)-1:0] ptr_reg = 0;
reg full_reg = 1'b0, full_next;
reg empty_reg = 1'b1, empty_next;
reg s_axis_ready;

wire [WIDTH-1:0] s_axis;

wire [WIDTH-1:0] m_axis = data_reg[ptr_reg-1];

assign s_axis_tready = s_axis_ready;

generate
    assign s_axis[DATA_WIDTH-1:0] = s_axis_tdata;
    if (KEEP_ENABLE) assign s_axis[KEEP_OFFSET +: KEEP_WIDTH] = s_axis_tkeep;
    if (LAST_ENABLE) assign s_axis[LAST_OFFSET]               = s_axis_tlast;
    if (ID_ENABLE)   assign s_axis[ID_OFFSET   +: ID_WIDTH]   = s_axis_tid;
    if (DEST_ENABLE) assign s_axis[DEST_OFFSET +: DEST_WIDTH] = s_axis_tdest;
    if (USER_ENABLE) assign s_axis[USER_OFFSET +: USER_WIDTH] = s_axis_tuser;
endgenerate

//assign m_axis_tvalid = !empty_reg;

//assign m_axis_tdata = m_axis[DATA_WIDTH-1:0];
assign m_axis_tkeep = KEEP_ENABLE ? m_axis[KEEP_OFFSET +: KEEP_WIDTH] : {KEEP_WIDTH{1'b1}};
assign m_axis_tlast = LAST_ENABLE ? m_axis[LAST_OFFSET]               : 1'b1;
assign m_axis_tid   = ID_ENABLE   ? m_axis[ID_OFFSET   +: ID_WIDTH]   : {ID_WIDTH{1'b0}};
assign m_axis_tdest = DEST_ENABLE ? m_axis[DEST_OFFSET +: DEST_WIDTH] : {DEST_WIDTH{1'b0}};
assign m_axis_tuser = USER_ENABLE ? m_axis[USER_OFFSET +: USER_WIDTH] : {USER_WIDTH{1'b0}};

assign count = ptr_reg;




reg stat;
reg op_stat;
reg old_tvalid;
reg old_m_tready;
reg [12:0] valid_reg;
reg [12:0] old_valid_reg;
reg old_s_axis_tlast;
reg inf_stat;
integer i;

initial begin
//    s_axis_ready = 0; 
//    for (i = 0; i < DEPTH; i = i + 1) begin
//        data_reg[i] <= 0;
//    end
//    valid = 13'b0000000000001;
    old_tvalid = 0;
    old_s_axis_tlast = 0;
end
assign full = full_reg;

integer flag;
always @(posedge clk) begin
    old_tvalid <= s_axis_tvalid;
    old_s_axis_tlast <= s_axis_tlast;
    old_m_tready <= m_axis_tready;
    if (old_tvalid && s_axis_tvalid) begin
//        old_valid_reg <= valid_reg;
//        valid <= old_valid_reg;
        valid <= valid_reg;
    end

    if (rst) begin
        //valid_reg <= 13'b0000000000001;
        valid_reg <= {{(PACKETS_NUM - 1){1'b0}},{1'b1}};
        stat = 0;
        flag = 0;
        inf_stat = 0;
    end
    else begin  
        
        if (!old_s_axis_tlast && s_axis_tlast) begin
            //valid_reg <= 13'b0000000000000;
            inf_stat = 0;
        end
        else 
        if (!old_tvalid && s_axis_tvalid) begin
            //valid_reg <= 13'b0000000000001;
            inf_stat = 1;
//            valid_reg <= {valid_reg[11:0],valid_reg[12]};
        end
        else if (old_tvalid && s_axis_tvalid && m_axis_tready) begin
            m_axis_tdata = s_axis_tdata;
            valid_reg <= {valid_reg[11:0],valid_reg[12]};
            s_axis_ready = 1;
            //flag = 1;
        end
        else if (!s_axis_tvalid && old_tvalid) begin
            s_axis_ready = 0;
            //flag = 2;
        end
        else if (!m_axis_tready && old_m_tready) begin
            s_axis_ready = 0;
        end
    end
end

endmodule

`resetall
