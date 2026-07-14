`timescale 1ns/1ps
// =============================================================================
// axis_fifo — synchronous AXI-Stream FIFO
// =============================================================================
// PURPOSE
//   Buffers incoming AXI-Stream beats so that the producer does not need to
//   synchronise with the FSM state in tm_accelerator. In the TM accelerator
//   context this allows all feature beats to be pushed before the FSM has
//   finished any previous housekeeping.
//
// IMPLEMENTATION
//   Two-pointer circular buffer. wptr and rptr are one bit wider than the
//   address so that full (pointers equal modulo depth but differ in MSB) is
//   distinguished from empty (pointers exactly equal) without a separate
//   occupancy counter.
//
//   Output is REGISTERED (not fall-through): m_tdata/m_tvalid appear one
//   cycle after the read pointer advances. The tm_accelerator FSM accounts
//   for this latency by sampling m_tdata in the clock after m_tvalid rises.
//
// HANDSHAKE
//   AXI-Stream: a beat transfers on a rising clock edge where
//   s_tvalid AND s_tready are both high.
//   s_tready is combinatorially asserted whenever the FIFO is not full.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// Depth must be a power of 2 (required for two-pointer full/empty logic).
// =============================================================================
module axis_fifo #(
    parameter DATA_WIDTH = 32,
    parameter DEPTH      = 16    // must be power of 2
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  s_tvalid,
    output wire                  s_tready,
    input  wire [DATA_WIDTH-1:0] s_tdata,
    input  wire                  s_tlast,
    output reg                   m_tvalid,
    input  wire                  m_tready,
    output reg  [DATA_WIDTH-1:0] m_tdata,
    output reg                   m_tlast
);

    localparam AW  = $clog2(DEPTH);
    localparam DW1 = DATA_WIDTH + 1;  // +1 for tlast

    reg [DW1-1:0] mem  [0:DEPTH-1];
    reg [AW:0]    wptr;
    reg [AW:0]    rptr;

    wire full  = (wptr[AW] != rptr[AW]) & (wptr[AW-1:0] == rptr[AW-1:0]);
    wire empty = (wptr == rptr);

    assign s_tready = ~full;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr     <= {(AW+1){1'b0}};
            rptr     <= {(AW+1){1'b0}};
            m_tvalid <= 1'b0;
            m_tdata  <= {DATA_WIDTH{1'b0}};
            m_tlast  <= 1'b0;
        end else begin
            if (s_tvalid & ~full) begin
                mem[wptr[AW-1:0]] <= {s_tlast, s_tdata};
                wptr <= wptr + 1'b1;
            end
            if (~m_tvalid | m_tready) begin
                if (~empty) begin
                    {m_tlast, m_tdata} <= mem[rptr[AW-1:0]];
                    rptr     <= rptr + 1'b1;
                    m_tvalid <= 1'b1;
                end else begin
                    m_tvalid <= 1'b0;
                end
            end
        end
    end

endmodule
