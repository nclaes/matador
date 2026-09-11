`timescale 1ns/1ps
// =============================================================================
// redress_bram — simple dual-port synchronous BRAM (Xilinx BRAM-inferable)
// =============================================================================
// PURPOSE
//   Generic write-then-read memory, instantiated twice by redress_accel:
//   once as Instruction Memory (WIDTH=16, DEPTH=max_includes) and once as
//   the Feature FIFO (WIDTH=batch_width, DEPTH=max_features) -- see
//   Figure 5.13 of the source thesis. Same shape as vanilla_gp_tiled's
//   tile_mem.v (one write port, one synchronous-read port, common clock).
//
// XILINX BRAM MAPPING
//   The read is SYNCHRONOUS (registered, 1-cycle latency) -- this is what
//   lets Vivado map the array to block RAM instead of LUTRAM.
//
// COLLISION SEMANTICS
//   Loading (write) and inference (read) are mutually exclusive FSM phases
//   in redress_accel.v, so the same address is never written and read in
//   the same cycle. No write-first/read-first behaviour is relied upon.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module redress_bram #(
    parameter WIDTH = 16,
    parameter DEPTH = 4096,
    parameter ADDR_W = 12      // $clog2(DEPTH)
)(
    input  wire              clk,
    // ── write port (load path) ────────────────────────────────────────────
    input  wire               we,
    input  wire [ADDR_W-1:0]  waddr,
    input  wire [WIDTH-1:0]   wdata,
    // ── read port (inference path), 1-cycle latency ───────────────────────
    input  wire               re,
    input  wire [ADDR_W-1:0]  raddr,
    output reg  [WIDTH-1:0]   rdata
);

    (* ram_style = "block" *)
    reg [WIDTH-1:0] mem [0:DEPTH-1];

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
        if (re)
            rdata <= mem[raddr];
    end

endmodule
