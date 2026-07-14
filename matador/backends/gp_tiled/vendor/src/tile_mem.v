`timescale 1ns/1ps
// =============================================================================
// tile_mem — reprogrammable tile memory (Xilinx BRAM-inferable)
// =============================================================================
// PURPOSE
//   Replaces the generation-time tile ROM of tm_accelerator.v with a runtime
//   writable memory so new TA Include/Exclude models can be streamed in over
//   AXI-Stream (see tm_accel_gp.v). Same row layout as the original ROM:
//     row address = feat_slice * n_clause_slices + clause_slice  (linear)
//     within a row, clause lane k occupies bits [k*2*FEAT_SLICE +: 2*FEAT_SLICE]
//       low  FEAT_SLICE bits = Include for positive literals of the window
//       high FEAT_SLICE bits = Include for negated  literals of the window
//
// XILINX BRAM MAPPING
//   * Simple dual port: one write port (model loader), one read port
//     (inference datapath), common clock.
//   * The read is SYNCHRONOUS (registered, 1-cycle latency) — this is what
//     lets Vivado map the array to block RAM instead of LUTRAM. The original
//     ROM's asynchronous read forced distributed RAM; the S_COMPUTE loop in
//     tm_accel_gp.v is pipelined by one stage to absorb the added latency.
//   * (* ram_style = "block" *) pins the choice. At the default geometry
//     (2048 x 128 = 256 Kb) Vivado builds the array from parallel BRAM36
//     columns (~29 x RAMB36 in SDP 72-bit mode, depth 512, partially used).
//     Set ram_style = "ultra" on UltraScale+ if URAM is preferred.
//   * Read enable (re) maps to the BRAM port enable: rdata HOLDS its value
//     on cycles where re=0.
//
// COLLISION SEMANTICS
//   Load and inference are mutually exclusive FSM phases in tm_accel_gp.v,
//   so the same address is never written and read in one cycle. No
//   write-first/read-first behaviour is relied upon.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module tile_mem #(
    parameter TILE_WIDTH = 2048,
    parameter N_TILES    = 128,
    parameter ADDR_W     = 7      // $clog2(N_TILES)
)(
    input  wire                  clk,
    // ── write port (model load path) ─────────────────────────────────────
    input  wire                  we,
    input  wire [ADDR_W-1:0]     waddr,
    input  wire [TILE_WIDTH-1:0] wdata,
    // ── read port (inference path), 1-cycle latency ──────────────────────
    input  wire                  re,
    input  wire [ADDR_W-1:0]     raddr,
    output reg  [TILE_WIDTH-1:0] rdata
);

    (* ram_style = "block" *)
    reg [TILE_WIDTH-1:0] mem [0:N_TILES-1];

    always @(posedge clk) begin
        if (we)
            mem[waddr] <= wdata;
        if (re)
            rdata <= mem[raddr];
    end

endmodule
