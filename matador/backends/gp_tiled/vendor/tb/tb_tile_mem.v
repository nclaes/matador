`timescale 1ns/1ps
// tb_tile_mem — unit test: write/readback, 1-cycle read latency, re gating,
//               simultaneous read+write on different addresses
module tb_tile_mem;
    parameter TILE_WIDTH = 2048;
    parameter N_TILES    = 128;
    parameter ADDR_W     = 7;

    reg                   clk;
    reg                   we, re;
    reg  [ADDR_W-1:0]     waddr, raddr;
    reg  [TILE_WIDTH-1:0] wdata;
    wire [TILE_WIDTH-1:0] rdata;

    tile_mem #(.TILE_WIDTH(TILE_WIDTH), .N_TILES(N_TILES), .ADDR_W(ADDR_W)) dut (
        .clk(clk), .we(we), .waddr(waddr), .wdata(wdata),
        .re(re), .raddr(raddr), .rdata(rdata)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // address-dependent pattern wide enough to exercise all 2048 bits
    function [TILE_WIDTH-1:0] pat;
        input [ADDR_W-1:0] a;
        integer w;
        begin
            pat = {TILE_WIDTH{1'b0}};
            for (w = 0; w < TILE_WIDTH/32; w = w + 1)
                pat[w*32 +: 32] = {a, 9'h0AB, w[7:0], a} ^ (32'h9E3779B9 * (w + 1));
        end
    endfunction

    integer fail_cnt, i;
    reg [TILE_WIDTH-1:0] held;

    initial begin
        $dumpfile("tb_tile_mem.vcd");
        $dumpvars(0, tb_tile_mem);
        fail_cnt = 0;
        we = 0; re = 0; waddr = 0; raddr = 0; wdata = 0;
        repeat (2) @(posedge clk);

        // ── write every row ─────────────────────────────────────────────
        for (i = 0; i < N_TILES; i = i + 1) begin
            @(negedge clk);
            we = 1; waddr = i[ADDR_W-1:0]; wdata = pat(i[ADDR_W-1:0]);
        end
        @(negedge clk); we = 0;

        // ── read back every row (checks 1-cycle latency) ────────────────
        for (i = 0; i < N_TILES; i = i + 1) begin
            @(negedge clk);
            re = 1; raddr = i[ADDR_W-1:0];
            @(negedge clk);
            re = 0;
            if (rdata !== pat(i[ADDR_W-1:0])) begin
                $display("FAIL readback row %0d", i);
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── re gating: rdata must HOLD when re=0 ────────────────────────
        @(negedge clk); re = 1; raddr = 7'd5;
        @(negedge clk); re = 0; raddr = 7'd9;   // address changes, re low
        held = rdata;
        @(negedge clk);
        if (rdata !== held || rdata !== pat(7'd5)) begin
            $display("FAIL re-gating: rdata changed while re=0");
            fail_cnt = fail_cnt + 1;
        end

        // ── simultaneous write(rowA) + read(rowB) ───────────────────────
        @(negedge clk);
        we = 1; waddr = 7'd20; wdata = {TILE_WIDTH{1'b1}};
        re = 1; raddr = 7'd21;
        @(negedge clk);
        we = 0; re = 0;
        if (rdata !== pat(7'd21)) begin
            $display("FAIL simultaneous r/w: read port corrupted");
            fail_cnt = fail_cnt + 1;
        end
        @(negedge clk); re = 1; raddr = 7'd20;
        @(negedge clk); re = 0;
        if (rdata !== {TILE_WIDTH{1'b1}}) begin
            $display("FAIL simultaneous r/w: write did not land");
            fail_cnt = fail_cnt + 1;
        end

        // ── overwrite a row (reprogramming) ─────────────────────────────
        @(negedge clk); we = 1; waddr = 7'd5; wdata = pat(7'd77);
        @(negedge clk); we = 0; re = 1; raddr = 7'd5;
        @(negedge clk); re = 0;
        if (rdata !== pat(7'd77)) begin
            $display("FAIL overwrite: old data survived");
            fail_cnt = fail_cnt + 1;
        end

        if (fail_cnt == 0) $display("tb_tile_mem: ALL PASSED");
        else               $display("tb_tile_mem: FAILED (%0d errors)", fail_cnt);
        $finish;
    end
endmodule
