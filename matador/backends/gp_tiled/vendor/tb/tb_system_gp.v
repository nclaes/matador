`timescale 1ns/1ps
// =============================================================================
// tb_system_gp — end-to-end test of the general-purpose (reprogrammable) core
// =============================================================================
// PHASES
//   A  INFER before any LOAD                       -> ERR_NOCFG ack
//   B  emulator-generated scenario (gp_stimulus/gp_expected.memh):
//        LOAD frozen model -> INFER the 10 original tb_system vectors
//        LOAD small model (4 cls x 8 cpc, 96 feat) -> INFER 6 random frames
//        LOAD frozen model again                   -> INFER 1 vector
//      Every expected beat (acks + predictions) comes from tm_emulator.py.
//   C  directed protocol errors + recovery:
//        bad magic, bad command, geometry out of range,
//        LOAD underrun (early TLAST)  -> model invalidated -> INFER = NOCFG
//        LOAD overrun  (late  TLAST)  -> model invalidated -> INFER = NOCFG
//        clean re-LOAD + INFER        -> recovered
//        INFER frame underrun         -> error, but model stays configured
//   D  spurious-beat watch + total handshake-count check
//
// STRESS APPLIED THROUGHOUT
//   * m_axis_tready runs a 2-of-3 duty pattern the entire test, covering the
//     "tready pulses before tvalid" and "tready low when tvalid rises" cases
//     that the hold-until-accept handshake must survive without losing or
//     duplicating beats.
//   * the sender inserts a 2-cycle tvalid gap after every 7th beat, forcing
//     FIFO-empty stalls inside S_CFG/S_LOAD/S_RECV/S_DRAIN.
//   * handshakes are sampled race-free via negedge shadow registers.
//
// Run from sim/ (paths to vectors/ are relative), after:
//   python3 ../py/tm_emulator.py genvec
// =============================================================================
module tb_system_gp;

    parameter AXIS_DATA_WIDTH = 32;
    parameter CLASS_WIDTH     = 4;
    parameter TO_MAX          = 30000;   // cycles to wait for one output beat

    `include "vectors/gp_sizes.vh"

    // ── DUT ────────────────────────────────────────────────────────────────
    reg                        clk, rst_n;
    reg                        s_tvalid;
    wire                       s_tready;
    reg  [AXIS_DATA_WIDTH-1:0] s_tdata;
    reg                        s_tlast;
    wire                       m_tvalid;
    wire                       m_tready;
    wire [AXIS_DATA_WIDTH-1:0] m_tdata;
    wire                       m_tlast;
    wire                       busy, configured;

    tm_accel_gp dut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
        .s_axis_tdata(s_tdata),   .s_axis_tlast(s_tlast),
        .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
        .m_axis_tdata(m_tdata),   .m_axis_tlast(m_tlast),
        .busy(busy), .configured(configured)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    // ── continuous output back-pressure: ready 2 cycles of every 3 ─────────
    integer cyc;
    initial cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;
    assign m_tready = ((cyc % 3) != 0);

    // ── race-free handshake sampling (negedge shadows) ─────────────────────
    reg                        rdy_s;                    // s_tready pre-edge
    reg                        mv_s, mr_s, ml_s;
    reg [AXIS_DATA_WIDTH-1:0]  md_s;
    always @(negedge clk) begin
        rdy_s <= s_tready;
        mv_s  <= m_tvalid;
        mr_s  <= m_tready;
        md_s  <= m_tdata;
        ml_s  <= m_tlast;
    end

    integer hs_cnt;                                       // every m_axis beat
    initial hs_cnt = 0;
    always @(posedge clk) if (mv_s && mr_s) hs_cnt <= hs_cnt + 1;

    // ── vectors ────────────────────────────────────────────────────────────
    reg [32:0] stim   [0:GP_N_STIM-1];        // {tlast, data}
    reg [32:0] expc   [0:GP_N_EXP-1];
    reg [32:0] sload  [0:GP_N_SMALL-1];
    reg [32:0] sinfer [0:GP_N_SMALL_INFER-1];

    // ── sender ─────────────────────────────────────────────────────────────
    integer sent;
    initial sent = 0;

    task send_beat;
        input [31:0] d;
        input        l;
        begin
            s_tvalid = 1'b1; s_tdata = d; s_tlast = l;
            @(posedge clk);
            while (!rdy_s) @(posedge clk);   // pre-edge ready => beat consumed
            // Deassert after EVERY beat: back-to-back calls re-assert within
            // the same timestep (no throughput loss), and a caller that stops
            // to wait for a response can never re-write the last word.
            s_tvalid = 1'b0; s_tlast = 1'b0;
            sent = sent + 1;
            if (sent % 7 == 0)               // periodic 2-cycle tvalid gap
                repeat (2) @(posedge clk);
        end
    endtask

    task send_idle;
        begin
            s_tvalid = 1'b0; s_tlast = 1'b0;
            @(posedge clk);
        end
    endtask

    // ── receiver ───────────────────────────────────────────────────────────
    integer fail_cnt;

    task expect_beat;
        input [31:0]  d;
        input         l;
        input [255:0] tag;      // ASCII label
        input integer idx;      // index within the label (-1 to omit)
        integer to;
        begin
            to = 0;
            @(posedge clk);
            while (!(mv_s && mr_s)) begin
                @(posedge clk);
                to = to + 1;
                if (to >= TO_MAX) begin
                    $display("TIMEOUT waiting for %0s[%0d]", tag, idx);
                    fail_cnt = fail_cnt + 1;
                    $finish;
                end
            end
            if (md_s !== d || ml_s !== l) begin
                $display("FAIL %0s[%0d]: exp data=%08h last=%b  got data=%08h last=%b",
                         tag, idx, d, l, md_s, ml_s);
                fail_cnt = fail_cnt + 1;
            end else begin
                $display("PASS %0s[%0d]: data=%08h last=%b", tag, idx, md_s, ml_s);
            end
        end
    endtask

    // ── protocol constants (mirror the DUT) ────────────────────────────────
    localparam [31:0] HDR_INFER = 32'hA501_0000;
    localparam [31:0] HDR_LOAD  = 32'hA502_0000;
    localparam [31:0] ACK_NOCFG = 32'hA5E5_0000;
    localparam [31:0] ACK_RANGE = 32'hA5E2_0000;

    integer i, exp_total;

    initial begin
        $dumpfile("tb_system_gp.vcd");
        $dumpvars(1, tb_system_gp);
        $dumpvars(1, dut);

        $readmemh("vectors/gp_stimulus.memh",    stim);
        $readmemh("vectors/gp_expected.memh",    expc);
        $readmemh("vectors/gp_small_load.memh",  sload);
        $readmemh("vectors/gp_small_infer.memh", sinfer);

        fail_cnt = 0; exp_total = 0;
        rst_n = 0; s_tvalid = 0; s_tdata = 0; s_tlast = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ══ PHASE A: inference before any load ═══════════════════════════
        $display("── phase A: INFER before LOAD");
        send_beat(HDR_INFER, 1'b0);
        send_beat(32'h0000_0000, 1'b0);
        send_beat(32'h0000_0000, 1'b1);
        expect_beat(ACK_NOCFG, 1'b1, "A.nocfg", -1);
        exp_total = exp_total + 1;

        // ══ PHASE B: emulator-generated scenario (fork sender/receiver) ══
        $display("── phase B: scripted loads + inference (%0d beats in, %0d out)",
                 GP_N_STIM, GP_N_EXP);
        fork
            begin
                for (i = 0; i < GP_N_STIM; i = i + 1)
                    send_beat(stim[i][31:0], stim[i][32]);
                s_tvalid = 1'b0; s_tlast = 1'b0;
            end
            begin : phaseB_rx
                integer e;
                for (e = 0; e < GP_N_EXP; e = e + 1)
                    expect_beat(expc[e][31:0], expc[e][32], "B.exp", e);
            end
        join
        exp_total = exp_total + GP_N_EXP;
        if (!configured) begin
            $display("FAIL B: configured should be 1 after scripted loads");
            fail_cnt = fail_cnt + 1;
        end

        // ══ PHASE C: directed protocol errors + recovery ══════════════════
        $display("── phase C: directed errors");

        // C1: bad magic, single-beat packet
        send_beat(32'hDEAD_0000, 1'b1);
        expect_beat(32'hA5E0_00DE, 1'b1, "C1.magic", -1);
        exp_total = exp_total + 1;

        // C2: unknown command, multi-beat packet (exercises S_DRAIN)
        send_beat(32'hA57F_0000, 1'b0);
        send_beat(32'h1111_1111, 1'b0);
        send_beat(32'h2222_2222, 1'b1);
        expect_beat(32'hA5E1_007F, 1'b1, "C2.cmd", -1);
        exp_total = exp_total + 1;

        // C3: geometry out of range (n_classes = 0) -> drain -> ERR_RANGE
        send_beat(HDR_LOAD, 1'b0);
        send_beat(32'h0004_0800, 1'b0);    // classes=0, cpc=8, thr=4
        send_beat(32'h0001_0303, 1'b0);    // beats=3, nfs=3, ncs=1
        send_beat(32'h0003_0020, 1'b0);    // nct=32, ntiles=3
        send_beat(32'hBAD0_BAD0, 1'b0);
        send_beat(32'hBAD1_BAD1, 1'b1);
        expect_beat(ACK_RANGE, 1'b1, "C3.range", -1);
        exp_total = exp_total + 1;

        // C4: LOAD underrun — TLAST after 20 beats of the small-model packet
        for (i = 0; i < 20; i = i + 1)
            send_beat(sload[i][31:0], (i == 19));
        expect_beat(32'hA5E3_0000, 1'b1, "C4.underrun", -1);
        exp_total = exp_total + 1;
        //     model must now be invalid
        for (i = 0; i < GP_N_SMALL_INFER; i = i + 1)
            send_beat(sinfer[i][31:0], sinfer[i][32]);
        expect_beat(ACK_NOCFG, 1'b1, "C4.nocfg_after", -1);
        exp_total = exp_total + 1;

        // C5: LOAD overrun — full payload, TLAST withheld, one extra word
        for (i = 0; i < GP_N_SMALL; i = i + 1)
            send_beat(sload[i][31:0], 1'b0);            // stripped TLAST
        send_beat(32'hFEED_FACE, 1'b1);                 // stray extra beat
        expect_beat({16'hA5E4, 16'd3}, 1'b1, "C5.overrun", -1);  // info = n_tiles
        exp_total = exp_total + 1;
        for (i = 0; i < GP_N_SMALL_INFER; i = i + 1)
            send_beat(sinfer[i][31:0], sinfer[i][32]);
        expect_beat(ACK_NOCFG, 1'b1, "C5.nocfg_after", -1);
        exp_total = exp_total + 1;

        // C6: clean re-LOAD + INFER — full recovery
        for (i = 0; i < GP_N_SMALL; i = i + 1)
            send_beat(sload[i][31:0], sload[i][32]);
        expect_beat(GP_SMALL_ACK, 1'b1, "C6.reload_ack", -1);
        exp_total = exp_total + 1;
        for (i = 0; i < GP_N_SMALL_INFER; i = i + 1)
            send_beat(sinfer[i][31:0], sinfer[i][32]);
        expect_beat(GP_SMALL_INFER_EXP, 1'b1, "C6.reload_infer", -1);
        exp_total = exp_total + 1;

        // C7: INFER frame underrun — model must SURVIVE inference errors
        send_beat(HDR_INFER, 1'b0);
        send_beat(32'h0000_0001, 1'b1);                 // 1 of GP_SMALL_NBEATS
        expect_beat(32'hA5E3_0000, 1'b1, "C7.frame_underrun", -1);
        exp_total = exp_total + 1;
        if (!configured) begin
            $display("FAIL C7: inference error must not invalidate the model");
            fail_cnt = fail_cnt + 1;
        end
        for (i = 0; i < GP_N_SMALL_INFER; i = i + 1)
            send_beat(sinfer[i][31:0], sinfer[i][32]);
        expect_beat(GP_SMALL_INFER_EXP, 1'b1, "C7.infer_after_err", -1);
        exp_total = exp_total + 1;

        // ══ PHASE D: spurious-beat watch + bookkeeping ════════════════════
        $display("── phase D: spurious-beat watch");
        s_tvalid = 1'b0; s_tlast = 1'b0;
        for (i = 0; i < 400; i = i + 1) begin
            @(posedge clk);
            if (mv_s && mr_s) begin
                $display("FAIL D: spurious output beat data=%08h", md_s);
                fail_cnt = fail_cnt + 1;
            end
        end
        if (hs_cnt !== exp_total) begin
            $display("FAIL D: handshake count %0d != expected %0d (lost or duplicated beats)",
                     hs_cnt, exp_total);
            fail_cnt = fail_cnt + 1;
        end
        if (busy) begin
            $display("FAIL D: busy stuck high at end of test");
            fail_cnt = fail_cnt + 1;
        end

        if (fail_cnt == 0)
            $display("tb_system_gp: ALL TESTS PASSED (%0d output beats checked)",
                     exp_total);
        else
            $display("tb_system_gp: FAILED (%0d errors)", fail_cnt);
        $finish;
    end

    // global watchdog
    initial begin
        #80_000_000;
        $display("TIMEOUT: global watchdog expired");
        $finish;
    end

endmodule
