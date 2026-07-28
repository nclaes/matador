`timescale 1ns/1ps
// tb_score_acc_rt — unit test: unclamped accumulation (threshold is
//                   vestigial and must have zero effect), inactive holds,
//                   clear, unused-class isolation
module tb_score_acc_rt;
    parameter N_CLASSES   = 16;
    parameter SCORE_WIDTH = 6;

    reg                              clk, rst_n;
    reg                              clear, valid, polarity, active;
    reg  [3:0]                       cls;
    reg  [SCORE_WIDTH-1:0]           threshold;
    wire [SCORE_WIDTH*N_CLASSES-1:0] scores_flat;

    score_acc_rt #(.N_CLASSES(N_CLASSES), .SCORE_WIDTH(SCORE_WIDTH)) dut (
        .clk(clk), .rst_n(rst_n), .clear(clear),
        .valid(valid), .cls(cls), .polarity(polarity), .active(active),
        .threshold(threshold), .scores_flat(scores_flat)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    function signed [SCORE_WIDTH-1:0] score_of;
        input integer c;
        begin score_of = scores_flat[c*SCORE_WIDTH +: SCORE_WIDTH]; end
    endfunction

    integer fail_cnt, i, t;

    task pump;                       // n votes on class c (unguarded)
        input integer n;
        input [3:0]   c;
        input         pol;
        begin
            for (i = 0; i < n; i = i + 1) begin
                valid = 1; cls = c; polarity = pol; active = 1; @(posedge clk);
            end
            valid = 0; @(posedge clk);
        end
    endtask

    task do_clear;
        begin clear = 1; @(posedge clk); clear = 0; @(posedge clk); end
    endtask

    initial begin
        $dumpfile("tb_score_acc_rt.vcd");
        $dumpvars(0, tb_score_acc_rt);
        fail_cnt = 0;
        clear = 0; valid = 0; polarity = 0; active = 0; cls = 0; threshold = 6'd8;
        rst_n = 0; repeat(4) @(posedge clk);
        rst_n = 1; @(posedge clk);

        // ── unclamped accumulation, threshold is vestigial: sweep several
        //    runtime threshold values (1, 3, 8, 31) while accumulating the
        //    SAME 20 votes each time -- the result must not depend on
        //    threshold at all, proving the field has no effect on the
        //    accumulator ──────────────────────────────────────────────────
        for (t = 0; t < 4; t = t + 1) begin
            case (t)
                0: threshold = 6'd1;
                1: threshold = 6'd3;
                2: threshold = 6'd8;
                default: threshold = 6'd31;
            endcase
            do_clear;
            pump(20, 4'd0, 1'b1);
            if ($signed(score_of(0)) !== 20) begin
                $display("FAIL accum+ threshold=%0d: exp=20 got=%0d", threshold, $signed(score_of(0)));
                fail_cnt = fail_cnt + 1;
            end
            do_clear;
            pump(20, 4'd0, 1'b0);
            if ($signed(score_of(0)) !== -20) begin
                $display("FAIL accum- threshold=%0d: exp=-20 got=%0d", threshold, $signed(score_of(0)));
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── mixed polarity bookkeeping: +5 then -2 = +3 ──────────────────
        threshold = 6'd8; do_clear;
        pump(5, 4'd2, 1'b1);
        pump(2, 4'd2, 1'b0);
        if ($signed(score_of(2)) !== 3) begin
            $display("FAIL mixed: exp=3 got=%0d", $signed(score_of(2)));
            fail_cnt = fail_cnt + 1;
        end

        // ── inactive vote holds the score ────────────────────────────────
        valid = 1; cls = 4'd2; polarity = 1'b1; active = 0; @(posedge clk);
        valid = 0; @(posedge clk);
        if ($signed(score_of(2)) !== 3) begin
            $display("FAIL inactive: score changed");
            fail_cnt = fail_cnt + 1;
        end

        // ── other classes untouched by class-2 traffic ───────────────────
        for (i = 0; i < N_CLASSES; i = i + 1) begin
            if (i != 2 && $signed(score_of(i)) !== 0) begin
                $display("FAIL isolation: class %0d = %0d", i, $signed(score_of(i)));
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── clear zeros everything ───────────────────────────────────────
        do_clear;
        for (i = 0; i < N_CLASSES; i = i + 1) begin
            if ($signed(score_of(i)) !== 0) begin
                $display("FAIL clear: class %0d = %0d", i, $signed(score_of(i)));
                fail_cnt = fail_cnt + 1;
            end
        end

        // ── threshold changing mid-flight still has no effect ────────────
        threshold = 6'd8; do_clear;
        pump(8, 4'd1, 1'b1);                          // score = 8
        threshold = 6'd3; @(posedge clk);
        pump(1, 4'd1, 1'b1);                          // unclamped: score = 9
        if ($signed(score_of(1)) !== 9) begin
            $display("FAIL mid-flight+: exp=9 got=%0d", $signed(score_of(1)));
            fail_cnt = fail_cnt + 1;
        end
        pump(1, 4'd1, 1'b0);                          // unclamped: score = 8
        if ($signed(score_of(1)) !== 8) begin
            $display("FAIL mid-flight-: exp=8 got=%0d", $signed(score_of(1)));
            fail_cnt = fail_cnt + 1;
        end

        if (fail_cnt == 0) $display("tb_score_acc_rt: ALL PASSED");
        else               $display("tb_score_acc_rt: FAILED (%0d errors)", fail_cnt);
        $finish;
    end
endmodule
