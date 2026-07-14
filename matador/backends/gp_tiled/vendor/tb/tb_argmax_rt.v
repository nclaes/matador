`timescale 1ns/1ps
// tb_argmax_rt — unit test: winner selection, tie-break, and — the point of
//                the _rt variant — masking of classes >= n_classes so phantom
//                zero scores of unused classes can never win
module tb_argmax_rt;
    parameter N_CLASSES   = 16;
    parameter SCORE_WIDTH = 6;
    parameter CLASS_WIDTH = 4;

    reg  [SCORE_WIDTH*N_CLASSES-1:0] scores_flat;
    reg  [CLASS_WIDTH:0]             n_classes;
    wire [CLASS_WIDTH-1:0]           pred_class;

    argmax_rt #(.N_CLASSES(N_CLASSES), .SCORE_WIDTH(SCORE_WIDTH),
                .CLASS_WIDTH(CLASS_WIDTH)) dut (
        .scores_flat(scores_flat), .n_classes(n_classes), .pred_class(pred_class)
    );

    integer fail_cnt, i;

    task set_score;
        input integer c;
        input integer v;
        begin scores_flat[c*SCORE_WIDTH +: SCORE_WIDTH] = v[SCORE_WIDTH-1:0]; end
    endtask

    task clear_all;
        begin scores_flat = {SCORE_WIDTH*N_CLASSES{1'b0}}; end
    endtask

    task check;
        input [CLASS_WIDTH-1:0] exp;
        input [63:0]            id;
        begin
            #1;
            if (pred_class !== exp) begin
                $display("FAIL test%0d: exp=%0d got=%0d", id, exp, pred_class);
                fail_cnt = fail_cnt + 1;
            end
        end
    endtask

    initial begin
        $dumpfile("tb_argmax_rt.vcd");
        $dumpvars(0, tb_argmax_rt);
        fail_cnt = 0;

        // ── plain winner, full width ─────────────────────────────────────
        clear_all; n_classes = 5'd16;
        set_score(3, 5); set_score(9, 7); set_score(12, -2);
        check(4'd9, 0);

        // ── tie goes to the lower index ──────────────────────────────────
        clear_all; n_classes = 5'd16;
        set_score(2, 6); set_score(11, 6);
        check(4'd2, 1);

        // ── all-negative field, full width: least negative wins ──────────
        clear_all; n_classes = 5'd16;
        for (i = 0; i < 16; i = i + 1) set_score(i, -7);
        set_score(5, -3);
        check(4'd5, 2);

        // ── THE MASKING CASE: 4-class model, all real classes negative,
        //    phantom classes 4..15 sit at 0 and must NOT win ──────────────
        clear_all; n_classes = 5'd4;
        set_score(0, -5); set_score(1, -2); set_score(2, -7); set_score(3, -4);
        check(4'd1, 3);

        // ── masked class holding a big score must be invisible ───────────
        clear_all; n_classes = 5'd3;
        set_score(0, 1); set_score(1, 4); set_score(2, 2);
        set_score(7, 31);                       // garbage beyond the model
        check(4'd1, 4);

        // ── n_classes = 1 degenerates to class 0 ─────────────────────────
        clear_all; n_classes = 5'd1;
        set_score(0, -8);
        set_score(1, 31);
        check(4'd0, 5);

        // ── boundary: exactly N_CLASSES classes active, winner at top ────
        clear_all; n_classes = 5'd16;
        set_score(15, 3);
        check(4'd15, 6);

        // ── boundary: winner at index n_classes-1 under masking ──────────
        clear_all; n_classes = 5'd10;
        set_score(9, 2); set_score(10, 30);
        check(4'd9, 7);

        if (fail_cnt == 0) $display("tb_argmax_rt: ALL PASSED");
        else               $display("tb_argmax_rt: FAILED (%0d errors)", fail_cnt);
        $finish;
    end
endmodule
