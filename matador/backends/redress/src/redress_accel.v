`timescale 1ns/1ps
// =============================================================================
// redress_accel — REDRESS/MUSTANG accelerator: runtime-reprogrammable,
//                 compressed-instruction-stream Tsetlin Machine inference
// =============================================================================
// PURPOSE
//   A hardware translation of the REDRESS 16-bit Include-Encoding scheme
//   (Chapter 4.3, Algorithms 11/12 of the source thesis) -- MUSTANG's
//   Single-Core configuration (Chapter 5.3, Figure 5.13). Unlike
//   vanilla_hardwired (one model wired into gates per synthesis) or
//   vanilla_gp_tiled (a dense per-clause tile ROM streamed in), the model
//   here is a SPARSE instruction stream: one 16-bit word per included
//   literal, nothing stored for excluded literals or empty clauses. Two
//   BRAMs hold the whole model -- Instruction Memory (the compressed
//   stream) and the Feature FIFO (this batch's input data) -- and decoding
//   one instruction per cycle drives W parallel vote accumulators, one per
//   datapoint in the batch.
//
// ── AXI-STREAM PROTOCOL ──────────────────────────────────────────────────────
//   Every s_axis packet begins with a Header beat (see Figure 5.13.2-5.13.3):
//
//     word0  [31]    NEW_STREAM (must be 1 on every header beat)
//            [30]    STREAM_TYPE: 1 = Instruction stream, 0 = Feature stream
//     Instruction header (STREAM_TYPE=1):
//            [29:22] n_classes      1..MAX_CLASSES
//            [21: 0] n_instructions 1..MAX_INCLUDES (compressed model size)
//     Feature header (STREAM_TYPE=0):
//            [29: 0] n_feat_words   1..MAX_FEATURES (one beat per feature --
//                                   this design only supports BATCH_WIDTH
//                                   <= AXIS_DATA_WIDTH, i.e. one feature's
//                                   whole packed batch word fits one beat;
//                                   see config.py)
//
//   Instruction packet: n_instructions beats follow the header, each
//   carrying one MUSTANG instruction word in bits [15:0] (bits [31:16]
//   reserved, ignored on read). TLAST on the final beat. Loads Instruction
//   Memory sequentially from address 0; does NOT trigger inference by
//   itself (a model can be loaded once, then inferenced against many
//   feature batches).
//
//   Feature packet: n_feat_words beats follow the header, each one
//   feature's BATCH_WIDTH-bit packed word (see codec.pack_features()) in
//   bits [BATCH_WIDTH-1:0]. TLAST on the final beat. Loads the Feature
//   FIFO sequentially from address 0, THEN AUTOMATICALLY RUNS INFERENCE
//   against the currently-loaded instruction stream (no separate "go"
//   command -- matches Figure 5.13's dataflow, where feature loading feeds
//   directly into Literal Select).
//
//   m_axis response: one beat per datapoint in the batch, bits
//   [CLASS_WIDTH-1:0] = predicted class (bits above that are 0), TLAST on
//   the final (index BATCH_WIDTH-1) beat. Hold-until-accept handshake
//   (exit gated on tvalid & tready, same convention as tm_accel_gp.v).
//
// ── INSTRUCTION WORD (bits [15:0] of each instruction beat) ─────────────────
//   Algorithm 11's word extended with MUSTANG's Class-Change bit (see
//   codec.py's module docstring for the full derivation, including a real
//   inconsistency found and resolved in the thesis's own Figure 4.8):
//     bit15  P/-   clause polarity (0 = positive, 1 = negative)
//     bit14  CC    clause-change marker (flips on every new non-empty clause)
//     bit13  E     class-change marker (flips on every new non-empty class)
//     bits1-12     offset = feature index into the Feature FIFO (12 bits)
//     bit0         L     literal polarity (0 = positive literal, 1 = negated)
//
// ── DECODE/ACCUMULATE DATAFLOW (Figure 5.13, steps 4-6) ─────────────────────
//   One instruction is processed every 3 cycles in this design (fetch the
//   instruction word from Instruction Memory, fetch the feature word from
//   the Feature FIFO using the just-decoded offset, then AND-accumulate) --
//   a deliberately simple, non-pipelined first implementation; a pipelined
//   version overlapping instruction/feature fetch could raise throughput to
//   1 instruction/cycle in a later iteration without changing behaviour.
//
//   clause_mask (BATCH_WIDTH bits, one bit per datapoint lane) accumulates
//   AND across a clause's literals. On a CC change (a new clause has
//   started) or an E change (a new class has started, which is also
//   necessarily a new clause), the just-closed clause is "flushed": each
//   score_acc_rt instance (one per lane, W total) receives one valid pulse
//   with active=clause_mask[lane], cls=the class that clause belonged to,
//   polarity=that clause's own P bit.
//
//   Scores are NOT read out incrementally as each class completes (unlike
//   the software codec's running best_score/best_class). Instead every
//   score_acc_rt bank keeps all MAX_CLASSES scores for its lane
//   simultaneously, and W parallel argmax_rt instances (combinational)
//   resolve the winner per lane once, after the final instruction --
//   simpler control logic than an incremental running-best register, at
//   the cost of MAX_CLASSES x BATCH_WIDTH x SCORE_WIDTH bits of score
//   storage instead of BATCH_WIDTH x SCORE_WIDTH.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module redress_accel #(
    parameter AXIS_DATA_WIDTH = 32,
    parameter MAX_CLASSES     = 16,
    parameter CLASS_WIDTH     = 4,   // $clog2(MAX_CLASSES)
    parameter MAX_INCLUDES    = 4096,
    parameter INSTR_ADDR_W    = 12,  // $clog2(MAX_INCLUDES)
    parameter MAX_FEATURES    = 256,
    parameter FEAT_ADDR_W     = 8,   // $clog2(MAX_FEATURES)
    parameter BATCH_WIDTH     = 32,
    parameter OUT_IDX_W       = 6,   // $clog2(BATCH_WIDTH)
    parameter SCORE_WIDTH     = 14
)(
    input  wire                        clk,
    input  wire                        rst_n,

    input  wire [AXIS_DATA_WIDTH-1:0]  s_axis_tdata,
    input  wire                        s_axis_tvalid,
    output wire                        s_axis_tready,
    input  wire                        s_axis_tlast,

    output reg  [AXIS_DATA_WIDTH-1:0]  m_axis_tdata,
    output reg                         m_axis_tvalid,
    input  wire                        m_axis_tready,
    output reg                         m_axis_tlast,

    output wire                        busy
);

    localparam OFFSET_W = 12; // fixed by the MUSTANG instruction word layout

    // ── FSM states ───────────────────────────────────────────────────────
    // Every BRAM read (redress_bram) is itself registered -- re/raddr must
    // be STABLE for a full cycle before rdata is valid the cycle after.
    // Each *_INSTR/*_FEAT state below issues re/raddr from the PREVIOUS
    // state's tail (so they're already stable on entry) and only reads the
    // result in the state after that -- four states per instruction, not
    // three; an earlier version of this file skipped the feature-side wait
    // state and read feat_rdata one cycle before redress_bram actually
    // produced it, off by one instruction the whole way through.
    localparam S_IDLE          = 4'd0;
    localparam S_LOAD_INSTR    = 4'd1;
    localparam S_LOAD_FEAT     = 4'd2;
    localparam S_INFER_INSTR   = 4'd3;  // instr_re/raddr stable this cycle; wait for instr_rdata
    localparam S_INFER_DECODE  = 4'd4;  // instr_rdata valid; decode; issue feat_re/raddr
    localparam S_INFER_FEAT    = 4'd5;  // feat_re/raddr stable this cycle; wait for feat_rdata
    localparam S_INFER_ACCUM   = 4'd6;  // feat_rdata valid; flush-check + AND-accumulate
    localparam S_INFER_FINAL   = 4'd7;  // unconditional flush of the last open clause
    localparam S_OUTPUT        = 4'd8;

    reg [3:0] state;

    // busy: combinational, 1 whenever the FSM is doing anything but
    // waiting for a new packet header -- gives an external observer (a
    // fault-injection harness, a waveform viewer) a black-box way to find
    // "somewhere mid-load-or-inference" without decoding `state` itself,
    // the same role gp_tiled's own `busy` output port plays.
    assign busy = (state != S_IDLE);

    // ── header-latched configuration ────────────────────────────────────
    reg                       is_instr_stream;
    reg  [7:0]                hdr_n_classes;
    reg  [INSTR_ADDR_W-1:0]   hdr_n_instr;      // n_instructions - 1 (last valid ptr)
    reg  [FEAT_ADDR_W-1:0]    hdr_n_feat_words; // n_feat_words - 1

    // ── load-path counters ──────────────────────────────────────────────
    reg  [INSTR_ADDR_W-1:0]   load_instr_addr;
    reg  [FEAT_ADDR_W-1:0]    load_feat_addr;

    // ── Instruction Memory ───────────────────────────────────────────────
    wire                      instr_we    = (state == S_LOAD_INSTR) && s_axis_tvalid && s_axis_tready;
    wire [INSTR_ADDR_W-1:0]   instr_waddr = load_instr_addr;
    wire [15:0]               instr_wdata = s_axis_tdata[15:0];
    reg                       instr_re;
    reg  [INSTR_ADDR_W-1:0]   instr_raddr;
    wire [15:0]               instr_rdata;

    redress_bram #(
        .WIDTH(16), .DEPTH(MAX_INCLUDES), .ADDR_W(INSTR_ADDR_W)
    ) instr_mem (
        .clk(clk),
        .we(instr_we), .waddr(instr_waddr), .wdata(instr_wdata),
        .re(instr_re), .raddr(instr_raddr), .rdata(instr_rdata)
    );

    // ── Feature FIFO ─────────────────────────────────────────────────────
    wire                      feat_we    = (state == S_LOAD_FEAT) && s_axis_tvalid && s_axis_tready;
    wire [FEAT_ADDR_W-1:0]    feat_waddr = load_feat_addr;
    wire [BATCH_WIDTH-1:0]    feat_wdata = s_axis_tdata[BATCH_WIDTH-1:0];
    reg                       feat_re;
    reg  [FEAT_ADDR_W-1:0]    feat_raddr;
    wire [BATCH_WIDTH-1:0]    feat_rdata;

    redress_bram #(
        .WIDTH(BATCH_WIDTH), .DEPTH(MAX_FEATURES), .ADDR_W(FEAT_ADDR_W)
    ) feature_fifo (
        .clk(clk),
        .we(feat_we), .waddr(feat_waddr), .wdata(feat_wdata),
        .re(feat_re), .raddr(feat_raddr), .rdata(feat_rdata)
    );

    // ── inference-phase decode/accumulate state ─────────────────────────
    reg  [INSTR_ADDR_W-1:0]   ptr;              // last-issued instruction address
    reg  [15:0]               instr_latched;    // decoded word from S_INFER_DECODE
    reg  [BATCH_WIDTH-1:0]    clause_mask;
    reg                       have_clause;
    reg                       have_class;
    reg                       P_prev, CC_prev, E_prev;
    reg  [CLASS_WIDTH-1:0]    current_class;

    wire                      P_cur      = instr_latched[15];
    wire                      CC_cur     = instr_latched[14];
    wire                      E_cur      = instr_latched[13];
    wire [OFFSET_W-1:0]       offset_cur = instr_latched[12:1];
    wire                      L_cur      = instr_latched[0];

    wire [BATCH_WIDTH-1:0]    selected_literal = L_cur ? ~feat_rdata : feat_rdata;

    // Single combinational decision, computed once, used everywhere below --
    // avoids ever assigning clause_mask (or the flush pulse) from more than
    // one place for the same cycle.
    wire class_boundary  = have_class  && (E_cur  != E_prev);
    wire clause_boundary = have_clause && (CC_cur != CC_prev);
    wire flush_this_cycle = class_boundary || clause_boundary;

    // score_acc_rt control -- shared (broadcast) across all W lanes; each
    // lane supplies its own `active` bit from clause_mask.
    //
    // COMBINATIONAL, not registered: score_acc_rt samples `active` as
    // clause_mask[lane] on the SAME edge these fire. clause_mask is reset
    // to the new clause's first literal on that identical edge (see
    // S_INFER_ACCUM below) via a non-blocking assign that only becomes
    // visible the FOLLOWING cycle -- so a registered (one-cycle-delayed)
    // acc_valid/acc_cls/acc_polarity would fire a cycle too late, by which
    // point clause_mask has already moved on to the next clause's data.
    // Keeping these combinational keeps them synchronised with the
    // still-old clause_mask value score_acc_rt's own posedge block reads
    // at this exact edge.
    wire                      flush_pulse       = (state == S_INFER_ACCUM) && flush_this_cycle;
    wire                      final_flush_pulse = (state == S_INFER_FINAL);
    wire                      batch_start_pulse = (state == S_LOAD_FEAT) && s_axis_tvalid && s_axis_tready && s_axis_tlast;

    wire                      acc_valid    = flush_pulse || final_flush_pulse;
    wire                      acc_clear    = batch_start_pulse;
    wire                      acc_polarity = ~P_prev;   // 1 = positive clause (score_acc_rt's own convention)
    wire [CLASS_WIDTH-1:0]    acc_cls      = current_class;

    genvar gi;

    // ── W parallel vote accumulators (one bank of MAX_CLASSES scores per
    //    datapoint lane) + W parallel argmax units ───────────────────────
    wire [SCORE_WIDTH*MAX_CLASSES-1:0] scores_flat [0:BATCH_WIDTH-1];
    wire [CLASS_WIDTH-1:0]             pred_class  [0:BATCH_WIDTH-1];

    generate
        for (gi = 0; gi < BATCH_WIDTH; gi = gi + 1) begin : lanes
            score_acc_rt #(
                .N_CLASSES(MAX_CLASSES), .SCORE_WIDTH(SCORE_WIDTH)
            ) acc (
                .clk(clk), .rst_n(rst_n),
                .clear(acc_clear),
                .valid(acc_valid),
                .cls(acc_cls),
                .polarity(acc_polarity),
                .active(clause_mask[gi]),
                .scores_flat(scores_flat[gi])
            );
            argmax_rt #(
                .N_CLASSES(MAX_CLASSES), .SCORE_WIDTH(SCORE_WIDTH), .CLASS_WIDTH(CLASS_WIDTH)
            ) amax (
                .scores_flat(scores_flat[gi]),
                .n_classes({1'b0, hdr_n_classes[CLASS_WIDTH-1:0]}),
                .pred_class(pred_class[gi])
            );
        end
    endgenerate

    // ── output-phase counter ────────────────────────────────────────────
    reg  [OUT_IDX_W-1:0]      out_idx;

    assign s_axis_tready = (state == S_LOAD_INSTR) || (state == S_LOAD_FEAT) || (state == S_IDLE);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state           <= S_IDLE;
            m_axis_tvalid   <= 1'b0;
            m_axis_tlast    <= 1'b0;
            m_axis_tdata    <= {AXIS_DATA_WIDTH{1'b0}};
            load_instr_addr <= {INSTR_ADDR_W{1'b0}};
            load_feat_addr  <= {FEAT_ADDR_W{1'b0}};
            ptr             <= {INSTR_ADDR_W{1'b0}};
            have_clause     <= 1'b0;
            have_class      <= 1'b0;
            clause_mask     <= {BATCH_WIDTH{1'b1}};
            current_class   <= {CLASS_WIDTH{1'b0}};
            instr_re        <= 1'b0;
            feat_re         <= 1'b0;
        end else begin
            // defaults, overridden below where needed
            instr_re  <= 1'b0;
            feat_re   <= 1'b0;

            case (state)
                S_IDLE: begin
                    m_axis_tvalid <= 1'b0;
                    if (s_axis_tvalid && s_axis_tready) begin
                        is_instr_stream <= s_axis_tdata[30];
                        if (s_axis_tdata[30]) begin
                            hdr_n_classes   <= s_axis_tdata[29:22];
                            hdr_n_instr     <= s_axis_tdata[INSTR_ADDR_W-1:0];
                            load_instr_addr <= {INSTR_ADDR_W{1'b0}};
                            state           <= S_LOAD_INSTR;
                        end else begin
                            hdr_n_feat_words <= s_axis_tdata[FEAT_ADDR_W-1:0];
                            load_feat_addr   <= {FEAT_ADDR_W{1'b0}};
                            state            <= S_LOAD_FEAT;
                        end
                    end
                end

                S_LOAD_INSTR: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        load_instr_addr <= load_instr_addr + 1'b1;
                        if (s_axis_tlast)
                            state <= S_IDLE;
                    end
                end

                S_LOAD_FEAT: begin
                    if (s_axis_tvalid && s_axis_tready) begin
                        load_feat_addr <= load_feat_addr + 1'b1;
                        if (s_axis_tlast) begin
                            // Start a fresh batch: acc_clear (combinational,
                            // see its declaration above) fires this same
                            // cycle from batch_start_pulse -- clears all W
                            // score banks. Reset clause/class tracking,
                            // begin fetching instruction 0.
                            have_clause   <= 1'b0;
                            have_class    <= 1'b0;
                            clause_mask   <= {BATCH_WIDTH{1'b1}};
                            current_class <= {CLASS_WIDTH{1'b0}};
                            ptr           <= {INSTR_ADDR_W{1'b0}};
                            instr_re    <= 1'b1;
                            instr_raddr <= {INSTR_ADDR_W{1'b0}};
                            state       <= S_INFER_INSTR;
                        end
                    end
                end

                S_INFER_INSTR: begin
                    // instr_re/instr_raddr were issued the cycle before
                    // entering this state (either from S_LOAD_FEAT's TLAST
                    // cycle, or from S_INFER_ACCUM below); this state just
                    // waits out the 1-cycle BRAM read latency.
                    state <= S_INFER_DECODE;
                end

                S_INFER_DECODE: begin
                    instr_latched <= instr_rdata;
                    feat_re       <= 1'b1;
                    feat_raddr    <= instr_rdata[12:1]; // offset field
                    state         <= S_INFER_FEAT;
                end

                S_INFER_FEAT: begin
                    // feat_re/feat_raddr were issued last cycle (S_INFER_
                    // DECODE, above); this state just waits out redress_
                    // bram's own 1-cycle read latency, same idiom as
                    // S_INFER_INSTR for the instruction side.
                    state <= S_INFER_ACCUM;
                end

                S_INFER_ACCUM: begin
                    // Flush whatever clause was open, exactly once, before
                    // starting this instruction's own literal into a
                    // (possibly fresh) clause_mask -- class_boundary implies
                    // clause_boundary too (a new class's first instruction
                    // is necessarily its first clause's first instruction),
                    // so only ONE flush pulse ever fires per cycle.
                    // acc_valid/acc_cls/acc_polarity (combinational, see
                    // their declarations above) fire this same cycle from
                    // flush_pulse -- driven straight off flush_this_cycle/
                    // current_class/P_prev, all stable register reads at
                    // this edge, so score_acc_rt samples them alongside
                    // the still-pre-reset clause_mask below.
                    if (flush_this_cycle) begin
                        clause_mask  <= selected_literal;
                    end else begin
                        clause_mask <= clause_mask & selected_literal;
                    end

                    if (class_boundary)
                        current_class <= current_class + 1'b1;

                    if (!have_class || class_boundary) begin
                        E_prev     <= E_cur;
                        have_class <= 1'b1;
                    end
                    if (!have_clause || flush_this_cycle) begin
                        P_prev      <= P_cur;
                        CC_prev     <= CC_cur;
                        have_clause <= 1'b1;
                    end

                    if (ptr == hdr_n_instr) begin
                        state <= S_INFER_FINAL;
                    end else begin
                        ptr         <= ptr + 1'b1;
                        instr_re    <= 1'b1;
                        instr_raddr <= ptr + 1'b1;
                        state       <= S_INFER_INSTR;
                    end
                end

                S_INFER_FINAL: begin
                    // Unconditionally flush the last open clause (there is
                    // always exactly one open at this point -- every model
                    // has at least one non-empty clause in its last class,
                    // else mustang_encode() would have rejected it).
                    // acc_valid/acc_cls/acc_polarity (combinational) fire
                    // this same cycle from final_flush_pulse.
                    out_idx       <= {OUT_IDX_W{1'b0}};
                    m_axis_tvalid <= 1'b0; // scores from this flush land next cycle
                    state         <= S_OUTPUT;
                end

                S_OUTPUT: begin
                    // Wait one extra cycle after S_INFER_FINAL for the last
                    // score_acc_rt update (and the argmax_rt combinational
                    // result it feeds) to settle before driving m_axis.
                    if (!m_axis_tvalid) begin
                        m_axis_tvalid <= 1'b1;
                        m_axis_tdata  <= {{(AXIS_DATA_WIDTH-CLASS_WIDTH){1'b0}}, pred_class[0]};
                        m_axis_tlast  <= (BATCH_WIDTH == 1);
                    end else if (m_axis_tvalid && m_axis_tready) begin
                        if (m_axis_tlast) begin
                            m_axis_tvalid <= 1'b0;
                            state         <= S_IDLE;
                        end else begin
                            out_idx      <= out_idx + 1'b1;
                            m_axis_tdata <= {{(AXIS_DATA_WIDTH-CLASS_WIDTH){1'b0}}, pred_class[out_idx + 1'b1]};
                            m_axis_tlast <= (out_idx + 1'b1 == BATCH_WIDTH - 1);
                        end
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
