`timescale 1ns/1ps
// =============================================================================
// tm_accel_gp — general-purpose (reprogrammable) Tsetlin Machine accelerator
// =============================================================================
// PURPOSE
//   Generalisation of tm_accelerator.v: the generation-time tile ROM is
//   replaced by a runtime-writable tile memory (tile_mem.v, Xilinx
//   BRAM-inferable), and every AXI-Stream input packet now begins with a
//   header word so the same s_axis interface carries BOTH
//     * model loads  (new TA Include/Exclude state matrices + geometry), and
//     * inference    (feature vectors, single frames or batches).
//   A model is loaded once, then any number of inference packets follow.
//   Loading a new model at any packet boundary reprograms the accelerator.
//
// ── AXI-STREAM PROTOCOL ──────────────────────────────────────────────────────
//   A packet is a TLAST-delimited burst on s_axis. Word 0 of every packet:
//
//     word0  [31:24] MAGIC = 0xA5
//            [23:16] CMD   = 0x01 INFER | 0x02 LOAD
//            [15: 0] reserved (ignored)
//
//   CMD_LOAD packet (reprogram the model):
//     word1  [ 7: 0] n_classes          1..MAX_CLASSES
//            [15: 8] clauses_per_class  even, >=2
//            [23:16] threshold          1..255 (VESTIGIAL -- retained for
//                                        wire-protocol compatibility only;
//                                        scores are unclamped, see
//                                        score_acc_rt.v)
//     word2  [ 7: 0] n_beats            1..MAX_FEAT_SLICES   (feature words/frame)
//            [15: 8] n_feat_slices      1..MAX_FEAT_SLICES
//            [23:16] n_clause_slices    1..MAX_CLAUSE_SLICES
//     word3  [15: 0] n_clauses_total    1..MAX_CLAUSES_TOTAL
//            [31:16] n_tiles            1..N_TILES_MAX
//     payload: n_tiles rows x (TILE_WIDTH/32)=64 words each, rows in linear
//              address order (feat_slice-major, clause_slice-minor, exactly
//              the S_COMPUTE sweep order). Word w of a row carries row bits
//              [w*32 +: 32] (LSW first). TLAST on the FINAL payload word.
//     Row bit layout (unchanged from the original ROM): clause lane k of a
//     row occupies bits [k*64 +: 64]; low 32 = Include(positive literals of
//     the feature window), high 32 = Include(negated literals).
//     The host-side packer is tm_emulator.py: encode_load_packet().
//
//   CMD_INFER packet (classify):
//     payload: K frames of n_beats words each (K>=1: one packet is a batch),
//              features packed LSB-first per word, word b = features
//              [b*32 +: 32] — identical to the original tm_accelerator.
//              TLAST on the final word of the FINAL frame only.
//
//   m_axis responses (in packet order, one stream):
//     after LOAD :  1 beat  {MAGIC, status, info[15:0]}          TLAST=1
//     after INFER:  1 beat per frame {pad[27:0], class[3:0]}     TLAST
//                   forwarded batch-style: 1 only on the final frame's
//                   result (same convention as the original core)
//     on error   :  1 beat  {MAGIC, error_code, info[15:0]}      TLAST=1
//
//     status/error codes ([23:16]):
//       0x00 OK           info = n_tiles written
//       0xE0 ERR_MAGIC    info = offending magic byte
//       0xE1 ERR_CMD      info = offending command byte
//       0xE2 ERR_RANGE    geometry field out of range / zero / odd cpc
//       0xE3 ERR_UNDERRUN TLAST before the packet was complete (info: progress)
//       0xE4 ERR_OVERRUN  more payload than the header promised
//       0xE5 ERR_NOCFG    INFER received before any successful LOAD
//     Any framing/geometry error during a LOAD leaves the accelerator
//     UNCONFIGURED (cfg_valid=0) until a subsequent LOAD succeeds — a
//     half-written model is never inferenced. After an error ack the FSM
//     returns to S_IDLE; the stream stays usable (errors are recoverable).
//
// ── MEMORY MAP (tile_mem) ────────────────────────────────────────────────────
//   N_TILES_MAX rows x TILE_WIDTH bits, synchronous 1-cycle read (BRAM).
//   Live region for the current model: rows 0 .. n_tiles-1 in the linear
//   sweep order above. Rows beyond n_tiles hold stale data and are never
//   addressed. Within the last clause slice, lanes >= n_clauses_total and,
//   within the last feature slice, literal bits >= n_features must be
//   zero-filled BY THE HOST PACKER (tm_emulator.py guarantees this); the
//   empty-clause / vacuous-pass rules then make them harmless.
//
// ── MICROARCHITECTURE CHANGES vs tm_accelerator.v ───────────────────────────
//   1) Tile fetch is pipelined one stage (BRAM synchronous read): addresses
//      issue on cycle t, the 32 clause_eval instances see the row on t+1.
//      S_COMPUTE therefore takes n_tiles+1 cycles (fill) instead of n_tiles.
//      The tile address is a plain linear counter — no feat*ncs multiplier.
//   2) clause->class / polarity decode uses running counters (intra-class
//      index + class index) instead of generated case tables, so any
//      geometry within the compile-time maxima works without regeneration.
//   3) Output handshake is hold-until-accept: exit from S_DONE/S_LACK/S_EACK
//      is gated on (m_axis_tvalid & m_axis_tready) and tvalid is dropped on
//      the exit edge. (The original sampled tready before tvalid was raised
//      and re-asserted tvalid on the exit edge, which loses a beat if tready
//      pulses early and duplicates one if tready rises late — visible only
//      under back-pressure. tb_system_gp toggles tready to lock this in.)
//
// ── CAPACITY (compile-time) vs GEOMETRY (runtime) ────────────────────────────
//   Capacity parameters below size the silicon; a CMD_LOAD header may select
//   any geometry within them. Defaults hold the shipped 512-feature /
//   10-class / 200-clause model with headroom (16 classes, 256 clauses).
//
// LATENCY  last feature TLAST -> result valid: n_tiles+1 (compute) +
//          n_clauses_total (score) + 1 (done) cycles; 314 for the frozen
//          model geometry. A full model load is 4 + n_tiles*64 beats.
//
// VERILOG-2001. No SystemVerilog. Timing-free RTL.
// =============================================================================
module tm_accel_gp #(
    // ── capacity (silicon sizing) ────────────────────────────────────────
    parameter MAX_CLASSES       = 16,
    parameter MAX_CLAUSES_TOTAL = 256,
    parameter MAX_FEAT_SLICES   = 16,
    parameter MAX_CLAUSE_SLICES = 8,
    parameter FEAT_SLICE        = 32,
    parameter CLAUSE_SLICE      = 32,
    // ── derived capacity (do not override independently) ─────────────────
    parameter MAX_FEAT_PADDED   = 512,   // MAX_FEAT_SLICES * FEAT_SLICE
    parameter TILE_WIDTH        = 2048,  // CLAUSE_SLICE * 2 * FEAT_SLICE
    parameter N_TILES_MAX       = 128,   // MAX_FEAT_SLICES * MAX_CLAUSE_SLICES
    parameter TILE_AW           = 7,     // $clog2(N_TILES_MAX)
    parameter WORD_CNT_W        = 6,     // $clog2(TILE_WIDTH/AXIS_DATA_WIDTH)
    // ── interface ─────────────────────────────────────────────────────────
    parameter AXIS_DATA_WIDTH   = 32,
    parameter SCORE_WIDTH       = 6,
    parameter CLASS_WIDTH       = 4,     // $clog2(MAX_CLASSES)
    parameter FIFO_DEPTH        = 16
)(
    input  wire                       clk,
    input  wire                       rst_n,
    // ── AXI-Stream input (headers + payload) ─────────────────────────────
    input  wire                       s_axis_tvalid,
    output wire                       s_axis_tready,
    input  wire [AXIS_DATA_WIDTH-1:0] s_axis_tdata,
    input  wire                       s_axis_tlast,
    // ── AXI-Stream output (acks + predictions) ───────────────────────────
    output reg                        m_axis_tvalid,
    input  wire                       m_axis_tready,
    output reg  [AXIS_DATA_WIDTH-1:0] m_axis_tdata,
    output reg                        m_axis_tlast,
    // ── status ────────────────────────────────────────────────────────────
    output reg                        busy,
    output reg                        configured   // 1 = a model is loaded
);

    // ── protocol constants ────────────────────────────────────────────────
    localparam [7:0] MAGIC        = 8'hA5;
    localparam [7:0] CMD_INFER    = 8'h01;
    localparam [7:0] CMD_LOAD     = 8'h02;
    localparam [7:0] ST_OK        = 8'h00;
    localparam [7:0] ERR_MAGIC    = 8'hE0;
    localparam [7:0] ERR_CMD      = 8'hE1;
    localparam [7:0] ERR_RANGE    = 8'hE2;
    localparam [7:0] ERR_UNDERRUN = 8'hE3;
    localparam [7:0] ERR_OVERRUN  = 8'hE4;
    localparam [7:0] ERR_NOCFG    = 8'hE5;

    localparam WORDS_PER_TILE = TILE_WIDTH / AXIS_DATA_WIDTH;   // 64
    // NOTE: threshold (header word1 byte [23:16]) no longer clamps scores
    // (see score_acc_rt.v) -- it is latched and range-checked purely for
    // wire-protocol backward compatibility with existing host packers.
    // geom_ok below only requires it nonzero; any value 1..255 is valid.

    // ── FSM states ────────────────────────────────────────────────────────
    localparam [3:0] S_IDLE    = 4'd0;   // wait/decode header word 0
    localparam [3:0] S_CFG     = 4'd1;   // capture LOAD geometry words 1..3
    localparam [3:0] S_LOAD    = 4'd2;   // stream payload rows into tile_mem
    localparam [3:0] S_LACK    = 4'd3;   // emit load-OK ack beat
    localparam [3:0] S_RECV    = 4'd4;   // capture feature beats of one frame
    localparam [3:0] S_COMPUTE = 4'd5;   // pipelined tile sweep
    localparam [3:0] S_SCORE   = 4'd6;   // one clause vote per cycle
    localparam [3:0] S_DONE    = 4'd7;   // emit prediction beat
    localparam [3:0] S_DRAIN   = 4'd8;   // swallow a bad packet to TLAST
    localparam [3:0] S_EACK    = 4'd9;   // emit error ack beat

    reg [3:0] state;

    // ── input FIFO (unchanged from the original design) ───────────────────
    wire                       fifo_m_tvalid;
    wire [AXIS_DATA_WIDTH-1:0] fifo_m_tdata;
    wire                       fifo_m_tlast;
    wire                       fifo_m_tready;

    assign fifo_m_tready = (state == S_IDLE) | (state == S_CFG) |
                           (state == S_LOAD) | (state == S_RECV) |
                           (state == S_DRAIN);

    axis_fifo #(.DATA_WIDTH(AXIS_DATA_WIDTH), .DEPTH(FIFO_DEPTH)) u_fifo (
        .clk      (clk),           .rst_n    (rst_n),
        .s_tvalid (s_axis_tvalid), .s_tready (s_axis_tready),
        .s_tdata  (s_axis_tdata),  .s_tlast  (s_axis_tlast),
        .m_tvalid (fifo_m_tvalid), .m_tready (fifo_m_tready),
        .m_tdata  (fifo_m_tdata),  .m_tlast  (fifo_m_tlast)
    );

    // ── runtime configuration (latched from a validated LOAD header) ──────
    reg [CLASS_WIDTH:0]   cfg_n_classes;       // 1..MAX_CLASSES (CLASS_WIDTH+1 bits: must hold MAX_CLASSES itself)
    reg [7:0]             cfg_cpc;             // clauses per class (even)
    reg [7:0]             cfg_half;            // cfg_cpc / 2
    reg [7:0]             cfg_threshold;       // vestigial, see NOTE above -- width matches the 8-bit header field
    reg [7:0]             cfg_n_beats;         // 1..MAX_FEAT_SLICES
    reg [7:0]             cfg_n_feat_slices;   // 1..MAX_FEAT_SLICES
    reg [7:0]             cfg_n_clause_slices; // 1..MAX_CLAUSE_SLICES
    reg [15:0]            cfg_n_clauses_total; // 1..MAX_CLAUSES_TOTAL
    reg [15:0]            cfg_n_tiles;         // 1..N_TILES_MAX

    // header staging
    reg [31:0] hdr_w1, hdr_w2;
    reg [1:0]  cfg_wcnt;

    // ── header geometry validation (combinational, at word 3) ────────────
    wire [7:0]  v_ncls  = hdr_w1[7:0];
    wire [7:0]  v_cpc   = hdr_w1[15:8];
    wire [7:0]  v_thr   = hdr_w1[23:16];
    wire [7:0]  v_nbts  = hdr_w2[7:0];
    wire [7:0]  v_nfs   = hdr_w2[15:8];
    wire [7:0]  v_ncs   = hdr_w2[23:16];
    wire [15:0] v_nct   = fifo_m_tdata[15:0];    // live word 3
    wire [15:0] v_ntil  = fifo_m_tdata[31:16];

    wire geom_ok =
        (v_ncls >= 8'd1) && (v_ncls <= MAX_CLASSES)                 &&
        (v_cpc  >= 8'd2) && (v_cpc[0] == 1'b0)                      &&
        (v_thr  >= 8'd1)                                            &&
        (v_nbts >= 8'd1) && (v_nbts <= MAX_FEAT_SLICES)             &&
        (v_nfs  >= 8'd1) && (v_nfs  <= MAX_FEAT_SLICES)             &&
        (v_ncs  >= 8'd1) && (v_ncs  <= MAX_CLAUSE_SLICES)           &&
        (v_nct  >= 16'd1) && (v_nct  <= MAX_CLAUSES_TOTAL)          &&
        (v_ntil >= 16'd1) && (v_ntil <= N_TILES_MAX);

    // ── model load path: row staging + tile write pointer ─────────────────
    // Words 0..62 of a row are staged; word 63 is written concatenated so a
    // full TILE_WIDTH row commits to tile_mem in the same cycle it completes.
    reg [TILE_WIDTH-AXIS_DATA_WIDTH-1:0] row_stage;
    reg [WORD_CNT_W-1:0]                 row_word;    // 0..63
    reg [15:0]                           tile_wr;     // rows committed so far

    wire load_word    = (state == S_LOAD) && fifo_m_tvalid;
    wire row_complete = load_word &&
                        (row_word == WORDS_PER_TILE[WORD_CNT_W-1:0] - 1'b1);
    wire last_tile    = row_complete && (tile_wr == cfg_n_tiles - 16'd1);

    wire                  mem_we    = row_complete;
    wire [TILE_AW-1:0]    mem_waddr = tile_wr[TILE_AW-1:0];
    wire [TILE_WIDTH-1:0] mem_wdata = {fifo_m_tdata, row_stage};

    // ── inference: frame capture ──────────────────────────────────────────
    reg [MAX_FEAT_PADDED-1:0] feature_reg;
    reg [7:0]                 beat_cnt;
    reg                       frame_tlast;   // batch-end marker (as original)

    // ── inference: pipelined tile sweep (stage 0 = address issue) ─────────
    reg  [7:0]          feat_cnt;
    reg  [7:0]          clause_slice_cnt;
    reg  [TILE_AW-1:0]  rd_addr;
    reg                 issue_done;
    // stage 1 (row in flight out of the BRAM)
    reg                 p_valid;
    reg                 p_first_row;
    reg  [7:0]          p_feat_cnt;
    reg  [7:0]          p_cs;

    wire mem_re = (state == S_COMPUTE) && !issue_done;

    wire [TILE_WIDTH-1:0] tile_data_q;
    tile_mem #(
        .TILE_WIDTH(TILE_WIDTH), .N_TILES(N_TILES_MAX), .ADDR_W(TILE_AW)
    ) u_tile_mem (
        .clk(clk),
        .we(mem_we), .waddr(mem_waddr), .wdata(mem_wdata),
        .re(mem_re), .raddr(rd_addr),   .rdata(tile_data_q)
    );

    // feature window selected by the PIPELINED slice index, so literals and
    // row data belong to the same tile. Base = p_feat_cnt*FEAT_SLICE, held in
    // a 16-bit wire (wide enough for MAX_FEAT_PADDED up to 65535) for clean lint.
    wire [15:0] feat_pos_base = p_feat_cnt * FEAT_SLICE;
    wire [FEAT_SLICE-1:0]   feat_pos     =
        feature_reg[feat_pos_base +: FEAT_SLICE];
    wire [2*FEAT_SLICE-1:0] partial_lits = {~feat_pos, feat_pos};

    // ── parallel clause evaluators (unchanged datapath) ───────────────────
    wire [CLAUSE_SLICE-1:0] tile_pass;
    wire [CLAUSE_SLICE-1:0] tile_has_actions;

    genvar k;
    generate
        for (k = 0; k < CLAUSE_SLICE; k = k + 1) begin : tile_eval
            wire [2*FEAT_SLICE-1:0] ta_actions_k;
            assign ta_actions_k = tile_data_q[k * 2 * FEAT_SLICE +: 2 * FEAT_SLICE];

            clause_eval #(.N_LITERALS(2 * FEAT_SLICE)) u_ce (
                .literals      (partial_lits),
                .ta_action_mask(ta_actions_k),
                .active        (tile_pass[k])
            );

            assign tile_has_actions[k] = |ta_actions_k;
        end
    endgenerate

    // ── clause accumulators ───────────────────────────────────────────────
    reg clause_pass        [0:MAX_CLAUSES_TOTAL-1];
    reg clause_has_actions [0:MAX_CLAUSES_TOTAL-1];

    // ── score phase: runtime class / polarity decode (no dividers) ────────
    reg  [15:0] score_cnt;     // global clause index
    reg  [7:0] intra_cnt;      // index within the current class (0..cpc-1)
    reg  [CLASS_WIDTH-1:0] score_class_r;

    wire score_valid  = (state == S_SCORE);
    wire score_is_pos = (intra_cnt < cfg_half);
    wire [15:0] score_idx = score_cnt;          // < MAX_CLAUSES_TOTAL by construction
    wire score_active = clause_pass[score_idx] & clause_has_actions[score_idx];
    reg  score_clear;

    wire [SCORE_WIDTH*MAX_CLASSES-1:0] scores_flat;
    // cfg_threshold is fixed at 8 bits (the header field width); zero/sign
    // -extend it to SCORE_WIDTH here via a plain assignment (correct either
    // direction, unlike a port connection of mismatched width, which some
    // tools warn on) since SCORE_WIDTH is capacity-derived and may be <, ==,
    // or > 8. threshold itself is vestigial -- see score_acc_rt.v.
    wire [SCORE_WIDTH-1:0] cfg_threshold_ext = cfg_threshold;

    score_acc_rt #(
        .N_CLASSES(MAX_CLASSES), .SCORE_WIDTH(SCORE_WIDTH)
    ) u_score_acc (
        .clk(clk), .rst_n(rst_n), .clear(score_clear), .valid(score_valid),
        .cls(score_class_r), .polarity(score_is_pos), .active(score_active),
        .threshold(cfg_threshold_ext), .scores_flat(scores_flat)
    );

    wire [CLASS_WIDTH-1:0] argmax_out;
    argmax_rt #(
        .N_CLASSES(MAX_CLASSES), .SCORE_WIDTH(SCORE_WIDTH),
        .CLASS_WIDTH(CLASS_WIDTH)
    ) u_argmax (
        .scores_flat(scores_flat),
        .n_classes  (cfg_n_classes),
        .pred_class (argmax_out)
    );

    // ── error bookkeeping ─────────────────────────────────────────────────
    reg [7:0]  err_code;
    reg [15:0] err_info;

    // ── loop variables ────────────────────────────────────────────────────
    integer k_upd;
    integer clause_g;
    integer ri;

    // ── FSM ───────────────────────────────────────────────────────────────
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= S_IDLE;
            busy             <= 1'b0;
            configured       <= 1'b0;
            m_axis_tvalid    <= 1'b0;
            m_axis_tdata     <= {AXIS_DATA_WIDTH{1'b0}};
            m_axis_tlast     <= 1'b0;
            cfg_n_classes    <= 8'd0;
            cfg_cpc          <= 8'd0;
            cfg_half         <= 8'd0;
            cfg_threshold    <= 8'd0;
            cfg_n_beats      <= 8'd0;
            cfg_n_feat_slices   <= 8'd0;
            cfg_n_clause_slices <= 8'd0;
            cfg_n_clauses_total <= 16'd0;
            cfg_n_tiles      <= 16'd0;
            hdr_w1           <= 32'd0;
            hdr_w2           <= 32'd0;
            cfg_wcnt         <= 2'd0;
            row_stage        <= {(TILE_WIDTH-AXIS_DATA_WIDTH){1'b0}};
            row_word         <= {WORD_CNT_W{1'b0}};
            tile_wr          <= 16'd0;
            feature_reg      <= {MAX_FEAT_PADDED{1'b0}};
            beat_cnt         <= 8'd0;
            frame_tlast      <= 1'b0;
            feat_cnt         <= 8'd0;
            clause_slice_cnt <= 8'd0;
            rd_addr          <= {TILE_AW{1'b0}};
            issue_done       <= 1'b0;
            p_valid          <= 1'b0;
            p_first_row      <= 1'b0;
            p_feat_cnt       <= 8'd0;
            p_cs             <= 8'd0;
            score_cnt        <= 16'd0;
            intra_cnt        <= 8'd0;
            score_class_r    <= {CLASS_WIDTH{1'b0}};
            score_clear      <= 1'b0;
            err_code         <= 8'd0;
            err_info         <= 16'd0;
            for (ri = 0; ri < MAX_CLAUSES_TOTAL; ri = ri + 1) begin
                clause_pass[ri]        = 1'b0;
                clause_has_actions[ri] = 1'b0;
            end
        end else begin
            score_clear <= 1'b0;   // single-cycle pulse

            case (state)
                // ─── S_IDLE: decode header word 0 ──────────────────────────
                S_IDLE: begin
                    busy     <= 1'b0;
                    beat_cnt <= 8'd0;
                    if (fifo_m_tvalid) begin
                        busy <= 1'b1;
                        if (fifo_m_tdata[31:24] != MAGIC) begin
                            err_code <= ERR_MAGIC;
                            err_info <= {8'd0, fifo_m_tdata[31:24]};
                            state    <= fifo_m_tlast ? S_EACK : S_DRAIN;
                        end else if (fifo_m_tdata[23:16] == CMD_LOAD) begin
                            if (fifo_m_tlast) begin
                                err_code <= ERR_UNDERRUN;
                                err_info <= 16'd0;
                                state    <= S_EACK;
                            end else begin
                                configured <= 1'b0;   // invalidate: a load has begun
                                cfg_wcnt   <= 2'd0;
                                state      <= S_CFG;
                            end
                        end else if (fifo_m_tdata[23:16] == CMD_INFER) begin
                            if (!configured) begin
                                err_code <= ERR_NOCFG;
                                err_info <= 16'd0;
                                state    <= fifo_m_tlast ? S_EACK : S_DRAIN;
                            end else if (fifo_m_tlast) begin
                                err_code <= ERR_UNDERRUN;   // header with no features
                                err_info <= 16'd0;
                                state    <= S_EACK;
                            end else begin
                                beat_cnt <= 8'd0;
                                state    <= S_RECV;
                            end
                        end else begin
                            err_code <= ERR_CMD;
                            err_info <= {8'd0, fifo_m_tdata[23:16]};
                            state    <= fifo_m_tlast ? S_EACK : S_DRAIN;
                        end
                    end
                end

                // ─── S_CFG: capture geometry words 1..3 and validate ───────
                S_CFG: begin
                    if (fifo_m_tvalid) begin
                        case (cfg_wcnt)
                            2'd0: hdr_w1 <= fifo_m_tdata;
                            2'd1: hdr_w2 <= fifo_m_tdata;
                            default: ;
                        endcase
                        if (fifo_m_tlast) begin
                            // header truncated, or complete header with no payload
                            err_code <= ERR_UNDERRUN;
                            err_info <= {14'd0, cfg_wcnt} + 16'd1;
                            state    <= S_EACK;
                        end else if (cfg_wcnt == 2'd2) begin
                            if (geom_ok) begin
                                cfg_n_classes       <= v_ncls;
                                cfg_cpc             <= v_cpc;
                                cfg_half            <= {1'b0, v_cpc[7:1]};
                                cfg_threshold       <= v_thr;
                                cfg_n_beats         <= v_nbts;
                                cfg_n_feat_slices   <= v_nfs;
                                cfg_n_clause_slices <= v_ncs;
                                cfg_n_clauses_total <= v_nct;
                                cfg_n_tiles         <= v_ntil;
                                row_word            <= {WORD_CNT_W{1'b0}};
                                tile_wr             <= 16'd0;
                                state               <= S_LOAD;
                            end else begin
                                err_code <= ERR_RANGE;
                                err_info <= 16'd0;
                                state    <= S_DRAIN;
                            end
                        end else begin
                            cfg_wcnt <= cfg_wcnt + 2'd1;
                        end
                    end
                end

                // ─── S_LOAD: assemble rows, write tile_mem ─────────────────
                // mem_we / mem_waddr / mem_wdata are combinational (see above)
                // so a row commits on the same edge its 64th word arrives.
                S_LOAD: begin
                    if (fifo_m_tvalid) begin
                        if (row_complete) begin
                            row_word <= {WORD_CNT_W{1'b0}};
                            tile_wr  <= tile_wr + 16'd1;
                        end else begin
                            row_stage[row_word * AXIS_DATA_WIDTH +: AXIS_DATA_WIDTH]
                                <= fifo_m_tdata;
                            row_word <= row_word + 1'b1;
                        end
                        if (fifo_m_tlast) begin
                            if (last_tile) begin
                                configured <= 1'b1;    // model complete & framed
                                state      <= S_LACK;
                            end else begin
                                err_code <= ERR_UNDERRUN;
                                err_info <= tile_wr;
                                state    <= S_EACK;
                            end
                        end else if (last_tile) begin
                            err_code <= ERR_OVERRUN;   // data done, framing wrong
                            err_info <= cfg_n_tiles;
                            state    <= S_DRAIN;
                        end
                    end
                end

                // ─── S_LACK: load-OK acknowledgement beat ──────────────────
                S_LACK: begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {MAGIC, ST_OK, cfg_n_tiles};
                    m_axis_tlast  <= 1'b1;
                    if (m_axis_tvalid && m_axis_tready) begin
                        m_axis_tvalid <= 1'b0;
                        m_axis_tlast  <= 1'b0;
                        busy          <= 1'b0;
                        state         <= S_IDLE;
                    end
                end

                // ─── S_RECV: capture one frame of feature beats ────────────
                S_RECV: begin
                    if (fifo_m_tvalid) begin
                        feature_reg[beat_cnt * AXIS_DATA_WIDTH +: AXIS_DATA_WIDTH]
                            <= fifo_m_tdata;
                        if (beat_cnt == cfg_n_beats - 8'd1) begin
                            // frame boundary: TLAST here means batch end,
                            // its absence means more frames follow (batch)
                            frame_tlast      <= fifo_m_tlast;
                            beat_cnt         <= 8'd0;
                            feat_cnt         <= 8'd0;
                            clause_slice_cnt <= 8'd0;
                            rd_addr          <= {TILE_AW{1'b0}};
                            issue_done       <= 1'b0;
                            p_valid          <= 1'b0;
                            state            <= S_COMPUTE;
                        end else if (fifo_m_tlast) begin
                            err_code <= ERR_UNDERRUN;   // frame cut short
                            err_info <= {8'd0, beat_cnt};
                            state    <= S_EACK;
                        end else begin
                            beat_cnt <= beat_cnt + 8'd1;
                        end
                    end
                end

                // ─── S_COMPUTE: pipelined tile sweep ───────────────────────
                // stage 0 issues the linear row address; stage 1 (p_*) tags
                // the row coming out of the BRAM. Accumulation is guarded by
                // p_valid so the fill cycle is harmless. Exit fires on the
                // edge that retires the final in-flight row.
                S_COMPUTE: begin
                    // stage-1 accumulate
                    if (p_valid) begin
                        for (k_upd = 0; k_upd < CLAUSE_SLICE; k_upd = k_upd + 1) begin
                            clause_g = p_cs * CLAUSE_SLICE + k_upd;
                            if (clause_g < MAX_CLAUSES_TOTAL) begin
                                if (p_first_row) begin
                                    clause_pass[clause_g]        <= tile_pass[k_upd] | ~tile_has_actions[k_upd];
                                    clause_has_actions[clause_g] <= tile_has_actions[k_upd];
                                end else begin
                                    clause_pass[clause_g] <=
                                        clause_pass[clause_g] & (tile_pass[k_upd] | ~tile_has_actions[k_upd]);
                                    clause_has_actions[clause_g] <=
                                        clause_has_actions[clause_g] | tile_has_actions[k_upd];
                                end
                            end
                        end
                    end
                    // stage-0 issue + stage-1 tag advance
                    if (!issue_done) begin
                        p_valid     <= 1'b1;
                        p_first_row <= (feat_cnt == 8'd0);
                        p_feat_cnt  <= feat_cnt;
                        p_cs        <= clause_slice_cnt;
                        rd_addr     <= rd_addr + {{(TILE_AW-1){1'b0}}, 1'b1};
                        if (clause_slice_cnt == cfg_n_clause_slices - 8'd1) begin
                            clause_slice_cnt <= 8'd0;
                            if (feat_cnt == cfg_n_feat_slices - 8'd1)
                                issue_done <= 1'b1;
                            else
                                feat_cnt <= feat_cnt + 8'd1;
                        end else begin
                            clause_slice_cnt <= clause_slice_cnt + 8'd1;
                        end
                    end else begin
                        p_valid <= 1'b0;
                    end
                    // exit as the last row retires
                    if (issue_done && p_valid) begin
                        score_cnt     <= 16'd0;
                        intra_cnt     <= 8'd0;
                        score_class_r <= {CLASS_WIDTH{1'b0}};
                        state         <= S_SCORE;
                    end
                end

                // ─── S_SCORE: one clause vote per cycle ────────────────────
                // class / polarity come from running counters, so any
                // (n_classes, clauses_per_class) within capacity works.
                S_SCORE: begin
                    score_cnt <= score_cnt + 16'd1;
                    if (intra_cnt == cfg_cpc - 8'd1) begin
                        intra_cnt     <= 8'd0;
                        score_class_r <= score_class_r + {{(CLASS_WIDTH-1){1'b0}}, 1'b1};
                    end else begin
                        intra_cnt <= intra_cnt + 8'd1;
                    end
                    if (score_cnt == cfg_n_clauses_total - 16'd1)
                        state <= S_DONE;
                end

                // ─── S_DONE: emit the prediction beat ──────────────────────
                // Hold-until-accept handshake; on accept, pulse score_clear
                // and either continue the batch (S_RECV) or await the next
                // packet header (S_IDLE), exactly mirroring the original's
                // batch semantics.
                S_DONE: begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {{(AXIS_DATA_WIDTH-CLASS_WIDTH){1'b0}}, argmax_out};
                    m_axis_tlast  <= frame_tlast;
                    if (m_axis_tvalid && m_axis_tready) begin
                        m_axis_tvalid <= 1'b0;
                        m_axis_tlast  <= 1'b0;
                        score_clear   <= 1'b1;
                        if (frame_tlast) begin
                            busy  <= 1'b0;
                            state <= S_IDLE;
                        end else begin
                            beat_cnt <= 8'd0;
                            state    <= S_RECV;    // next frame of the batch
                        end
                    end
                end

                // ─── S_DRAIN: swallow the rest of a bad packet ─────────────
                S_DRAIN: begin
                    if (fifo_m_tvalid && fifo_m_tlast)
                        state <= S_EACK;
                end

                // ─── S_EACK: error acknowledgement beat ────────────────────
                S_EACK: begin
                    m_axis_tvalid <= 1'b1;
                    m_axis_tdata  <= {MAGIC, err_code, err_info};
                    m_axis_tlast  <= 1'b1;
                    if (m_axis_tvalid && m_axis_tready) begin
                        m_axis_tvalid <= 1'b0;
                        m_axis_tlast  <= 1'b0;
                        busy          <= 1'b0;
                        state         <= S_IDLE;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
