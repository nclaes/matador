`timescale 1ns/1ps
// =============================================================================
// adder_tree — generic parameterized popcount / balanced binary adder tree
// =============================================================================
// Sums N_INPUTS independent 1-bit signals (e.g. one class's positive- or
// negative-polarity clause fired/not-fired bits) into a single OUT_WIDTH-bit
// unsigned count, via a balanced binary reduction tree.
//
// Built by RECURSIVE module instantiation (splits its input bus in half,
// instantiates itself on each half, bottoms out at a single bit) rather than
// per-model Python-generated tree text — this is the "templated" half of
// vanilla_hardwired's scoring_mode="templated" option (see hw_score.v's own
// header comment): one tested, reusable module instead of a bespoke tree
// hand-built fresh for every model's exact class/clause count.
//
// Purely combinational. hw_score.v (templated mode) optionally registers
// its OWN per-class score (the signed subtraction downstream of two
// adder_tree instances), not this module internally — see its header for
// why that's the simpler, sufficient pipelining point rather than
// registering mid-tree the way "full_hardwired" mode's hand-built tree does.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module adder_tree #(
    parameter N_INPUTS  = 8,
    parameter OUT_WIDTH = 4     // must be >= clog2(N_INPUTS+1); caller sizes this
)(
    input  wire [N_INPUTS-1:0]  in,
    output wire [OUT_WIDTH-1:0] sum
);
    generate
        if (N_INPUTS <= 1) begin : g_leaf
            assign sum = (N_INPUTS == 0) ? {OUT_WIDTH{1'b0}}
                                          : {{(OUT_WIDTH-1){1'b0}}, in[0]};
        end else begin : g_split
            localparam N_LO = N_INPUTS / 2;
            localparam N_HI = N_INPUTS - N_LO;

            wire [OUT_WIDTH-1:0] sum_lo, sum_hi;

            adder_tree #(.N_INPUTS(N_LO), .OUT_WIDTH(OUT_WIDTH)) u_lo (
                .in(in[N_LO-1:0]), .sum(sum_lo)
            );
            adder_tree #(.N_INPUTS(N_HI), .OUT_WIDTH(OUT_WIDTH)) u_hi (
                .in(in[N_INPUTS-1:N_LO]), .sum(sum_hi)
            );

            assign sum = sum_lo + sum_hi;
        end
    endgenerate
endmodule
