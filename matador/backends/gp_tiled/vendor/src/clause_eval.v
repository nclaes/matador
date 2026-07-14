`timescale 1ns/1ps
// =============================================================================
// clause_eval — combinatorial single-clause evaluator
// =============================================================================
// PURPOSE
//   Implements one clause-firing test from the Tsetlin Machine inference rule.
//   Given the Include-action bits for one clause (ta_action_mask) and the
//   current literal values (literals), outputs whether the clause is active.
//
// TM CLAUSE RULE
//   A Tsetlin Automaton (TA) is trained per (clause, literal) pair.
//   When a TA's state exceeds n_states/2 it issues an Include action for its
//   literal; below that midpoint it issues Exclude. After training, the states
//   are thresholded to a single bit: Include = (state > n_states/2).
//   This module receives those Include bits as ta_action_mask.
//
//   Firing rule:
//     (1) The clause must have at least one Include  (ta_action_mask != 0)
//     (2) Every Included literal must currently be 1
//   An empty clause (all Exclude) is NEVER active regardless of literals.
//
// LITERAL ORDERING  — must match the ROM layout in tm_accelerator.v
//   literals[l]             feature[l]          (positive literal)
//   literals[N_LITERALS/2 + l]  NOT feature[l]  (negated  literal)
//   ta_action_mask[l]       Include bit for literal l (same layout)
//
// COMBINATORIAL — no registers; output settles within the clock cycle.
//   In tm_accelerator, CLAUSE_SLICE instances share one feature window.
//   Only ta_action_mask differs across instances (one ROM row per clause).
//   all_active and no_actions are exposed as named wires for waveform
//   inspection in tb_clause_eval.
//
// VERILOG-2001. No SystemVerilog. No timing constructs.
// =============================================================================
module clause_eval #(
    parameter N_LITERALS = 16
)(
    input  wire [N_LITERALS-1:0] literals,       // [F-1:0]=positive  [2F-1:F]=negated
    input  wire [N_LITERALS-1:0] ta_action_mask, // compiled TA Include-action bits
    output wire                  active
);
    wire no_actions = ~|ta_action_mask;
    wire all_active = &(literals | ~ta_action_mask);
    assign active = all_active & ~no_actions;
endmodule
