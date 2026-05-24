"""ClauseEvalEmulator — pure combinatorial model mirroring clause_eval.v."""

from __future__ import annotations


class ClauseEvalEmulator:
    """Stateless clause evaluator.

    Mirrors the RTL logic:
        no_actions = (ta_action_mask == 0)
        all_active = AND over all literals l: (literal[l] == 1 OR ta_action_mask[l] == 0)
        active     = all_active AND NOT no_actions
    """

    def eval(self, partial_lits: int, ta_action_mask: int, n_literals: int) -> tuple:
        """Evaluate one clause for one feature-slice tile.

        Args:
            partial_lits:   2*FEAT_SLICE-bit integer; bit l = literal l value.
            ta_action_mask: 2*FEAT_SLICE-bit integer; bit l = 1 → Include action.
            n_literals:     Total number of literal bits (2 * feat_slice).

        Returns:
            (fires, has_actions): fires=True iff clause passes this tile;
                                  has_actions=True iff any Include action in tile.
        """
        no_actions = (ta_action_mask & ((1 << n_literals) - 1)) == 0
        # all_active: for every bit where ta_action_mask=1, partial_lits must also be 1
        all_active = (partial_lits | ~ta_action_mask) & ((1 << n_literals) - 1) == (1 << n_literals) - 1
        fires = all_active and not no_actions
        return fires, not no_actions
