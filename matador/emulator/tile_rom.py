"""TileROMEmulator — tile ROM reader mirroring the RTL tile layout.

Builds the same flat ROM as _tile_rom_init() in rtl/accelerator.py.
Tile address = feat_slice_idx * N_CLAUSE_SLICES + clause_slice_idx.

Per-tile layout (CLAUSE_SLICE × 2 × FEAT_SLICE bits total):
  bits [k*2*FS +: FS]      = TA Include actions for positive literals, clause k
  bits [k*2*FS + FS +: FS] = TA Include actions for negated  literals, clause k
"""

from __future__ import annotations

import math

from matador.emulator.trace import ROMAccessEvent, TAActionVectorEvent


class TileROMEmulator:
    def __init__(self, ta_actions, n_features: int, n_clauses_total: int,
                 n_literals: int, feat_slice: int, clause_slice: int, cycle_ref: list):
        """
        Args:
            ta_actions: bool array, shape (n_clauses_total, n_literals).
                        ta_actions[g, l] = True → clause g issues Include for literal l.
            n_features:     Number of input features.
            n_clauses_total: Total clause count.
            n_literals:     2 * n_features.
            feat_slice:     Features per tile column.
            clause_slice:   Clauses per tile row.
            cycle_ref:      Shared mutable [cycle] reference.
        """
        self._ta_actions = ta_actions
        self.n_features = n_features
        self.n_clauses_total = n_clauses_total
        self.n_literals = n_literals
        self.feat_slice = feat_slice
        self.clause_slice = clause_slice
        self._cycle_ref = cycle_ref

        self.n_feat_slices = math.ceil(n_features / feat_slice)
        self.n_clause_slices = math.ceil(n_clauses_total / clause_slice)
        self.tile_width = clause_slice * 2 * feat_slice

        self._rom = self._build_rom()

    def _build_rom(self) -> list:
        FS = self.feat_slice
        CS = self.clause_slice
        NCS = self.n_clause_slices

        rom = []
        for fi in range(self.n_feat_slices):
            for ci in range(NCS):
                tile_val = 0
                for k in range(CS):
                    g = ci * CS + k
                    if g >= self.n_clauses_total:
                        continue
                    for l_local in range(FS):
                        feat_global = fi * FS + l_local
                        if feat_global >= self.n_features:
                            continue
                        if self._ta_actions[g, feat_global]:
                            tile_val |= 1 << (k * 2 * FS + l_local)
                        if self._ta_actions[g, self.n_features + feat_global]:
                            tile_val |= 1 << (k * 2 * FS + FS + l_local)
                rom.append(tile_val)
        return rom

    def read(self, feat_slice_idx: int, clause_slice_idx: int) -> tuple:
        """Read one tile and return (tile_data, ROMAccessEvent)."""
        addr = feat_slice_idx * self.n_clause_slices + clause_slice_idx
        tile_data = self._rom[addr]
        ev = ROMAccessEvent(
            cycle=self._cycle_ref[0],
            tile_addr=addr,
            feat_slice_idx=feat_slice_idx,
            clause_slice_idx=clause_slice_idx,
            ta_action_matrix=tile_data,
        )
        return tile_data, ev

    def extract_ta_actions_k(self, tile_data: int, k: int) -> tuple:
        """Extract TA action bits for clause k within a tile.

        Returns:
            (ta_actions_pos, ta_actions_neg, ta_action_mask)
            ta_actions_pos: FEAT_SLICE bits — Include actions for positive literals
            ta_actions_neg: FEAT_SLICE bits — Include actions for negated literals
            ta_action_mask: 2*FEAT_SLICE bits — combined mask for clause_eval
        """
        FS = self.feat_slice
        mask = (1 << FS) - 1
        ta_actions_pos = (tile_data >> (k * 2 * FS)) & mask
        ta_actions_neg = (tile_data >> (k * 2 * FS + FS)) & mask
        # ta_action_mask layout matches partial_lits: {neg_bits, pos_bits}
        ta_action_mask = ta_actions_pos | (ta_actions_neg << FS)
        return ta_actions_pos, ta_actions_neg, ta_action_mask
