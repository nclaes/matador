"""Device BRAM-budget table for the vanilla_gp_tiled backend.

tile_mem is the only capacity-scaling resource that matters here — its size
in bits is 2 x features_padded x clauses_padded (see config.py), independent
of class count. Sizing capacity against a real device's BRAM budget (instead
of arbitrary per-field ceilings) is what actually determines whether a given
(feat_slice, clause_slice, max_features, max_clauses_total) combination is
synthesizable.

Figures below are total on-chip Block RAM (not counting UltraRAM, LUTRAM, or
any other resource), sourced from public datasheets. Deliberately not an
exhaustive device list — target_fpga can be any string; if it's not here,
GPTiledAcceleratorConfig requires an explicit bram_bits_budget override.
"""

from __future__ import annotations

# device key -> total Block RAM bits
FPGA_BRAM_BITS: dict[str, int] = {
    # Zynq-7020 (XC7Z020): 140x 36Kb BRAM blocks
    "xc7z020": 4_900_000,
    # Kintex UltraScale KU040 (XCKU040): ~38 Mbit BRAM (no URAM on this part)
    "xcku040": 38_000_000,
}

# Fraction of a device's total BRAM budget reserved for tile_mem; the rest is
# left for the input FIFO and any other on-chip logic sharing the same BRAM
# pool. Not currently user-configurable — tune here if it needs to move.
TILE_MEM_BRAM_FRACTION = 0.75


def bram_budget_bits(target_fpga: str, bram_bits_budget: int | None) -> int:
    """Resolve the BRAM-bit budget available for tile_mem.

    Args:
        target_fpga: device key into FPGA_BRAM_BITS, or any string when
            bram_bits_budget is given explicitly.
        bram_bits_budget: explicit override of the device's total BRAM bits.
            Required when target_fpga isn't in FPGA_BRAM_BITS.

    Returns:
        Bits available for tile_mem (device total x TILE_MEM_BRAM_FRACTION).
    """
    if bram_bits_budget is not None:
        total_bits = bram_bits_budget
    elif target_fpga in FPGA_BRAM_BITS:
        total_bits = FPGA_BRAM_BITS[target_fpga]
    else:
        known = ", ".join(sorted(FPGA_BRAM_BITS))
        raise ValueError(
            f"Unknown target_fpga {target_fpga!r} and no bram_bits_budget override given. "
            f"Known devices: {known}. Either use one of these or set bram_bits_budget "
            f"to the target device's total Block RAM bits directly."
        )
    return int(total_bits * TILE_MEM_BRAM_FRACTION)
