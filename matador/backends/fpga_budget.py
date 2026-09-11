"""Shared device BRAM-budget table, used by any backend that needs to check
whether its on-chip memory (tile_mem for vanilla_gp_tiled, Instruction
Memory + Feature FIFO for redress, ...) fits a real target device.

Figures below are total on-chip Block RAM (not counting UltraRAM, LUTRAM, or
any other resource), sourced from public datasheets. Deliberately not an
exhaustive device list — target_fpga can be any string; if it's not here,
callers require an explicit bram_bits_budget override.
"""

from __future__ import annotations

# device key -> total Block RAM bits
FPGA_BRAM_BITS: dict[str, int] = {
    # Zynq-7020 (XC7Z020): 140x 36Kb BRAM blocks
    "xc7z020": 4_900_000,
    # Kintex UltraScale KU040 (XCKU040): ~38 Mbit BRAM (no URAM on this part)
    "xcku040": 38_000_000,
}


def bram_budget_bits(target_fpga: str, bram_bits_budget: int | None, fraction: float) -> int:
    """Resolve the BRAM-bit budget available to a backend's primary on-chip
    memory resource.

    Args:
        target_fpga: device key into FPGA_BRAM_BITS, or any string when
            bram_bits_budget is given explicitly.
        bram_bits_budget: explicit override of the device's total BRAM bits.
            Required when target_fpga isn't in FPGA_BRAM_BITS.
        fraction: fraction of the device's total BRAM reserved for the
            caller's primary memory resource — the rest is left for FIFOs
            and any other on-chip logic sharing the same BRAM pool. Callers
            own this number (it's backend-specific, not a device property).

    Returns:
        Bits available (device total x fraction).
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
    return int(total_bits * fraction)
