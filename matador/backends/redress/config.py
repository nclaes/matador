"""REDRESS backend configuration.

Capacity here is shaped very differently from vanilla_gp_tiled's. gp_tiled's
tile_mem cost is a DENSE product (2 x features_padded x clauses_padded) that
depends only on geometry, not on how sparse the trained model actually is —
every clause gets a slot whether it has one Include or a hundred. REDRESS's
whole premise is the opposite: only Includes are stored (codec.py's
Algorithm 11), so the compressed size of a given model is DATA-dependent —
a function of how many literals it actually includes, not of n_features x
n_clauses_total. Two models with identical geometry can compress to wildly
different sizes.

Consequently, the compile-time capacity ceiling here is expressed directly
in the unit that actually matters: max_includes, the Instruction Memory's
depth in 16-bit words. Whether a SPECIFIC loaded TMIR's own encoded size
(codec.encode() run on it) fits inside max_includes is checked at generate()
time in rtl.py, once the model is actually loaded and encoded — this module
only validates that the CONFIG itself (its capacity numbers against the
target device's real BRAM budget) is internally consistent, the same
division of responsibility vanilla_gp_tiled uses between this file and
tmir_bridge.check_capacity().

The other capacity number, max_features, bounds two different things at
once: the Feature FIFO's depth (one batch_width-bit word per feature) and
the width of the offset field the RTL's actual on-wire format allocates.
That format is MUSTANG's 16-bit instruction (see codec.py's "MUSTANG"
section), not classic REDRESS's -- the 12-bit offset MUSTANG's Class-Change
bit leaves room for (after fixing a real inconsistency in the thesis's own
Figure 4.8, confirmed against its prose) is the true hard ceiling,
codec.MUSTANG_MAX_FEATURES (4096), not classic REDRESS's own 13-bit/8192
ceiling (codec.MAX_FEATURES) -- that one only bounds the software/MCU
reference path (encode()/decode_and_infer()), which this RTL backend does
not stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matador.backends.fpga_budget import bram_budget_bits
from matador.backends.redress.codec import MUSTANG_MAX_FEATURES

# Fraction of a device's total BRAM budget reserved for Instruction Memory +
# Feature FIFO; the rest is left for the Output FIFO and any other on-chip
# logic sharing the same BRAM pool. Mirrors gp_tiled's
# TILE_MEM_BRAM_FRACTION — tune here if it needs to move.
REDRESS_BRAM_FRACTION = 0.75


class RedressAcceleratorConfig(BaseModel):
    """Configuration for the REDRESS accelerator (MUSTANG's Single-Core
    configuration — see Chapter 5.3 of the source thesis): an
    AXI4-Stream-attached decoder for the 16-bit Include-Encoded instruction
    stream codec.py produces, batching batch_width datapoints per pass
    through the stream.

    One synthesis of this core, sized by target_fpga/max_includes/
    max_features/max_classes below, can be reprogrammed at runtime with any
    model whose encoded size and geometry fit within that capacity — no
    resynthesis needed, the same runtime-tunability story as
    vanilla_gp_tiled, achieved through a completely different (compressed
    instruction stream vs. dense tile memory) representation.
    """
    model_config = ConfigDict(extra="ignore")

    model_path: Path = Field(description="Path to a TMIR .yaml or .npz file.")
    output_dir: Path = Field(
        description="Root output directory; RTL is written to <output_dir>/RTL/."
    )
    axis_data_width: Literal[32] = Field(
        default=32,
        description="AXI-Stream TDATA width in bits for header/instruction/feature packets.",
    )
    fifo_depth: int = Field(
        default=16,
        ge=4,
        description="Input-side AXI-Stream buffering depth in beats (must be a power of 2).",
    )

    # ── batching width ("how many datapoints per pass through the stream") ─
    batch_width: Literal[16, 32, 64] = Field(
        default=32,
        description=(
            "Datapoints classified per pass through the encoded instruction "
            "stream (Algorithm 12's W). Also the Feature FIFO word width and "
            "the number of parallel AND-accumulate lanes. The thesis's three "
            "MUSTANG evaluation points; 32 is the default used throughout."
        ),
    )

    # ── target device (drives the BRAM-fit check below) ─────────────────────
    target_fpga: str = Field(
        description=(
            "Target FPGA device key (e.g. 'xc7z020', 'xcku040' — see "
            "matador.backends.fpga_budget.FPGA_BRAM_BITS). Any string is "
            "accepted; if it isn't a known device, bram_bits_budget must be "
            "set explicitly."
        ),
    )
    bram_bits_budget: Optional[int] = Field(
        default=None,
        description=(
            "Explicit override of the target device's total Block RAM bits. "
            "Required when target_fpga isn't in fpga_budget.FPGA_BRAM_BITS."
        ),
    )

    # ── compile-time capacity (real units; silicon sizing) ──────────────────
    max_includes: int = Field(
        ge=1,
        description=(
            "Compile-time Instruction Memory depth, in 16-bit words — i.e. "
            "the largest compressed model (sum of Include counts across all "
            "classes) any loaded model may need. This is a DATA-dependent "
            "number, not derivable from n_features/n_clauses_total alone — "
            "size it from the sparsest-case model(s) you actually intend to "
            "load, the same way you'd size any compressed-format buffer."
        ),
    )
    max_features: int = Field(
        ge=1,
        le=MUSTANG_MAX_FEATURES,
        description=(
            f"Compile-time max n_features any loaded model may use. Hard "
            f"ceiling {MUSTANG_MAX_FEATURES} (MUSTANG's 16-bit instruction "
            "word allocates a 12-bit feature offset)."
        ),
    )
    max_classes: int = Field(
        default=32,
        ge=1,
        description="Compile-time max n_classes any loaded model may use.",
    )

    @model_validator(mode="after")
    def _check_budget(self) -> "RedressAcceleratorConfig":
        instr_mem_bits = self.max_includes * 16
        feature_fifo_bits = self.max_features * self.batch_width
        required_bits = instr_mem_bits + feature_fifo_bits

        budget_bits = bram_budget_bits(self.target_fpga, self.bram_bits_budget, REDRESS_BRAM_FRACTION)
        if required_bits > budget_bits:
            raise ValueError(
                f"Instruction Memory + Feature FIFO for this config need "
                f"{required_bits:,} bits ({instr_mem_bits:,} for "
                f"max_includes={self.max_includes:,} x 16-bit words + "
                f"{feature_fifo_bits:,} for max_features={self.max_features:,} x "
                f"batch_width={self.batch_width}-bit words), which exceeds the "
                f"{budget_bits:,}-bit budget for target_fpga={self.target_fpga!r} "
                f"(device BRAM x REDRESS_BRAM_FRACTION). Reduce max_includes/"
                f"max_features/batch_width, choose a bigger device, or raise "
                f"bram_bits_budget."
            )
        return self

    @model_validator(mode="after")
    def _model_path_must_exist(self) -> "RedressAcceleratorConfig":
        v = self.model_path
        if not v.exists():
            raise ValueError(f"model_path does not exist: {v}")
        if v.suffix.lower() not in {".yaml", ".yml", ".npz"}:
            raise ValueError("model_path must be .yaml, .yml, or .npz")
        return self

    # ── derived capacity (not stored fields — always computed from the
    #    real-unit fields above, so they can never drift out of sync) ───────
    @property
    def instr_mem_bits(self) -> int:
        return self.max_includes * 16

    @property
    def feature_fifo_bits(self) -> int:
        return self.max_features * self.batch_width

    @property
    def required_bram_bits(self) -> int:
        return self.instr_mem_bits + self.feature_fifo_bits
