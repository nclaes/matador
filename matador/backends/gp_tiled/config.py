"""GP-tiled backend configuration.

Unlike TiledAcceleratorConfig/HardwiredAcceleratorConfig, this config
separates three distinct sizing concepts:

  * per-cycle SIMD width (feat_slice, clause_slice) — "how many rounds of
    computation are required" for one inference: n_feat_slices =
    ceil(max_features/feat_slice), n_clause_slices =
    ceil(max_clauses_total/clause_slice). Mirrors vanilla_tiled's existing
    feat_slice/clause_slice knobs (matador/config/schema.py).
  * compile-time CAPACITY (max_features, max_clauses_total, max_classes) —
    sizes the synthesized silicon in real units. Chosen once; the same
    bitstream can then be reprogrammed at runtime with any model that fits
    inside it, without resynthesis.
  * the target FPGA's BRAM budget (target_fpga / bram_bits_budget) — the
    real constraint on how big max_features x max_clauses_total can be.
    tile_mem's bit cost is 2 x features_padded x clauses_padded, essentially
    independent of max_classes. See matador.backends.gp_tiled.fpga_budget.

The runtime MODEL geometry (n_classes, clauses_per_class, ... — read from
the TMIR at model_path) must fit within the configured capacity, checked by
matador.backends.gp_tiled.tmir_bridge.check_capacity().
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matador.backends.gp_tiled.fpga_budget import bram_budget_bits
from matador.backends.gp_tiled.tmir_bridge import (
    HARD_MAX_CLASSES,
    HARD_MAX_CLAUSE_SLICES,
    HARD_MAX_CLAUSES_TOTAL,
    HARD_MAX_FEAT_SLICES,
    HARD_MAX_TILES,
)


class GPTiledAcceleratorConfig(BaseModel):
    """Configuration for the runtime-reprogrammable GP-tiled accelerator
    (vendored from GP_TM_Inference_Accelerator's tm_accel_gp.v).

    One synthesis of this core, sized by target_fpga/max_features/
    max_clauses_total/max_classes below, can be reprogrammed with any
    vanilla/per_class TM model that fits within that capacity via an
    AXI-Stream CMD_LOAD packet — no resynthesis needed to swap models,
    unlike vanilla_tiled/vanilla_hardwired.
    """
    model_config = ConfigDict(extra="ignore")

    model_path: Path = Field(description="Path to a TMIR .yaml or .npz file.")
    output_dir: Path = Field(
        description="Root output directory; RTL is written to <output_dir>/RTL/."
    )
    axis_data_width: Literal[32] = Field(
        default=32,
        description=(
            "AXI-Stream TDATA width in bits. Only 32 is currently supported: "
            "the vendored golden model and packet encoder (tm_emulator.py) "
            "hardcode AXIS_W=32 for the LOAD/INFER packet layout."
        ),
    )
    fifo_depth: int = Field(
        default=16,
        ge=4,
        description="Input FIFO depth in beats (must be a power of 2).",
    )

    # ── target device (drives the BRAM-fit check below) ─────────────────────
    target_fpga: str = Field(
        description=(
            "Target FPGA device key (e.g. 'xc7z020', 'xcku040' — see "
            "matador.backends.gp_tiled.fpga_budget.FPGA_BRAM_BITS). Any "
            "string is accepted; if it isn't a known device, bram_bits_budget "
            "must be set explicitly."
        ),
    )
    bram_bits_budget: Optional[int] = Field(
        default=None,
        description=(
            "Explicit override of the target device's total Block RAM bits. "
            "Required when target_fpga isn't in fpga_budget.FPGA_BRAM_BITS."
        ),
    )

    # ── per-cycle SIMD width ("how many rounds of computation") ────────────
    feat_slice: int = Field(
        default=32,
        ge=1,
        description="Features evaluated per cycle. Smaller = more rounds (higher latency, less logic).",
    )
    clause_slice: int = Field(
        default=32,
        ge=1,
        description="Clauses evaluated per cycle. Smaller = more rounds (higher latency, less logic).",
    )

    # ── compile-time capacity (real units; silicon sizing) ─────────────────
    max_features: int = Field(
        ge=1,
        description="Compile-time max n_features any loaded model may use.",
    )
    max_clauses_total: int = Field(
        ge=2,
        description="Compile-time max n_clauses_total any loaded model may use.",
    )
    max_classes: int = Field(
        default=32,
        ge=1,
        le=HARD_MAX_CLASSES,
        description=(
            f"Compile-time max n_classes any loaded model may use. Hard ceiling "
            f"{HARD_MAX_CLASSES} (word1's n_classes header field is 8 bits). "
            "Negligible BRAM cost — this doesn't factor into the device-fit check."
        ),
    )

    @model_validator(mode="after")
    def _check_capacity_and_budget(self) -> "GPTiledAcceleratorConfig":
        max_feat_slices = math.ceil(self.max_features / self.feat_slice)
        max_clause_slices = math.ceil(self.max_clauses_total / self.clause_slice)
        n_tiles_max = max_feat_slices * max_clause_slices

        problems = []
        if max_feat_slices > HARD_MAX_FEAT_SLICES:
            problems.append(
                f"max_features={self.max_features} / feat_slice={self.feat_slice} needs "
                f"{max_feat_slices} feature slices, exceeding the protocol's hard limit of "
                f"{HARD_MAX_FEAT_SLICES} (word2's n_feat_slices header field is 8 bits)"
            )
        if max_clause_slices > HARD_MAX_CLAUSE_SLICES:
            problems.append(
                f"max_clauses_total={self.max_clauses_total} / clause_slice={self.clause_slice} "
                f"needs {max_clause_slices} clause slices, exceeding the protocol's hard limit "
                f"of {HARD_MAX_CLAUSE_SLICES} (word2's n_clause_slices header field is 8 bits)"
            )
        if self.max_clauses_total > HARD_MAX_CLAUSES_TOTAL:
            problems.append(
                f"max_clauses_total={self.max_clauses_total} exceeds the protocol's hard limit "
                f"of {HARD_MAX_CLAUSES_TOTAL} (word3's n_clauses_total header field is 16 bits)"
            )
        if n_tiles_max > HARD_MAX_TILES:
            problems.append(
                f"max_feat_slices({max_feat_slices}) x max_clause_slices({max_clause_slices}) "
                f"= {n_tiles_max} tiles, exceeding the protocol's hard limit of {HARD_MAX_TILES} "
                f"(word3's n_tiles header field is 16 bits)"
            )
        if problems:
            raise ValueError(
                "GPTiledAcceleratorConfig exceeds the AXI-Stream protocol's wire-format "
                "ceilings:\n  " + "\n  ".join(problems)
            )

        required_bits = 2 * (max_feat_slices * self.feat_slice) * (max_clause_slices * self.clause_slice)
        budget_bits = bram_budget_bits(self.target_fpga, self.bram_bits_budget)
        if required_bits > budget_bits:
            raise ValueError(
                f"tile_mem for this config needs {required_bits:,} bits "
                f"(2 x {max_feat_slices * self.feat_slice:,} features_padded x "
                f"{max_clause_slices * self.clause_slice:,} clauses_padded), which exceeds the "
                f"{budget_bits:,}-bit budget for target_fpga={self.target_fpga!r} "
                f"(device BRAM x fpga_budget.TILE_MEM_BRAM_FRACTION). Reduce max_features/"
                f"max_clauses_total, choose a bigger device, or raise bram_bits_budget."
            )

        return self

    @model_validator(mode="after")
    def _model_path_must_exist(self) -> "GPTiledAcceleratorConfig":
        v = self.model_path
        if not v.exists():
            raise ValueError(f"model_path does not exist: {v}")
        if v.suffix.lower() not in {".yaml", ".yml", ".npz"}:
            raise ValueError("model_path must be .yaml, .yml, or .npz")
        return self

    # ── derived capacity (not stored fields — always computed from the
    #    real-unit fields above, so they can never drift out of sync) ──────
    @property
    def max_feat_slices(self) -> int:
        return math.ceil(self.max_features / self.feat_slice)

    @property
    def max_clause_slices(self) -> int:
        return math.ceil(self.max_clauses_total / self.clause_slice)

    @property
    def n_tiles_max(self) -> int:
        return self.max_feat_slices * self.max_clause_slices

    @property
    def required_bram_bits(self) -> int:
        return 2 * (self.max_feat_slices * self.feat_slice) * (self.max_clause_slices * self.clause_slice)
