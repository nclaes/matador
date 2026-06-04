"""Hardwired backend configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class HardwiredAcceleratorConfig(BaseModel):
    """Configuration for the hardwired combinational TM accelerator.

    TA Include actions are unrolled directly into AND-gate logic — no tile ROM,
    no sequential FSM for clause evaluation.  Lower latency than tiled for
    small-to-medium models; LUT cost grows with clause count.
    """
    model_config = ConfigDict(extra="forbid")

    model_path: Path = Field(description="Path to a TMIR .yaml or .npz file.")
    output_dir: Path = Field(
        description="Root output directory; RTL is written to <output_dir>/RTL/."
    )
    axis_data_width: Literal[32, 64] = Field(
        default=32,
        description="AXI-Stream TDATA width in bits.",
    )
    pipeline_stages: int = Field(
        default=3,
        ge=0,
        description=(
            "Register stages inserted in the clause-score adder tree. "
            "0 = fully combinational (minimum latency; may not meet timing at high clock). "
            "ceil(log2(n_clauses_per_class / 2)) = fully pipelined (best Fmax)."
        ),
    )

    @field_validator("model_path", mode="after")
    @classmethod
    def _model_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"model_path does not exist: {v}")
        if v.suffix.lower() not in {".yaml", ".yml", ".npz"}:
            raise ValueError("model_path must be .yaml, .yml, or .npz")
        return v
