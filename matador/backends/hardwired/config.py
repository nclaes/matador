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
    model_config = ConfigDict(extra="ignore")

    model_path: Path = Field(description="Path to a TMIR .yaml or .npz file.")
    output_dir: Path = Field(
        description="Root output directory; RTL is written to <output_dir>/RTL/."
    )
    axis_data_width: Literal[32, 64] = Field(
        default=32,
        description="AXI-Stream TDATA width in bits.",
    )
    fifo_depth: int = Field(
        default=16,
        ge=4,
        description="Input FIFO depth in beats (must be a power of 2).",
    )
    pipeline_stages: int = Field(
        default=3,
        ge=0,
        description=(
            "Register stages inserted in the clause-score adder tree. "
            "0 = fully combinational (minimum latency; may not meet timing at high clock). "
            "ceil(log2(n_clauses_per_class / 2)) = fully pipelined (best Fmax). "
            "In scoring_mode='templated', only 0 vs >0 is distinguished — see "
            "hw_score.v's header comment for why."
        ),
    )
    scoring_mode: Literal["full_hardwired", "templated"] = Field(
        default="full_hardwired",
        description=(
            "'full_hardwired' (default): hw_score.v is bespoke Python-generated "
            "Verilog per model — a hand-built adder tree (with pipeline_stages "
            "register levels placed inside it) plus a hand-unrolled linear-scan "
            "argmax comparator chain. "
            "'templated': hw_score.v instead instantiates two small, static, "
            "reusable modules per class — adder_tree.v (a generic parameterized "
            "popcount tree, shared with no other backend needs it) and "
            "argmax_rt.v (the same combinational argmax vanilla_gp_tiled and "
            "redress already use, from matador/backends/common/src/) — trading "
            "full_hardwired's per-level pipeline tuning for one shared, "
            "already-tested implementation reused across backends."
        ),
    )
    n_test_vectors: int = Field(
        default=8,
        ge=1,
        description=(
            "How many of the source TMIR's verification.test_vectors to embed "
            "into BOTH generated testbenches -- tb_hw_system.v (iverilog/xsim) "
            "and tb_top.cpp (the Verilator harness) -- taken from the front of "
            "that list, one shared cap so the two harnesses' embedded vector "
            "counts never drift apart. Each embedded vector costs real "
            "generated-file bulk (tb_hw_system.v: its own register array "
            "declaration, an init assignment per AXI beat, a case-statement "
            "arm; tb_top.cpp: one array-literal entry per beat) and adds to "
            "the simulation's runtime -- neither harness has file-based "
            "($readmemh) or plusarg-driven vector loading; both are "
            "deliberately simple, self-contained. The real "
            "verification.test_vectors pool size varies enormously across "
            "models (observed: 170 for statlog up to 20,064 for sports), so "
            "this field intentionally defaults to a small, fast-to-simulate "
            "number rather than 'use everything available' -- do not set it "
            "anywhere near a model's full pool size for tb_hw_system.v's sake "
            "(Verilator can comfortably compile a much larger embedded array "
            "than the per-vector Verilog idiom can stay fast to elaborate, "
            "but both still share this one field, by design, for consistency "
            "over independent tuning). "
            "tb_hw_system.v's own N_VEC is additionally a plain Verilog "
            "parameter, so a user can run FEWER of the embedded set at "
            "simulate time without regenerating anything (e.g. "
            "`iverilog -Ptb_hw_system.N_VEC=3 ...`) -- there is no equivalent "
            "runtime knob for tb_top.cpp; running fewer there means "
            "regenerating with a smaller n_test_vectors. Neither generated "
            "harness is intended for a large-scale accuracy sweep, only a "
            "functional smoke test -- raise n_test_vectors deliberately, and "
            "expect both harnesses' generated-file size and elaboration time "
            "to grow with it."
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
