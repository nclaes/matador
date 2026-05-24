from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TMType(str, Enum):
    vanilla = "vanilla"
    coalesced = "coalesced"


class TrainingConfig(BaseModel):
    tm_type: TMType = Field(description="Tsetlin Machine variant: 'vanilla' or 'coalesced'")
    clauses: int = Field(gt=0, description="Number of clauses (must be even)")
    classes: int = Field(gt=0, description="Number of output classes")
    features: int = Field(gt=0, description="Number of Boolean input features")
    s: float = Field(gt=1.0, description="Specificity parameter")
    T: int = Field(gt=0, description="Voting threshold")
    epochs: int = Field(gt=0, description="Number of training epochs")
    max_included_literals: int = Field(gt=0, description="Max literals per clause")
    seed: int = Field(default=42, description="Random seed")
    train_data: Path = Field(description="Path to training data file")
    test_data: Path = Field(description="Path to test data file")
    output_dir: Path = Field(description="Directory for output files")

    @field_validator("clauses")
    @classmethod
    def clauses_must_be_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError("clauses must be an even number")
        return v

    @model_validator(mode="after")
    def max_literals_within_range(self) -> "TrainingConfig":
        limit = self.features * 2
        if self.max_included_literals > limit:
            raise ValueError(
                f"max_included_literals ({self.max_included_literals}) "
                f"exceeds features*2 ({limit})"
            )
        return self


class ValidationConfig(BaseModel):
    model_path: Path = Field(
        description="Path to a TMIR model file (.yaml or .npz)."
    )
    test_data: Optional[Path] = Field(
        default=None,
        description=(
            "Path to a space-separated Boolean test dataset (last column = label). "
            "When provided, inference accuracy is reported."
        ),
    )

    @field_validator("model_path", mode="after")
    @classmethod
    def model_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"model_path does not exist: {v}")
        return v

    @field_validator("model_path", mode="after")
    @classmethod
    def model_path_must_be_tmir(cls, v: Path) -> Path:
        if v.suffix not in {".yaml", ".yml", ".npz"}:
            raise ValueError(
                f"model_path must be a .yaml, .yml, or .npz file, got: {v.suffix!r}"
            )
        return v

    @field_validator("test_data", mode="after")
    @classmethod
    def test_data_must_exist(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"test_data does not exist: {v}")
        return v


class TMAcceleratorConfig(BaseModel):
    """Configuration for RTL generation from a vanilla TMIR model."""
    model_config = ConfigDict(extra="forbid")

    model_path: Path = Field(description="Path to a TMIR .yaml or .npz file.")
    output_dir: Path = Field(description="Root output directory; RTL is written to <output_dir>/RTL/.")
    axis_data_width: Literal[32, 64] = Field(
        default=32,
        description="AXI-Stream TDATA width in bits.",
    )
    fifo_depth: int = Field(
        default=16,
        ge=4,
        description="Input FIFO depth in beats (must be a power of 2).",
    )
    feat_slice: int = Field(
        default=4,
        ge=1,
        description="Feature slice width: literals per tile column (powers of 2 recommended).",
    )
    clause_slice: int = Field(
        default=4,
        ge=1,
        description="Clause slice width: clauses per tile row (powers of 2 recommended).",
    )

    @field_validator("model_path", mode="after")
    @classmethod
    def _model_path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"model_path does not exist: {v}")
        if v.suffix.lower() not in {".yaml", ".yml", ".npz"}:
            raise ValueError("model_path must be .yaml, .yml, or .npz")
        return v

    @field_validator("fifo_depth", mode="after")
    @classmethod
    def _fifo_depth_power_of_two(cls, v: int) -> int:
        if v & (v - 1) != 0:
            raise ValueError(f"fifo_depth must be a power of 2, got {v}")
        return v
