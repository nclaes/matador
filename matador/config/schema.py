from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

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
    model_name: Optional[str] = Field(
        default=None,
        description=(
            "Namespaces this model's output under <output_dir>/TMIR/<model_name>/, so "
            "multiple models can coexist in one work directory without overwriting each "
            "other's validation_config.yaml/rom_inference.py/etc. If not given, derived "
            "from train_data's filename (e.g. digits_train.txt -> 'digits') — matching "
            "matador booleanize's own <name>_train.txt convention."
        ),
    )

    @field_validator("clauses")
    @classmethod
    def clauses_must_be_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError("clauses must be an even number")
        return v

    @field_validator("model_name", mode="after")
    @classmethod
    def _model_name_must_be_safe(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and (not v.strip() or "/" in v or "\\" in v):
            raise ValueError("model_name must be a non-empty string with no path separators")
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


class FeatureEncoderSpec(BaseModel):
    """One column (or column range) -> Boolean-bit-block mapping. `column`
    is required inside BooleanisationConfig.features (an int index, or an
    inclusive "lo-hi" range string applying the same encoder independently
    to each covered column); it's ignored on default_encoder, which is a
    template applied to every column not otherwise listed."""
    column: Optional[Union[int, str]] = None
    encoder: Literal["thermometer", "threshold", "onehot", "passthrough"]
    bits: Optional[int] = Field(default=None, gt=0, description="thermometer only")
    range: Optional[tuple[float, float]] = Field(default=None, description="thermometer only")
    bins: Optional[list[float]] = Field(default=None, description="thermometer only: explicit thresholds")
    quantile: bool = Field(default=False, description="thermometer only: fit thresholds from train quantiles")
    threshold: Optional[float] = Field(default=None, description="threshold encoder only")
    categories: Optional[list[Any]] = Field(default=None, description="onehot only: explicit category list")


class BooleanizationRecipe(BaseModel):
    """The pure encoding-scheme half of booleanization — which raw columns
    get which encoder — with no I/O fields. Reused standalone as a
    catalog entry's optional default recipe (matador.preprocessing.sources.
    RawDataSourceSpec.booleanization) and as the base of BooleanisationConfig
    below, which adds where the data actually comes from / goes."""
    features: list[FeatureEncoderSpec] = Field(default_factory=list)
    default_encoder: Optional[FeatureEncoderSpec] = Field(
        default=None, description="Fallback applied to any raw column not covered by `features`."
    )


class BooleanisationConfig(BooleanizationRecipe):
    """Turns raw arrays (matador ingest's npz output, or any x/y npz) into
    the exact Boolean train/test text format TrainingConfig.train_data /
    test_data already consumes — no changes needed to matador train."""
    raw_npz: Path = Field(description="npz with x_train/y_train/x_test/y_test, or unsplit x/y.")
    name: str = Field(description="Output file stem: <name>_train.txt / <name>_test.txt")
    output_dir: Path = Field(description="Directory for output files.")
    test_size: float = Field(default=0.2, gt=0, lt=1, description="Only used when raw_npz has unsplit x/y.")
    seed: int = Field(default=0)
    stratify: bool = Field(default=True)

    @field_validator("raw_npz", mode="after")
    @classmethod
    def _raw_npz_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"raw_npz does not exist: {v}")
        return v


class TMAcceleratorConfig(BaseModel):
    """Configuration for RTL generation from a vanilla TMIR model."""
    model_config = ConfigDict(extra="ignore")

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


# Canonical name used by the tiled backend module.
# TMAcceleratorConfig remains the primary definition here; this alias
# will be removed once all call sites migrate to the backend-namespaced import.
TiledAcceleratorConfig = TMAcceleratorConfig
