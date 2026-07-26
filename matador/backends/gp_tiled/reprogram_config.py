"""Config for `matador reprogram-suite` — an ordered list of {model,
dataset} steps to reprogram an already-generated bundle with, in one
continuous run. Lives alongside GPTiledAcceleratorConfig since only
vanilla_gp_tiled (RTLBackend.supports_reprogramming) can use it today, but
the shape itself has no gp_tiled-specific fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ReprogramStepConfig(BaseModel):
    model: Path = Field(description="Path to a TMIR .yaml/.yml/.npz file.")
    dataset: Optional[Path] = Field(
        default=None,
        description="Booleanized *_test.txt-shaped file (space-separated, last column=label); label is dropped.",
    )
    vectors: Optional[Path] = Field(
        default=None, description="Raw bit-vector file (one vector per line, no label column)."
    )
    n_samples: Optional[int] = Field(default=None, gt=0, description="Subsample this many rows from `dataset`.")
    seed: int = Field(default=0, description="Subsampling seed.")
    name: Optional[str] = Field(default=None, description="Display name; defaults to the model file's stem.")

    @field_validator("model", mode="after")
    @classmethod
    def _model_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"model does not exist: {v}")
        if v.suffix.lower() not in {".yaml", ".yml", ".npz"}:
            raise ValueError(f"model must be a .yaml, .yml, or .npz file, got: {v.suffix!r}")
        return v

    @field_validator("dataset", mode="after")
    @classmethod
    def _dataset_must_exist(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"dataset does not exist: {v}")
        return v

    @field_validator("vectors", mode="after")
    @classmethod
    def _vectors_must_exist(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            raise ValueError(f"vectors does not exist: {v}")
        return v


class ReprogramSuiteConfig(BaseModel):
    """Just the step list — which backend/bundle to target is given the
    same way `matador simulate` already does it (--backend + --config
    pointing at that backend's own accelerator config), so this file can't
    drift out of sync with a separately-recorded output_dir/backend pair.
    """
    steps: list[ReprogramStepConfig] = Field(description="Ordered reprogramming steps.")

    @field_validator("steps", mode="after")
    @classmethod
    def _steps_nonempty(cls, v: list[ReprogramStepConfig]) -> list[ReprogramStepConfig]:
        if not v:
            raise ValueError("steps must be non-empty")
        return v
