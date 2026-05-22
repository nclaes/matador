from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


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
