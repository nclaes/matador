"""Config schemas for the coal_tm CLI.

Replaces MATADOR_NO_GUI.json's untyped dict (whose checkconfig() was a
no-op — see legacy/MATADOR_NO_GUI.json and legacy/GUI.py's caller for the
old flow). These models fail fast, with a specific message, instead of
letting a malformed config run for minutes before crashing inside TMU or
producing silently-wrong RTL.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class TrainingConfig(BaseModel):
    """Everything needed to train a TMCoalescedClassifier and export the
    TAs.txt/weights.txt that coal_tm generate consumes. Coalesced only —
    Vanilla TM training is out of scope on this branch."""

    clauses: int = Field(gt=0)
    classes: int = Field(gt=0)
    features: int = Field(gt=0)
    s: float = Field(gt=0)
    T: int = Field(gt=0)
    epochs: int = Field(gt=0)
    max_included_literals: int = Field(gt=0)
    training_data: Path
    test_data: Path
    output_dir: Path
    seed: int = 42

    @model_validator(mode="after")
    def _check(self) -> "TrainingConfig":
        if not self.training_data.exists():
            raise ValueError(f"training_data does not exist: {self.training_data}")
        if not self.test_data.exists():
            raise ValueError(f"test_data does not exist: {self.test_data}")
        if self.training_data.resolve() == self.test_data.resolve():
            raise ValueError(
                "training_data and test_data point at the same file "
                f"({self.training_data}) — this trains and evaluates on "
                "identical data and will silently overstate accuracy."
            )
        if self.max_included_literals > 2 * self.features:
            raise ValueError(
                f"max_included_literals ({self.max_included_literals}) exceeds "
                f"2 * features ({2 * self.features}) — a clause can never "
                "include more literals than exist."
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


class RTLConfig(BaseModel):
    """Everything needed to generate Coalesced-TM RTL from an existing
    TAs.txt/weights.txt pair. Deliberately independent of TrainingConfig —
    supplying externally-produced TAs/weights and generating straight from
    them (no coal_tm train run at all) must keep working.

    bus_width must be < features (enforced below): a model whose whole
    feature vector fits in one packet gives the generated pipeline no
    natural gap to drain between back-to-back inferences, and the
    backpressure logic doesn't fully close that gap on its own. See
    README.md § "Known limitation: bus_width must be less than features"
    for the full explanation and how to pick a narrower bus_width instead.
    """

    output_dir: Path
    tas: Path
    weights: Path
    classes: int = Field(gt=0)
    clauses: int = Field(gt=0)
    features: int = Field(gt=0)
    bus_width: int = Field(gt=0, le=64)
    adder_stages: int = Field(default=1, gt=0)
    test_data: Path | None = None

    @model_validator(mode="after")
    def _check(self) -> "RTLConfig":
        if not self.tas.exists():
            raise ValueError(f"tas file does not exist: {self.tas}")
        if not self.weights.exists():
            raise ValueError(f"weights file does not exist: {self.weights}")
        if self.test_data is not None and not self.test_data.exists():
            raise ValueError(f"test_data does not exist: {self.test_data}")
        if self.clauses % self.adder_stages != 0:
            raise ValueError(
                f"clauses ({self.clauses}) must be evenly divisible by "
                f"adder_stages ({self.adder_stages}) — new_adder.sv's "
                "generate loop assumes CLAUSE_NUM/STAGE_NUM is exact; a "
                "remainder silently drops the leftover clauses from every "
                "class sum."
            )
        if self.features <= self.bus_width:
            raise ValueError(
                f"features ({self.features}) must be greater than bus_width "
                f"({self.bus_width}) — a model whose features fit in a single "
                "packet gives the generated core no natural gap between "
                "back-to-back inferences for the argmax pipeline (HCB_done -> "
                "adder_en -> adder_done -> argmax, ~4 cycles) to drain before "
                "the next one starts; this is a known, not-yet-fixed limit of "
                "the current backpressure logic, not just a slow path. Use a "
                "narrower bus_width (equivalently: more than one packet per "
                "vector) instead."
            )

        tas_expected = self.clauses * self.features * 2
        tas_actual = len(Path(self.tas).read_text().split())
        if tas_actual != tas_expected:
            raise ValueError(
                f"{self.tas} has {tas_actual} values, expected "
                f"clauses * features * 2 = {self.clauses} * {self.features} * 2 "
                f"= {tas_expected}"
            )

        weights_expected = self.classes * self.clauses
        weights_actual = len(Path(self.weights).read_text().split())
        if weights_actual != weights_expected:
            raise ValueError(
                f"{self.weights} has {weights_actual} values, expected "
                f"classes * clauses = {self.classes} * {self.clauses} "
                f"= {weights_expected}"
            )
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "RTLConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))
