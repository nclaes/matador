"""Abstract base classes for Matador accelerator backends.

Every backend must supply three paired components:

  AcceleratorSpec       — pure behavioral model (what the hardware must do)
  RTLBackend            — RTL generator (how to implement it in Verilog)
  CycleAccurateModel    — cycle-accurate Python model (enables testbench generation)

The invariant that the verification framework enforces:

    spec.simulate(x).predicted_class
  == cycle_model.run(x).predicted_class
  == RTL_simulation(x).output

  for all x in the test set.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data containers shared across backends
# ---------------------------------------------------------------------------

@dataclass
class RTLArtifacts:
    """Manifest of files produced by RTLBackend.generate()."""
    rtl_dir:    Path
    sources:    list[Path] = field(default_factory=list)
    testbenches: list[Path] = field(default_factory=list)
    sim_scripts: list[Path] = field(default_factory=list)


@dataclass
class ResourceEstimate:
    """Pre-synthesis resource estimate (informational only)."""
    lut_estimate: int = 0
    bram_bits:    int = 0
    dsp_blocks:   int = 0
    notes:        str = ""


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class AcceleratorSpec(ABC):
    """Pure behavioral model — the definition of correct output for any input.

    Implementations must be correct-first. No timing, no cycles, no RTL
    concepts. This is the contract every backend RTL implementation must
    satisfy.  For most TM variants, this delegates to
    matador.inference.reference.predict() which is already correct-first.
    """

    def __init__(self, tmir) -> None:
        self.tmir = tmir

    @abstractmethod
    def simulate(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (predictions, scores) for a batch of Boolean feature vectors.

        Args:
            X: uint8 array of shape (n_samples, n_features), values 0/1.

        Returns:
            predictions: shape (n_samples,)           — argmax class per sample.
            scores:      shape (n_samples, n_classes)  — raw class scores.
        """


class RTLBackend(ABC):
    """Generates synthesisable RTL for one accelerator architecture.

    Each backend owns its config schema (feat_slice/clause_slice for tiled,
    pipeline_stages for hardwired, etc.) and its complete set of generated
    files.  The CLI routes to the correct backend via the --backend flag.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used as the --backend flag value."""

    @property
    @abstractmethod
    def config_class(self) -> type:
        """Pydantic BaseModel subclass for this backend's YAML config."""

    @abstractmethod
    def generate(self, tmir, config) -> RTLArtifacts:
        """Write all RTL sources, testbenches, and sim scripts.

        Args:
            tmir:   Loaded and validated TMIR model.
            config: Instance of self.config_class.

        Returns:
            RTLArtifacts manifest of every file written.
        """

    def resource_estimate(self, tmir, config) -> ResourceEstimate:
        """Pre-synthesis resource estimate. Override for synthesis guidance."""
        return ResourceEstimate(notes="No estimate available for this backend.")

    @property
    def emulator_class(self):
        """Return the CycleAccurateModel class paired with this backend, or None."""
        return None


class CycleAccurateModel(ABC):
    """Cycle-accurate Python model of one specific RTLBackend.

    Shipped paired with its RTLBackend. Enables emulation-based debugging
    without a simulator, and drives automatic testbench generation: the
    model computes expected signal values at each cycle, the TestbenchFactory
    turns those into Verilog assertions.

    A backend that provides a CycleAccurateModel gets unit testbenches for
    free.  A backend that does not must write testbenches by hand.
    """

    def __init__(self, tmir, config) -> None:
        self.tmir   = tmir
        self.config = config

    @abstractmethod
    def run(self, input_vector: list[int]):
        """Simulate one inference sample, returning an InferenceTrace."""

    @abstractmethod
    def unit_testbenches(self, vectors: list) -> dict[str, str]:
        """Generate Verilog testbench source for each sub-module.

        Returns:
            Dict mapping module name to Verilog source string, e.g.:
              {"clause_eval": "module tb_clause_eval; ..."}
        """
