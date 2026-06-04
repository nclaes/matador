"""Hardwired backend RTL generator — Phase 2 placeholder.

TA Include actions will be unrolled into combinational AND-gate logic.
A pipelined adder tree accumulates per-class scores.

Status: Not yet implemented.  Use --backend tiled in the meantime.
"""

from __future__ import annotations

from matador.backends.base import RTLArtifacts, RTLBackend


class HardwiredBackend(RTLBackend):
    """Hardwired combinational TM accelerator (Phase 2).

    Each clause becomes a fixed AND-gate mask wired directly to the feature
    inputs — no ROM lookup, no sequential FSM for clause evaluation.
    A balanced binary adder tree accumulates the per-class scores in
    ceil(log2(n_clauses_per_class/2)) pipeline stages.

    Tuning knob (in HardwiredAcceleratorConfig):
      pipeline_stages — adder tree register depth (0 = fully combinational)

    Trade-off: lower latency than tiled for small models; LUT cost grows
    linearly with clause × feature count.
    """

    @property
    def name(self) -> str:
        return "hardwired"

    @property
    def config_class(self) -> type:
        from matador.backends.hardwired.config import HardwiredAcceleratorConfig
        return HardwiredAcceleratorConfig

    def generate(self, tmir, config) -> RTLArtifacts:
        raise NotImplementedError(
            "The hardwired backend is not yet implemented (Phase 2). "
            "Use --backend tiled for now."
        )
