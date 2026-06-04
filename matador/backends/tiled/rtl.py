"""Tiled backend RTL generator.

Delegates to matador.rtl.accelerator.TMAccelerator — the existing
production implementation.  This wrapper satisfies the RTLBackend ABC
so the tiled design participates in the plugin registry.
"""

from __future__ import annotations

from matador.backends.base import ResourceEstimate, RTLArtifacts, RTLBackend


class TiledBackend(RTLBackend):
    """Tiled feature-clause matrix accelerator.

    Sequential FSM iterates over (N_FEAT_SLICES × N_CLAUSE_SLICES) tiles per
    inference, reading Include-action bits from an on-chip tile ROM each cycle.
    CLAUSE_SLICE clause_eval instances run in parallel per cycle.

    Tuning knobs (in TiledAcceleratorConfig):
      feat_slice    — features per tile column  (default 4, powers of 2 recommended)
      clause_slice  — clauses per tile row      (default 4, powers of 2 recommended)

    Trade-off: larger slices → fewer FSM cycles → more LUTs and wider ROM port.
    """

    @property
    def name(self) -> str:
        return "tiled"

    @property
    def config_class(self) -> type:
        from matador.backends.tiled.config import TiledAcceleratorConfig
        return TiledAcceleratorConfig

    def generate(self, tmir, config) -> RTLArtifacts:
        from matador.rtl.accelerator import TMAccelerator
        accel   = TMAccelerator(tmir, config)
        rtl_dir = accel.generate()
        src_dir = rtl_dir / "src"
        tb_dir  = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"
        return RTLArtifacts(
            rtl_dir    = rtl_dir,
            sources    = sorted(src_dir.glob("*.v")),
            testbenches= sorted(tb_dir.glob("*.v")),
            sim_scripts= sorted(sim_dir.glob("*.sh")) + sorted(sim_dir.glob("*.gtkw")),
        )

    def resource_estimate(self, tmir, config) -> ResourceEstimate:
        return ResourceEstimate(
            notes=(
                "Tile ROM inferred as distributed RAM; "
                f"depth = N_FEAT_SLICES × N_CLAUSE_SLICES, "
                f"width = CLAUSE_SLICE × 2 × FEAT_SLICE bits. "
                "Run Vivado synthesis for exact LUT/BRAM numbers."
            )
        )
