"""GP-tiled backend emulator.

Cycle-accurate (per-tile) Python model of tm_accel_gp, delegating to
GP_TM_Inference_Accelerator's own golden model (TMModel.infer_tiled),
which replicates the RTL's per-tile AND/OR accumulation exactly — not just
the mathematical reference semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from matador.backends.gp_tiled.tmir_bridge import check_capacity, tmir_to_tmmodel

if TYPE_CHECKING:
    from matador.backends.gp_tiled.config import GPTiledAcceleratorConfig
    from matador.ir.tm_ir import TMIR

_VENDOR_TB_DIR = Path(__file__).parent / "vendor" / "tb"


@dataclass
class GPArgmaxEvent:
    scores: list
    predicted_class: int


@dataclass
class GPInferenceTrace:
    """Result of one tm_accel_gp inference, computed via TMModel.infer_tiled()."""
    feature_vector: list
    scores: list = field(default_factory=list)
    argmax_event: GPArgmaxEvent | None = None
    predicted_class: int = -1


class GPTiledEmulator:
    """Cycle-accurate (per-tile) emulator for tm_accel_gp.

    Does not model the AXI-Stream LOAD/INFER protocol or FIFO — like
    HardwiredEmulator, it takes a flat feature vector and returns a trace.
    Protocol-level behaviour (LOAD/INFER framing, error injection/recovery)
    is covered by the vendored tb_system_gp.v testbench instead, exercised
    through RTL simulation (matador simulate), not this Python model.
    """

    def __init__(self, tmir: "TMIR", config: "GPTiledAcceleratorConfig") -> None:
        if tmir.variant != "vanilla":
            raise ValueError(f"vanilla_gp_tiled only supports 'vanilla' TM, got '{tmir.variant}'")
        check_capacity(tmir, config)
        self.tmir = tmir
        self.config = config
        self.model = tmir_to_tmmodel(tmir, name=config.model_path.stem)

    def run(self, input_vector: list) -> GPInferenceTrace:
        """Emulate one inference. Matches tm_accel_gp RTL behaviour exactly
        (via TMModel.infer_tiled's per-tile AND/OR replica)."""
        n = self.model.n_features
        if len(input_vector) != n:
            raise ValueError(f"Expected {n} features, got {len(input_vector)}")

        fv_int = 0
        for f, bit in enumerate(input_vector):
            if int(bit) & 1:
                fv_int |= 1 << f

        predicted, scores = self.model.infer_tiled(fv_int)

        trace = GPInferenceTrace(feature_vector=list(input_vector), scores=scores)
        trace.argmax_event = GPArgmaxEvent(scores=scores, predicted_class=predicted)
        trace.predicted_class = predicted
        return trace

    def unit_testbenches(self, vectors: list) -> dict[str, str]:
        """Return the vendored (hand-written, not per-vector generated)
        testbenches, keyed by module name. Coarser than HardwiredBackend's
        per-vector generation: tb_system_gp exercises the LOAD/INFER
        protocol generically rather than being regenerated per test vector."""
        names = ("tb_tile_mem", "tb_score_acc_rt", "tb_argmax_rt", "tb_system_gp")
        return {name: (_VENDOR_TB_DIR / f"{name}.v").read_text() for name in names}
