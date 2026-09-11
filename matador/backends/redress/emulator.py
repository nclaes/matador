"""REDRESS backend emulator.

Like GPTiledEmulator, does not model the AXI-Stream Header/Instruction/
Feature packet protocol or FIFO -- that's protocol-level behaviour covered
by RTL simulation (matador simulate), not this Python model. Unlike
GPTiledEmulator (which re-derives its own per-tile trace), this one calls
straight into matador.backends.redress.codec's already-verified MUSTANG
decode -- codec.py is the single source of truth for what "correct" means
here (see its own verification against matador.inference.reference.predict);
duplicating that logic a second time inside the emulator would just be a
second place for the same bug to hide.

Single-inference semantics: run() decodes exactly one datapoint by feeding
mustang_decode_and_infer() a batch of width 1 (a real MUSTANG accelerator
always processes config.batch_width datapoints in parallel; the individual
per-lane behaviour a width-1 batch exercises is identical to what lane 0 of
a full-width batch would compute for the same input, so this is a faithful
single-sample view, not an approximation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from matador.backends.redress import codec

if TYPE_CHECKING:
    from matador.backends.redress.config import RedressAcceleratorConfig
    from matador.ir.tm_ir import TMIR


@dataclass
class RedressArgmaxEvent:
    scores: list[int]
    predicted_class: int


@dataclass
class RedressInferenceTrace:
    """Result of one REDRESS inference, computed via
    codec.mustang_decode_and_infer()."""
    feature_vector: list[int]
    scores: list[int] = field(default_factory=list)
    argmax_event: RedressArgmaxEvent | None = None
    predicted_class: int = -1


class RedressEmulator:
    """Cycle-accurate (per-instruction) emulator for the REDRESS
    accelerator, built directly on codec.py's MUSTANG decode."""

    def __init__(self, tmir: "TMIR", config: "RedressAcceleratorConfig") -> None:
        if tmir.variant != "vanilla":
            raise ValueError(f"redress only supports 'vanilla' TM, got '{tmir.variant}'")
        self.tmir = tmir
        self.config = config
        self.n_classes = tmir.architecture.n_classes
        self.n_features = tmir.architecture.n_features
        # Encoded once here, not per run() call -- same rationale as
        # GPTiledEmulator caching pack_tiles(): mustang_encode() walks the
        # whole model, so a many-vector test run should pay that cost once.
        self.E_words = codec.mustang_encode(tmir)

    def run(self, input_vector: list[int]) -> RedressInferenceTrace:
        """Emulate one inference (a width-1 batch through mustang_decode_and_infer)."""
        trace = RedressInferenceTrace(feature_vector=list(input_vector))

        X_pack = codec.pack_features(
            [input_vector[: self.n_features]], W=1
        )
        best_class, best_score = codec.mustang_decode_and_infer(
            self.E_words, self.n_classes, X_pack, W=1
        )
        all_scores = codec._mustang_full_class_scores(
            self.E_words, self.n_classes, X_pack, W=1
        )
        scores = all_scores[0].tolist()

        trace.scores = scores
        trace.argmax_event = RedressArgmaxEvent(scores=scores, predicted_class=best_class[0])
        trace.predicted_class = best_class[0]
        return trace

    def unit_testbenches(self, vectors: list) -> dict[str, str]:
        """No unit testbenches for the REDRESS backend (single-module
        design, same as hardwired) -- see rtl.py for the system-level
        testbench this backend does generate."""
        return {}
