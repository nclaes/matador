"""redress_lookahead emulator.

Unlike RedressEmulator (which calls the codec, the format's specification),
this one is built on decode_faithful(): the software model of
redress_la_accel.v in this config's decoder mode, on the hardened stream the
RTL actually loads. On clean data the two give the same answers; only
decode_faithful also gives the RTL's answers once bits are flipped.

Single-inference semantics: run() decodes one datapoint as lane 0 of a
width-1 batch, the same convention as RedressEmulator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from matador.backends.redress.codec import pack_features
from matador.backends.redress.emulator import RedressArgmaxEvent, RedressInferenceTrace
from matador.backends.redress_lookahead.codec_la import encode
from matador.backends.redress_lookahead.faithful import Geometry, decode_faithful

if TYPE_CHECKING:
    from matador.backends.redress_lookahead.config import RedressLookaheadConfig
    from matador.ir.tm_ir import TMIR


class RedressLookaheadEmulator:
    """Per-inference emulator for redress_la_accel, on decode_faithful()."""

    def __init__(self, tmir: TMIR, config: RedressLookaheadConfig, words=None) -> None:
        if tmir.variant != "vanilla":
            raise ValueError(f"redress_lookahead only supports 'vanilla' TM, got '{tmir.variant}'")
        self.tmir = tmir
        self.config = config
        self.n_classes = tmir.architecture.n_classes
        self.n_features = tmir.architecture.n_features
        self.geometry = Geometry.from_config(config)
        # `words` lets a caller emulate a corrupted Instruction Memory.
        self.words = list(words) if words is not None else encode(
            tmir, e_depth=config.e_lookahead_depth, cc_lookahead=config.cc_lookahead)

    def run(self, input_vector: list[int]) -> RedressInferenceTrace:
        trace = RedressInferenceTrace(feature_vector=list(input_vector))
        X_pack = pack_features([input_vector[: self.n_features]], W=1)
        g = self.geometry
        pred, scores = decode_faithful(
            self.words, self.n_classes, X_pack, 1,
            class_width=g.class_width, max_classes=g.max_classes, feat_addr_w=g.feat_addr_w,
            **self.config.decoder_mode,
        )
        trace.scores = scores[0, : self.n_classes].tolist()
        trace.argmax_event = RedressArgmaxEvent(scores=trace.scores, predicted_class=pred[0])
        trace.predicted_class = pred[0]
        return trace

    def unit_testbenches(self, vectors: list) -> dict[str, str]:
        """No unit testbenches (single-module design, same as redress)."""
        return {}
