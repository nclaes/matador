"""RTL generator for redress_lookahead.

RedressBackend does all the work; this subclass only swaps what the Stage 1
hooks expose:

  _src_dir / _top_module   redress_la_accel.v (static submodules are reused
                           from redress/src and common/src)
  encode_stream()          codec_la.encode(): mustang_encode + hardening
  expected_predictions()   decode_faithful() in this config's decoder mode
  _extra_params()          E_LA_DEPTH, CC_LA
"""

from __future__ import annotations

from pathlib import Path

from matador.backends.redress.rtl import RedressBackend

_SRC_DIR = Path(__file__).parent / "src"


class RedressLookaheadBackend(RedressBackend):
    """redress with E/CC look-ahead decoding: a single flipped E (or CC) bit
    in Instruction Memory no longer shifts every later class (clause)."""

    _src_dir = _SRC_DIR
    _top_module = "redress_la_accel"

    @property
    def name(self) -> str:
        return "redress_lookahead"

    @property
    def config_class(self) -> type:
        from matador.backends.redress_lookahead.config import RedressLookaheadConfig
        return RedressLookaheadConfig

    # ── Stage 1 hooks ─────────────────────────────────────────────────────

    def encode_stream(self, tmir, config) -> list[int]:
        from matador.backends.redress_lookahead.codec_la import encode
        return encode(tmir, e_depth=config.e_lookahead_depth, cc_lookahead=config.cc_lookahead)

    def expected_predictions(self, words, n_classes: int, X_pack, config) -> list[int]:
        from matador.backends.redress_lookahead.faithful import Geometry, decode_faithful

        g = Geometry.from_config(config)
        pred, _ = decode_faithful(
            words, n_classes, X_pack, config.batch_width,
            class_width=g.class_width, max_classes=g.max_classes, feat_addr_w=g.feat_addr_w,
            **config.decoder_mode,
        )
        return pred

    def _extra_params(self, config) -> list[tuple[str, int]]:
        return [("E_LA_DEPTH", config.e_lookahead_depth), ("CC_LA", int(config.cc_lookahead))]

    # ── waveform layout: add the look-ahead decision signals ─────────────

    def _gen_tb_gtkw(self, tb_name: str) -> str:
        from matador.waves.gtkwave import _BIN, _DEC, _HEX, _blank, _group, _group_end, _sig

        T = f"{tb_name}.dut"
        group = "Look-ahead decision (S_INFER_ACCUM: state=6)"
        extra = [_blank(), _group(group)]
        extra += [_sig(f"{T}.state", _DEC), _sig(f"{T}.ptr", _DEC),
                  _sig(f"{T}.instr_latched", _HEX), _sig(f"{T}.instr_rdata", _HEX)]
        for name in ("E_cur", "E_prev", "E_next", "have_next", "class_boundary",
                     "CC_cur", "CC_prev", "CC_next", "clause_boundary"):
            extra.append(_sig(f"{T}.{name}", _BIN))
        extra += [_sig(f"{T}.current_class", _DEC), _group_end(group)]
        return super()._gen_tb_gtkw(tb_name) + "".join(extra)
