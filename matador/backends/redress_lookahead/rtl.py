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

    @property
    def emulator_class(self):
        from matador.backends.redress_lookahead.emulator import RedressLookaheadEmulator
        return RedressLookaheadEmulator

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

    # ── generated RTL/README.md ──────────────────────────────────────────

    def _gen_readme(self, config, tmir, E_words: list[int]) -> str:
        from matador.backends.redress.codec import mustang_encode

        text = super()._gen_readme(config, tmir, E_words)
        text = text.replace(
            "# redress — MUSTANG Single-Core accelerator",
            "# redress_lookahead — MUSTANG Single-Core accelerator with E/CC look-ahead", 1)
        text = text.replace(
            "against `codec.mustang_decode_and_infer()`",
            "against `redress_lookahead.faithful.decode_faithful()` (the RTL-exact model)", 1)
        plain = len(mustang_encode(tmir))
        extra = 100.0 * (len(E_words) - plain) / plain if plain else 0.0
        d, cc = config.e_lookahead_depth, config.cc_lookahead
        return text + f"""
## Look-ahead (what this backend adds to redress)

  e_lookahead_depth = {d}   (E_LA_DEPTH)
  cc_lookahead      = {cc}   (CC_LA = {int(cc)})
  stored words      = {len(E_words):,}  (plain MUSTANG: {plain:,}; +{extra:.1f}% from hardening)

**The failure it fixes.** As built, a flipped E bit on a word inside a class
differs from the latched E, so the decoder starts a new class; the next
(correct) word differs again, so it starts another. Every later class is then
decoded two classes too high. CC flips do the same to clause boundaries.

**The rule.** An E change is accepted only if the next word (depth 1) or the
next two words (depth 2) exist and carry the same new E. A rejected change
leaves the E latch untouched. The E latch starts every batch at 1 (class 0's
E) and counts as set, so word 0 is checked too. `cc_lookahead` applies the
one-word rule to CC. The next word is read in Instruction Memory's idle slot
(issued in S_INFER_DECODE, on `instr_rdata` in S_INFER_ACCUM); at depth 2 the
word after it is read in the other idle slot (issued in S_INFER_INSTR,
captured into `E_next2` in S_INFER_FEAT). Inference takes exactly as many
cycles as redress.

**The encoder** (`redress_lookahead.codec_la.encode`) pads every class to at
least 4 words at depth 1 (the minimum for which no single E flip can cascade)
or 7 at depth 2 (the minimum for which no two adjacent E flips can), and, with
`cc_lookahead`, stores every one-literal clause's literal twice (x AND x = x).
The stored format is unchanged: a hardened stream also runs on plain redress.

**In the waves**, open the "Look-ahead decision" group. Read `class_boundary` /
`clause_boundary` only where `state = 6` (S_INFER_ACCUM); in other states
`instr_rdata` is not yet the next word.

**Fault-injection checks** (from the matador repo):

```
# exhaustive single-bit Instruction Memory campaign, N x P per field
python -m matador.backends.redress_lookahead.campaign --model <this TMIR> --variant v0 v3 v4

# corrupted-model replay: load a flipped stream into this RTL and check it
# against decode_faithful -- redress_lookahead/replay.py:
#   build_replay_bundle(tmir, flip(words, i, bit), config, mode=config.decoder_mode,
#                       backend_cls=RedressLookaheadBackend), then run_iverilog(...)
```
"""

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
