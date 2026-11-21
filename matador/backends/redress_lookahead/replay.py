"""Corrupted-model replay: load a stream with known bit flips into the RTL
and check its predictions against decode_faithful().

This is the check that makes decode_faithful trustworthy as a fault model: if
the RTL and the software model agree on corrupted streams, campaign numbers
computed in software (fast) stand for the hardware (slow).

It reuses RedressBackend's Stage 1 hooks: a throwaway subclass whose
encode_stream() returns the corrupted words, and whose expected_predictions()
is decode_faithful() in the matching decoder mode.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead.faithful import Geometry, decode_faithful

FIELD_BITS = {"P": [15], "CC": [14], "E": [13], "offset": list(range(1, 13)), "L": [0]}
BIT_FIELD = {b: f for f, bits in FIELD_BITS.items() for b in bits}


def flip(words: list[int], index: int, bit: int) -> list[int]:
    """A copy of words with one bit of one word inverted."""
    out = list(words)
    out[index] ^= 1 << bit
    return out


def replay_safe(word: int, n_features: int, feat_addr_w: int) -> bool:
    """False if this word reads a Feature FIFO entry the batch never loads
    (X in simulation, 0 in decode_faithful -- see faithful.py)."""
    offset = (word >> 1) & 0xFFF
    return (offset & ((1 << feat_addr_w) - 1)) < n_features


def build_replay_bundle(tmir, words, config, *, mode=None, backend_cls=RedressBackend):
    """Generate a bundle whose Instruction Memory holds `words` exactly and
    whose expected outputs come from decode_faithful(**mode)."""
    mode = dict(mode or {})
    geometry = Geometry.from_config(config)
    fixed_words = list(words)

    class ReplayBackend(backend_cls):
        def encode_stream(self, tmir, config):
            return list(fixed_words)

        def expected_predictions(self, w, n_classes, X_pack, config):
            pred, _ = decode_faithful(
                w, n_classes, X_pack, config.batch_width,
                class_width=geometry.class_width, max_classes=geometry.max_classes,
                feat_addr_w=geometry.feat_addr_w, **mode,
            )
            return pred

    return ReplayBackend().generate(tmir, config)


def run_iverilog(artifacts, tb_name: str = "tb_redress_model") -> tuple[bool, str]:
    """Compile and run one testbench of a generated bundle under iverilog.
    Returns (passed, combined output)."""
    if shutil.which("iverilog") is None:
        raise RuntimeError("iverilog not found on PATH")
    rtl_dir = Path(artifacts.rtl_dir).resolve()  # vvp runs with cwd=sim/
    sim_dir = rtl_dir / "sim"
    tb = rtl_dir / "tb" / f"{tb_name}.v"
    binary = sim_dir / tb_name
    cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(binary)]
        + [str(Path(s).resolve()) for s in artifacts.sources] + [str(tb)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        return False, cp.stdout + cp.stderr
    run = subprocess.run(["vvp", str(binary)], capture_output=True, text=True, cwd=str(sim_dir))
    out = run.stdout + run.stderr
    return ("TESTS PASSED" in out and "FAIL" not in out), out
