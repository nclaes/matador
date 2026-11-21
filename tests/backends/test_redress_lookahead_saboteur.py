"""Stage 7: the mini saboteur (docs/lookahead_exercise/scripts/saboteur.py),
made deterministic. Flip a named register at a chosen inference cycle and
check which design notices.

Inference cycle 4*j + k is word j's INSTR (k=0), DECODE (1), FEAT (2) or
ACCUM (3) state, counting from the first S_INFER_INSTR of the batch.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest

from matador.backends.redress.config import RedressAcceleratorConfig
from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead import campaign
from matador.backends.redress_lookahead.codec_la import split_stream
from matador.backends.redress_lookahead.config import RedressLookaheadConfig
from matador.backends.redress_lookahead.rtl import RedressLookaheadBackend

pytestmark = pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")

_SCRIPT = Path(__file__).parents[2] / "docs" / "lookahead_exercise" / "scripts" / "saboteur.py"
_spec = importlib.util.spec_from_file_location("saboteur", _SCRIPT)
saboteur = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(saboteur)

DESIGNS = {
    "redress": (RedressBackend, RedressAcceleratorConfig),
    "redress_lookahead": (RedressLookaheadBackend, RedressLookaheadConfig),
}


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    out = tmp_path_factory.mktemp("sab")
    tmir = campaign.make_synthetic_tmir(n_classes=4, cpc=12, density=0.08, n_samples=32, seed=1)
    tmir.to_yaml(out / "model.yaml")
    built = {}
    for name, (cls, cfg_cls) in DESIGNS.items():
        cfg = cfg_cls.model_validate(dict(
            model_path=str(out / "model.yaml"), output_dir=str(out / name), target_fpga="xc7z020",
            max_includes=4096, max_features=256, max_classes=4, batch_width=32))
        backend = cls()
        art = backend.generate(tmir, cfg)
        sim = (art.rtl_dir / "sim").resolve()
        tb = sim / "tb_saboteur.v"
        model_tb = (art.rtl_dir / "tb" / "tb_redress_model.v").read_text()
        tb.write_text(saboteur.make_saboteur_tb(model_tb))
        words = backend.encode_stream(tmir, cfg)
        lens = [sum(map(len, c)) for c in split_stream(words)]
        j = lens[0] + lens[1] // 2  # a word in the middle of class 1
        built[name] = (sim, [str(Path(s).resolve()) for s in art.sources], tb, j)
    return built


def _changed(bundle, ref: str, cycle: int) -> bool:
    sim, srcs, tb, _ = bundle
    binary = sim / f"sab_{abs(hash(ref))}"
    if not binary.exists():
        cp = subprocess.run(["iverilog", "-g2001", "-Wno-timescale", f"-DSAB_TARGET={ref}",
                             "-o", str(binary), *srcs, str(tb)], capture_output=True, text=True)
        assert cp.returncode == 0, cp.stderr
    r = subprocess.run(["vvp", "-n", str(binary), f"+FLIP_CYCLE={cycle}"],
                       capture_output=True, text=True, cwd=str(sim))
    # (the log line prints the macro name, not its value: iverilog doesn't
    # expand macros inside string literals)
    assert f"at inference cycle {cycle} (state {3 + cycle % 4})" in r.stdout, r.stdout[-300:]
    return "TESTS PASSED" not in r.stdout


def test_no_flip_passes(bundles):
    for b in bundles.values():
        sim, srcs, tb, _ = b
        binary = sim / "sab_none"
        subprocess.run(["iverilog", "-g2001", "-Wno-timescale", "-DSAB_TARGET=dut.E_prev",
                        "-o", str(binary), *srcs, str(tb)], check=True, capture_output=True)
        r = subprocess.run(["vvp", "-n", str(binary)], capture_output=True, text=True, cwd=str(sim))
        assert "TESTS PASSED" in r.stdout and "SABOTEUR" not in r.stdout


def test_current_word_e_flip_only_hurts_as_built(bundles):
    """instr_latched[13] during ACCUM of an interior word: the same thing an E
    flip in memory does, for one batch. Look-ahead rejects it."""
    for name, expect in (("redress", True), ("redress_lookahead", False)):
        j = bundles[name][3]
        assert _changed(bundles[name], "dut.instr_latched[13]", 4 * j + 3) is expect, name


def test_e_latch_flip_hurts_both_designs(bundles):
    """E_prev is state, not a word: once it is wrong, the next CORRECT word
    looks like a change, and the word after confirms it. Look-ahead can't tell
    -- the latch needs its own (Tier 1) protection."""
    for name in DESIGNS:
        j = bundles[name][3]
        assert _changed(bundles[name], "dut.E_prev", 4 * j + 3), name
