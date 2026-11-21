"""Stage 5: redress_lookahead as a registered backend -- registry, config,
emulator, generate, iverilog/Verilator simulation, reprogram suite and CLI.
Mirrors tests/backends/test_redress_rtl.py."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from matador.backends.base import ReprogramStep
from matador.backends.redress import codec
from matador.backends.redress.config import RedressAcceleratorConfig
from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead import campaign
from matador.backends.redress_lookahead.config import RedressLookaheadConfig
from matador.backends.redress_lookahead.emulator import RedressLookaheadEmulator
from matador.backends.redress_lookahead.rtl import RedressLookaheadBackend
from matador.backends.registry import config_class_for, describe, get, list_backends
from matador.cli import main
from matador.inference.reference import predict

_HAVE_IVERILOG = shutil.which("iverilog") is not None
_HAVE_VERILATOR = shutil.which("verilator") is not None


def _tmir(n_classes=3, n_samples=40, seed=0, density=0.08, cpc=12):
    return campaign.make_synthetic_tmir(n_classes=n_classes, cpc=cpc, density=density,
                                        n_samples=n_samples, seed=seed)


def _payload(tmp_path, tmir, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.yaml"
    tmir.to_yaml(path)
    payload = dict(model_path=str(path), output_dir=str(tmp_path), target_fpga="xc7z020",
                   max_includes=4096, max_features=256, max_classes=16, batch_width=32)
    payload.update(overrides)
    return payload


def _cfg(tmp_path, tmir, **overrides) -> RedressLookaheadConfig:
    return RedressLookaheadConfig.model_validate(_payload(tmp_path, tmir, **overrides))


# ── registry ─────────────────────────────────────────────────────────────

def test_registered():
    assert "redress_lookahead" in list_backends()
    assert get("redress_lookahead") is RedressLookaheadBackend
    assert config_class_for("redress_lookahead") is RedressLookaheadConfig
    assert "look-ahead" in describe("redress_lookahead")


def test_backend_properties():
    b = RedressLookaheadBackend()
    assert b.name == "redress_lookahead"
    assert b.supports_reprogramming is True
    assert b.emulator_class is RedressLookaheadEmulator


# ── config ───────────────────────────────────────────────────────────────

def test_config_defaults_and_inheritance(tmp_path):
    cfg = _cfg(tmp_path, _tmir())
    assert isinstance(cfg, RedressAcceleratorConfig)
    assert cfg.e_lookahead_depth == 1 and cfg.cc_lookahead is True
    assert cfg.decoder_mode == {"e_depth": 1, "cc_lookahead": True, "e_init": 1}


@pytest.mark.parametrize("depth", [3, -1])
def test_config_rejects_unimplemented_depths(tmp_path, depth):
    with pytest.raises(ValueError, match="no RTL implementation"):
        _cfg(tmp_path, _tmir(), e_lookahead_depth=depth)


def test_config_still_enforces_redress_budget(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, _tmir(), max_includes=1_000_000)


# ── emulator ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("depth,cc", [(0, False), (1, False), (1, True), (2, True)])
def test_emulator_matches_reference(tmp_path, depth, cc):
    tmir = _tmir(n_classes=4, n_samples=30, seed=3)
    emu = RedressLookaheadEmulator(tmir, _cfg(tmp_path, tmir, e_lookahead_depth=depth,
                                              cc_lookahead=cc))
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    ref, ref_scores = predict(tmir, X)
    for i in range(len(X)):
        trace = emu.run(X[i].tolist())
        assert trace.predicted_class == int(ref[i])
        assert trace.scores == ref_scores[i].tolist()


def test_emulator_can_run_a_corrupted_stream(tmp_path):
    """Depth 0 + CC off on a corrupted stream = redress's as-built hardware,
    which a single interior E flip breaks; depth 1 shrugs it off."""
    tmir = _tmir(n_classes=4, n_samples=30, seed=3)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = RedressLookaheadBackend().encode_stream(tmir, _cfg(tmp_path, tmir))
    bad = list(words)
    bad[len(words) // 2] ^= 1 << 13
    clean = [RedressLookaheadEmulator(tmir, _cfg(tmp_path, tmir)).run(x.tolist()).predicted_class
             for x in X]
    la = RedressLookaheadEmulator(tmir, _cfg(tmp_path, tmir), words=bad)
    v0 = RedressLookaheadEmulator(tmir, _cfg(tmp_path, tmir, e_lookahead_depth=0,
                                             cc_lookahead=False), words=bad)
    assert [la.run(x.tolist()).predicted_class for x in X] == clean
    assert [v0.run(x.tolist()).predicted_class for x in X] != clean


# ── generate ─────────────────────────────────────────────────────────────

def test_generate_writes_expected_files(tmp_path):
    tmir = _tmir()
    art = RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir))
    assert {p.name for p in art.sources} == {
        "redress_bram.v", "score_acc_rt.v", "argmax_rt.v", "redress_la_accel.v"}
    sim = art.rtl_dir / "sim"
    for name in ("model_stimulus.memh", "model_expected.memh", "run_iverilog.sh", "waves.sh",
                 "tb_redress_model.gtkw", "test_vectors.txt", "verilator/Makefile",
                 "verilator/tb_top.cpp"):
        assert (sim / name).exists(), name
    assert "Vredress_la_accel" in (sim / "verilator" / "Makefile").read_text()
    readme = (art.rtl_dir / "README.md").read_text()
    assert readme.startswith("# redress_lookahead")
    assert "## Look-ahead (what this backend adds to redress)" in readme
    assert "decode_faithful" in readme


def test_depth0_cc_off_reproduces_redress_bundle_inputs(tmp_path):
    """The A/B switch: with look-ahead off, the stored stream and every
    expected output are identical to redress's."""
    tmir = _tmir(n_classes=4, n_samples=70, seed=6)
    la_cfg = _cfg(tmp_path / "la", tmir, e_lookahead_depth=0, cc_lookahead=False)
    rd_cfg = RedressAcceleratorConfig.model_validate(_payload(tmp_path / "rd", tmir))
    la = RedressLookaheadBackend().generate(tmir, la_cfg)
    rd = RedressBackend().generate(tmir, rd_cfg)
    for f in ("model_stimulus.memh", "model_expected.memh"):
        assert (la.rtl_dir / "sim" / f).read_text() == (rd.rtl_dir / "sim" / f).read_text()


def test_generate_rejects_model_exceeding_max_includes(tmp_path):
    tmir = _tmir(density=0.6)
    with pytest.raises(ValueError, match="max_includes"):
        RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir, max_includes=8))


def test_capacity_check_counts_hardening_words(tmp_path):
    """A model that fits redress exactly can fail redress_lookahead: the
    capacity check runs on the HARDENED stream."""
    tmir = _tmir(n_classes=4, density=0.03, seed=2)
    plain = len(codec.mustang_encode(tmir))
    hard = len(RedressLookaheadBackend().encode_stream(tmir, _cfg(tmp_path, tmir)))
    assert hard > plain
    with pytest.raises(ValueError, match="max_includes"):
        RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir, max_includes=plain))


def test_generate_rejects_empty_class(tmp_path):
    tmir = _tmir(n_classes=2)
    inc = np.asarray(tmir.representation.includes).copy()
    inc[1] = False
    tmir.representation.includes = inc
    with pytest.raises(codec.MustangEmptyClassError):
        RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir))


# ── simulation ───────────────────────────────────────────────────────────

def _run_bundle_iverilog(rtl_dir):
    cp = subprocess.run(["bash", str(rtl_dir / "sim" / "run_iverilog.sh")],
                        capture_output=True, text=True)
    return cp.stdout + cp.stderr


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
@pytest.mark.parametrize("n_samples", [8, 97], ids=["single-batch", "multi-batch"])
def test_rtl_simulation_passes(tmp_path, n_samples):
    tmir = _tmir(n_classes=4, n_samples=n_samples, seed=7)
    art = RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir))
    out = _run_bundle_iverilog(art.rtl_dir)
    n_beats = -(-n_samples // 32) * 32
    assert f"ALL {n_beats} TESTS PASSED" in out, out[-500:]


@pytest.mark.skipif(not _HAVE_VERILATOR, reason="verilator not installed")
@pytest.mark.parametrize("depth", [1, 2])
def test_verilator_harness_passes(tmp_path, depth):
    tmir = _tmir(n_classes=4, n_samples=70, seed=8)
    art = RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir, e_lookahead_depth=depth))
    vdir = art.rtl_dir / "sim" / "verilator"
    build = subprocess.run(["make", "-C", str(vdir), "all"], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout[-1000:] + build.stderr[-1000:]
    run = subprocess.run([str(vdir / "obj_dir" / "Vredress_la_accel")], capture_output=True,
                         text=True, cwd=str(vdir))
    assert "ALL TESTS PASSED (96 beats checked)" in run.stdout + run.stderr, run.stdout[-500:]


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_reprogram_suite_across_two_models(tmp_path):
    a, b = _tmir(n_classes=3, n_samples=40, seed=10), _tmir(n_classes=4, n_samples=50, seed=11)
    path_b = tmp_path / "b.yaml"
    cfg = _cfg(tmp_path, a)
    b.to_yaml(path_b)
    backend = RedressLookaheadBackend()
    art = backend.generate(a, cfg)
    suite = backend.build_reprogram_suite(
        art.rtl_dir, [ReprogramStep(tmir_path=cfg.model_path, name="a"),
                      ReprogramStep(tmir_path=path_b, name="b")], cfg)
    assert "redress_la_accel #(" in suite.testbenches[0].read_text()
    out = subprocess.run(["bash", str(art.rtl_dir / "sim" / "run_iverilog.sh")],
                         capture_output=True, text=True).stdout
    assert "tb_reprogram_suite: ALL" in out and "FAIL" not in out, out[-500:]


# ── CLI ──────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_cli_generate_and_simulate(tmp_path):
    tmir = _tmir(n_classes=4, n_samples=40, seed=12)
    cfg_path = tmp_path / "redress_lookahead.yaml"
    cfg_path.write_text(yaml.safe_dump(_payload(tmp_path, tmir)))
    runner = CliRunner()
    args = ["--backend", "redress_lookahead", "--config", str(cfg_path)]
    gen = runner.invoke(main, ["generate", *args])
    assert gen.exit_code == 0, gen.output
    assert (tmp_path / "redress_lookahead" / "RTL" / "src" / "redress_la_accel.v").exists()
    sim = runner.invoke(main, ["simulate", *args])
    assert sim.exit_code == 0, sim.output
    assert "0 failed" in sim.output


@pytest.mark.skipif(not (_HAVE_IVERILOG and _HAVE_VERILATOR), reason="needs iverilog + verilator")
def test_cli_simulate_with_relative_output_dir(tmp_path, monkeypatch):
    """Regression: validate_rtl ran the Verilator binary by a path relative to
    the caller's cwd while setting cwd=verilator_dir, so `matador simulate`
    failed for any config with a relative output_dir (every backend)."""
    monkeypatch.chdir(tmp_path)
    tmir = _tmir(n_classes=4, n_samples=40, seed=13)
    tmir.to_yaml(tmp_path / "model.yaml")
    payload = _payload(tmp_path, tmir)
    payload.update(model_path="model.yaml", output_dir="out")
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(payload))
    runner = CliRunner()
    args = ["--backend", "redress_lookahead", "--config", "cfg.yaml"]
    assert runner.invoke(main, ["generate", *args]).exit_code == 0
    sim = runner.invoke(main, ["simulate", *args])
    assert sim.exit_code == 0, sim.output
    assert "[verilator]      PASS" in sim.output and "0 failed" in sim.output


def test_cli_list_backends_and_export(tmp_path):
    runner = CliRunner()
    assert "redress_lookahead" in runner.invoke(main, ["list-backends"]).output
    tmir = _tmir()
    path = tmp_path / "m.yaml"
    tmir.to_yaml(path)
    out = tmp_path / "m.json"
    res = runner.invoke(main, ["export-model-json", "--backend", "redress_lookahead",
                               "--model", str(path), "-o", str(out)])
    assert res.exit_code == 0, res.output
    assert out.exists()


def test_example_config_validates(tmp_path):
    from pathlib import Path

    raw = yaml.safe_load(Path("examples/redress_lookahead.yaml").read_text())
    model = tmp_path / "model.yaml"
    _tmir().to_yaml(model)
    raw["model_path"] = str(model)  # the example's placeholder path doesn't exist here
    cfg = RedressLookaheadConfig.model_validate(raw)
    assert cfg.e_lookahead_depth == 1 and cfg.cc_lookahead
