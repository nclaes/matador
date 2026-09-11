"""Tests for matador.backends.redress — the MUSTANG Single-Core RTL backend
(config, emulator, RTL generation, and reprogram-suite). codec.py's own
Algorithm 11/12 correctness is covered separately in test_redress_codec.py;
these tests exercise the layer built on top of it: capacity checking,
RTL rendering, and (where iverilog is available) real simulation — including
the multi-batch "load once, infer many" path, which surfaced a real
AXI-Stream testbench-timing bug during development (see redress_accel.v's
and _render_model_testbench()'s header comments)."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from matador.backends.base import ReprogramStep
from matador.backends.redress import codec
from matador.backends.redress.config import RedressAcceleratorConfig
from matador.backends.redress.emulator import RedressEmulator
from matador.backends.redress.rtl import RedressBackend
from matador.backends.registry import config_class_for, get, list_backends
from matador.inference.reference import predict
from matador.ir.tm_ir import Architecture, EvalVector, Hyperparameters, Representation, TMIR, Verification

_HAVE_IVERILOG = shutil.which("iverilog") is not None
_HAVE_VERILATOR = shutil.which("verilator") is not None


def _make_tmir(n_features=15, n_classes=2, cpc=8, density=0.3, n_samples=8, seed=0) -> TMIR:
    """Every class gets at least one guaranteed include (MUSTANG can't
    represent a fully-empty class — see codec.MustangEmptyClassError)."""
    rng = np.random.default_rng(seed)
    inc = rng.random((n_classes, cpc, 2 * n_features)) < density
    for c in range(n_classes):
        inc[c, 0, 0] = True  # guarantee class c is non-empty

    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=2 * n_features,
            n_classes=n_classes, n_clauses_per_class=cpc,
            n_clauses_total=n_classes * cpc,
            clause_organization="per_class", threshold=0,
        ),
        hyperparameters=Hyperparameters(s=1.0),
        representation=Representation(includes=inc),
    )

    X = (rng.random((n_samples, n_features)) < 0.5).astype(np.uint8)
    preds, scores = predict(tmir, X)
    tmir.verification = Verification(test_vectors=[
        EvalVector(input=X[i].tolist(), expected_class=int(preds[i]), expected_scores=scores[i].tolist())
        for i in range(len(X))
    ])
    return tmir


def _make_cfg(tmp_path, tmir=None, **overrides) -> RedressAcceleratorConfig:
    tmir = tmir if tmir is not None else _make_tmir()
    tmir_path = tmp_path / "model.yaml"
    tmir.to_yaml(tmir_path)
    payload = {
        "model_path": str(tmir_path),
        "output_dir": str(tmp_path),
        "target_fpga": "xc7z020",
        "max_includes": 4096,
        "max_features": 256,
        "max_classes": 16,
        "batch_width": 32,
    }
    payload.update(overrides)
    return RedressAcceleratorConfig.model_validate(payload)


def _run_iverilog(sources, tb_path, sim_dir):
    out_bin = sim_dir / tb_path.stem
    compile_cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(out_bin)]
        + [str(s) for s in sources] + [str(tb_path)],
        capture_output=True, text=True,
    )
    assert compile_cp.returncode == 0, compile_cp.stderr
    run_cp = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(sim_dir))
    return run_cp.stdout + run_cp.stderr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registered_in_backend_list():
    assert "redress" in list_backends()


def test_registry_resolves_backend_and_config_class():
    backend_cls = get("redress")
    assert backend_cls().name == "redress"
    assert config_class_for("redress") is RedressAcceleratorConfig


def test_supports_reprogramming_is_true():
    assert RedressBackend().supports_reprogramming is True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_rejects_missing_model_path(tmp_path):
    with pytest.raises(ValueError):
        RedressAcceleratorConfig.model_validate({
            "model_path": str(tmp_path / "nope.yaml"),
            "output_dir": str(tmp_path),
            "target_fpga": "xc7z020",
            "max_includes": 4096, "max_features": 256, "max_classes": 16,
        })


def test_config_defaults(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.axis_data_width == 32
    assert cfg.batch_width == 32
    assert cfg.instr_mem_bits == 4096 * 16
    assert cfg.feature_fifo_bits == 256 * 32
    assert cfg.required_bram_bits == cfg.instr_mem_bits + cfg.feature_fifo_bits


def test_config_rejects_capacity_exceeding_device_budget(tmp_path):
    with pytest.raises(ValueError):
        _make_cfg(tmp_path, max_includes=1_000_000, target_fpga="xc7z020")


def test_config_rejects_max_features_beyond_mustang_offset_ceiling(tmp_path):
    with pytest.raises(ValueError):
        _make_cfg(tmp_path, max_features=codec.MUSTANG_MAX_FEATURES + 1)


def test_config_accepts_max_features_at_mustang_offset_ceiling(tmp_path):
    cfg = _make_cfg(tmp_path, max_includes=1000, max_features=codec.MUSTANG_MAX_FEATURES,
                     target_fpga="xcku040")
    assert cfg.max_features == codec.MUSTANG_MAX_FEATURES


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------

def test_emulator_matches_embedded_test_vectors(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    emu = RedressEmulator(tmir, cfg)
    for tv in tmir.verification.test_vectors:
        trace = emu.run(tv.input)
        assert trace.predicted_class == tv.expected_class
        assert trace.scores == tv.expected_scores


def test_emulator_matches_reference_on_fresh_random_vectors(tmp_path):
    tmir = _make_tmir(n_features=20, n_classes=3, cpc=6, density=0.15, seed=3)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    emu = RedressEmulator(tmir, cfg)

    rng = np.random.default_rng(99)
    X = (rng.random((25, 20)) < 0.5).astype(np.uint8)
    ref_pred, _ = predict(tmir, X)
    for i in range(len(X)):
        trace = emu.run(X[i].tolist())
        assert trace.predicted_class == int(ref_pred[i])


def test_emulator_unit_testbenches_returns_empty(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    emu = RedressEmulator(tmir, cfg)
    assert emu.unit_testbenches([]) == {}


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------

def test_generate_writes_expected_files(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()

    artifacts = backend.generate(tmir, cfg)

    assert artifacts.rtl_dir.exists()
    src_names = {p.name for p in artifacts.sources}
    assert src_names == {"redress_bram.v", "score_acc_rt.v", "argmax_rt.v", "redress_accel.v"}
    tb_names = {p.name for p in artifacts.testbenches}
    assert "tb_redress_model.v" in tb_names
    sim_names = {p.name for p in artifacts.sim_scripts}
    assert {"model_stimulus.memh", "model_expected.memh", "run_iverilog.sh",
            "test_vectors.txt", "tb_redress_model.gtkw", "provenance.json", "README.md"} <= sim_names
    assert (artifacts.rtl_dir / "sim" / "verilator" / "Makefile").exists()
    assert (artifacts.rtl_dir / "sim" / "verilator" / "tb_top.cpp").exists()


def test_generate_bakes_capacity_into_top_module(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, max_includes=1024, max_features=64, max_classes=4)
    backend = RedressBackend()
    artifacts = backend.generate(tmir, cfg)

    top_text = (artifacts.rtl_dir / "src" / "redress_accel.v").read_text()
    assert "parameter MAX_CLASSES     = 4" in top_text
    assert "parameter MAX_INCLUDES    = 1024" in top_text
    assert "parameter INSTR_ADDR_W    = 10" in top_text  # clog2(1024)
    assert "parameter MAX_FEATURES    = 64" in top_text
    assert "parameter FEAT_ADDR_W     = 6" in top_text   # clog2(64)
    assert "parameter BATCH_WIDTH     = 32" in top_text
    assert "parameter OUT_IDX_W       = 5" in top_text   # clog2(32)


def test_generate_rejects_model_exceeding_max_includes(tmp_path):
    tmir = _make_tmir(n_features=40, n_classes=2, cpc=20, density=0.9, seed=5)
    cfg = _make_cfg(tmp_path, tmir=tmir, max_includes=4)  # far too small
    backend = RedressBackend()
    with pytest.raises(ValueError, match="max_includes"):
        backend.generate(tmir, cfg)


def test_generate_rejects_empty_class_model(tmp_path):
    """A model with a fully-empty class is a real MUSTANG format limitation
    (codec.MustangEmptyClassError), not an RTL bug — generate() must
    surface it clearly rather than silently emitting a broken stream."""
    n_features, n_classes, cpc = 8, 2, 4
    inc = np.zeros((n_classes, cpc, 2 * n_features), dtype=bool)
    inc[0, 0, 0] = True  # class 0 non-empty; class 1 entirely empty
    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=2 * n_features,
            n_classes=n_classes, n_clauses_per_class=cpc,
            n_clauses_total=n_classes * cpc,
            clause_organization="per_class", threshold=0,
        ),
        hyperparameters=Hyperparameters(s=1.0),
        representation=Representation(includes=inc),
    )
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    with pytest.raises(codec.MustangEmptyClassError):
        backend.generate(tmir, cfg)


def test_generate_rejects_model_exceeding_max_classes(tmp_path):
    tmir = _make_tmir(n_classes=5)
    cfg = _make_cfg(tmp_path, tmir=tmir, max_classes=2)
    backend = RedressBackend()
    with pytest.raises(ValueError, match="max_classes"):
        backend.generate(tmir, cfg)


def test_resource_estimate_reports_bram_bits(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    est = backend.resource_estimate(tmir, cfg)
    assert est.bram_bits == cfg.required_bram_bits
    assert "Instruction Memory" in est.notes


# ---------------------------------------------------------------------------
# RTL simulation (real iverilog, skipped if unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_rtl_simulation_single_batch_passes(tmp_path):
    """n_samples < batch_width -- exactly one (padded) batch."""
    tmir = _make_tmir(n_features=15, n_classes=2, cpc=10, density=0.3, n_samples=8, seed=1)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    artifacts = backend.generate(tmir, cfg)

    sim_dir = artifacts.rtl_dir / "sim"
    tb_path = [p for p in artifacts.testbenches if p.name == "tb_redress_model.v"][0]
    output = _run_iverilog(artifacts.sources, tb_path, sim_dir)
    assert "ALL " in output and "TESTS PASSED" in output, output
    assert "FAIL" not in output


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_rtl_simulation_multi_batch_passes(tmp_path):
    """n_samples spans several batch_width-sized batches, all against ONE
    Instruction-stream load ('load once, infer many') -- this is the exact
    scenario that surfaced a real AXI-Stream stimulus-driver bug during
    development (a procedural blocking-assignment loop could re-present a
    stale beat across the idle gap between batches; fixed by driving
    stimulus from a single synchronous always block instead -- see
    RedressBackend._render_model_testbench()'s header comment)."""
    tmir = _make_tmir(n_features=15, n_classes=2, cpc=10, density=0.3, n_samples=97, seed=2)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    artifacts = backend.generate(tmir, cfg)

    n_batches = -(-97 // cfg.batch_width)
    assert n_batches >= 3  # actually exercises multiple batches, not a fluke

    sim_dir = artifacts.rtl_dir / "sim"
    tb_path = [p for p in artifacts.testbenches if p.name == "tb_redress_model.v"][0]
    output = _run_iverilog(artifacts.sources, tb_path, sim_dir)
    assert f"ALL {n_batches * cfg.batch_width} TESTS PASSED" in output, output
    assert "FAIL" not in output


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_rtl_simulation_handles_no_embedded_vectors(tmp_path):
    """A model with no verification.test_vectors still generates a valid
    (load-only) testbench rather than an empty/degenerate one."""
    tmir = _make_tmir(n_samples=8)
    tmir.verification = None
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    artifacts = backend.generate(tmir, cfg)

    sim_dir = artifacts.rtl_dir / "sim"
    tb_path = [p for p in artifacts.testbenches if p.name == "tb_redress_model.v"][0]
    output = _run_iverilog(artifacts.sources, tb_path, sim_dir)
    assert "no test vectors embedded" in output
    assert "FAIL" not in output


def test_generate_exports_verilator_harness(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    artifacts = backend.generate(tmir, cfg)

    verilator_dir = artifacts.rtl_dir / "sim" / "verilator"
    assert (verilator_dir / "Makefile").exists()
    text = (verilator_dir / "tb_top.cpp").read_text()
    assert "Vredress_accel.h" in text
    # Explanatory comments may mention matador; what matters is that it has
    # no #include of anything matador-specific, since it must compile
    # standalone against just the Verilator-generated model.
    includes = [line for line in text.splitlines() if line.strip().startswith("#include")]
    assert includes, "expected at least the Verilator/stdlib #include lines"
    assert not any("matador" in line.lower() for line in includes)


@pytest.mark.skipif(not _HAVE_VERILATOR, reason="verilator not installed")
def test_verilator_harness_passes_against_model_specific_vectors(tmp_path):
    """Same correctness contract as the iverilog testbench, second
    independent simulator, real 'load once, infer many' multi-batch run."""
    tmir = _make_tmir(n_features=15, n_classes=2, cpc=10, density=0.3, n_samples=70, seed=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    artifacts = backend.generate(tmir, cfg)

    verilator_dir = artifacts.rtl_dir / "sim" / "verilator"
    build_cp = subprocess.run(["make", "-C", str(verilator_dir), "all"], capture_output=True, text=True)
    assert build_cp.returncode == 0, build_cp.stdout + build_cp.stderr

    bin_path = verilator_dir / "obj_dir" / "Vredress_accel"
    assert bin_path.exists()
    run_cp = subprocess.run([str(bin_path)], capture_output=True, text=True, cwd=str(verilator_dir))
    output = run_cp.stdout + run_cp.stderr
    n_batches = -(-70 // cfg.batch_width)
    assert f"ALL TESTS PASSED ({n_batches * cfg.batch_width} beats checked)" in output, output
    assert "FAIL" not in output


# ---------------------------------------------------------------------------
# build_reprogram_suite()
# ---------------------------------------------------------------------------

def test_build_reprogram_suite_requires_prior_generate(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    step = ReprogramStep(tmir_path=cfg.model_path)
    with pytest.raises(FileNotFoundError):
        backend.build_reprogram_suite(tmp_path / "RTL", [step], cfg)


def test_build_reprogram_suite_rejects_empty_steps(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    backend.generate(tmir, cfg)
    with pytest.raises(ValueError):
        backend.build_reprogram_suite(cfg.output_dir / "RTL", [], cfg)


def test_build_reprogram_suite_rejects_model_exceeding_capacity(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = RedressBackend()
    backend.generate(tmir, cfg)

    big_tmir = _make_tmir(n_features=40, n_classes=2, cpc=20, density=0.9, seed=6)
    big_path = tmp_path / "big.yaml"
    big_tmir.to_yaml(big_path)
    small_cfg = _make_cfg(tmp_path, tmir=tmir, max_includes=4)
    step = ReprogramStep(tmir_path=big_path)
    with pytest.raises(ValueError):
        backend.build_reprogram_suite(cfg.output_dir / "RTL", [step], small_cfg)


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_reprogram_suite_rtl_passes_across_two_models(tmp_path):
    tmir_a = _make_tmir(n_features=15, n_classes=2, cpc=8, density=0.3, n_samples=40, seed=10)
    tmir_b = _make_tmir(n_features=15, n_classes=2, cpc=8, density=0.25, n_samples=50, seed=11)
    path_a = tmp_path / "a.yaml"
    path_b = tmp_path / "b.yaml"
    tmir_a.to_yaml(path_a)
    tmir_b.to_yaml(path_b)

    cfg = _make_cfg(tmp_path, tmir=tmir_a)
    backend = RedressBackend()
    backend.generate(tmir_a, cfg)

    steps = [
        ReprogramStep(tmir_path=path_a, name="model_a"),
        ReprogramStep(tmir_path=path_b, name="model_b"),
    ]
    artifacts = backend.build_reprogram_suite(cfg.output_dir / "RTL", steps, cfg)

    manifest = [p for p in artifacts.sim_scripts if p.name == "reprogram_manifest.txt"][0]
    assert "model_a" in manifest.read_text()
    assert "model_b" in manifest.read_text()

    sim_dir = cfg.output_dir / "RTL" / "sim"
    tb_path = artifacts.testbenches[0]
    sources = [cfg.output_dir / "RTL" / "src" / f for f in
               ("redress_bram.v", "score_acc_rt.v", "argmax_rt.v", "redress_accel.v")]
    output = _run_iverilog(sources, tb_path, sim_dir)
    assert "ALL " in output and "TESTS PASSED" in output, output
    assert "FAIL" not in output
