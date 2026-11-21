"""Tests for matador.backends.hardwired — the HCB-streaming hardwired TM
accelerator, including scoring_mode="templated" (adder_tree.v + the shared
argmax_rt.v from matador/backends/common/src/) as an alternative to the
default "full_hardwired" (bespoke Python-generated tree + inline argmax).
"""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from matador.backends.hardwired.config import HardwiredAcceleratorConfig
from matador.backends.hardwired.rtl import HardwiredBackend
from matador.backends.registry import config_class_for, get, list_backends
from matador.inference.reference import predict
from matador.ir.tm_ir import Architecture, EvalVector, Hyperparameters, Representation, TMIR, Verification

INCLUDE = 200
EXCLUDE = 50

_HAVE_IVERILOG = shutil.which("iverilog") is not None
_HAVE_VERILATOR = shutil.which("verilator") is not None


def _make_tmir(n_features=15, n_classes=2, n_clauses_pc=8, seed=0, n_samples=16) -> TMIR:
    n_literals = 2 * n_features
    rng = np.random.default_rng(seed)
    ta = np.where(
        rng.random((n_classes, n_clauses_pc, n_literals)) < 0.25, INCLUDE, EXCLUDE
    ).astype(np.int32)

    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=n_literals,
            n_classes=n_classes, n_clauses_per_class=n_clauses_pc,
            n_clauses_total=n_classes * n_clauses_pc,
            clause_organization="per_class", threshold=0,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
    )
    X = (rng.random((n_samples, n_features)) < 0.5).astype(np.uint8)
    preds, scores = predict(tmir, X)
    tmir.verification = Verification(test_vectors=[
        EvalVector(input=X[i].tolist(), expected_class=int(preds[i]), expected_scores=scores[i].tolist())
        for i in range(len(X))
    ])
    return tmir


def _make_cfg(tmp_path, tmir=None, **overrides) -> HardwiredAcceleratorConfig:
    tmir = tmir if tmir is not None else _make_tmir()
    tmir_path = tmp_path / "model.yaml"
    tmir.to_yaml(tmir_path)
    payload = {"model_path": str(tmir_path), "output_dir": str(tmp_path)}
    payload.update(overrides)
    return HardwiredAcceleratorConfig.model_validate(payload)


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
# Registry / config
# ---------------------------------------------------------------------------

def test_registered_in_backend_list():
    assert "vanilla_hardwired" in list_backends()


def test_registry_resolves_backend_and_config_class():
    assert get("vanilla_hardwired")().name == "vanilla_hardwired"
    assert config_class_for("vanilla_hardwired") is HardwiredAcceleratorConfig


def test_scoring_mode_defaults_to_full_hardwired(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.scoring_mode == "full_hardwired"


def test_scoring_mode_rejects_unknown_value(tmp_path):
    with pytest.raises(ValueError):
        _make_cfg(tmp_path, scoring_mode="bogus")


def test_n_test_vectors_defaults_to_8(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.n_test_vectors == 8


def test_n_test_vectors_rejects_non_positive(tmp_path):
    with pytest.raises(ValueError):
        _make_cfg(tmp_path, n_test_vectors=0)


# ---------------------------------------------------------------------------
# n_test_vectors — embedded vector count actually follows the config field
# ---------------------------------------------------------------------------

def test_n_test_vectors_controls_embedded_count(tmp_path):
    # _make_tmir's default n_samples=16 means 16 real vectors are available;
    # requesting fewer than that must actually embed fewer, not silently
    # fall back to a hardcoded 8.
    tmir = _make_tmir(n_samples=16)
    cfg = _make_cfg(tmp_path, tmir=tmir, n_test_vectors=5)
    HardwiredBackend().generate(tmir, cfg)
    tb_text = (tmp_path / "RTL" / "tb" / "tb_hw_system.v").read_text()
    assert "parameter N_VEC    = 5;" in tb_text
    assert "vec5" not in tb_text   # vectors 5..15 must not be embedded
    assert "vec4" in tb_text


def test_n_test_vectors_caps_at_available_pool(tmp_path):
    # Requesting more than the TMIR actually has must clip to what's
    # available, not emit empty/garbage vector slots.
    tmir = _make_tmir(n_samples=4)
    cfg = _make_cfg(tmp_path, tmir=tmir, n_test_vectors=100)
    HardwiredBackend().generate(tmir, cfg)
    tb_text = (tmp_path / "RTL" / "tb" / "tb_hw_system.v").read_text()
    assert "parameter N_VEC    = 4;" in tb_text


# ---------------------------------------------------------------------------
# generate() — file layout per mode
# ---------------------------------------------------------------------------

def test_generate_full_hardwired_writes_expected_files(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, scoring_mode="full_hardwired")
    artifacts = HardwiredBackend().generate(tmir, cfg)
    src_names = {p.name for p in artifacts.sources}
    assert src_names == {"axis_fifo.v", "hcb_blocks.v", "hw_score.v", "hw_tm_accelerator.v"}


def test_generate_templated_writes_expected_files(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, scoring_mode="templated")
    artifacts = HardwiredBackend().generate(tmir, cfg)
    src_names = {p.name for p in artifacts.sources}
    assert src_names == {
        "axis_fifo.v", "hcb_blocks.v", "hw_score.v", "hw_tm_accelerator.v",
        "adder_tree.v", "argmax_rt.v",
    }
    # argmax_rt.v must be byte-identical to the shared canonical copy.
    from matador.backends.hardwired.rtl import _COMMON_DIR
    shared = (_COMMON_DIR / "argmax_rt.v").read_text()
    copied = [p for p in artifacts.sources if p.name == "argmax_rt.v"][0].read_text()
    assert shared == copied


def test_templated_hw_score_has_no_hand_generated_comparator_chain(tmp_path):
    """The whole point of templated mode: no per-model Python-generated
    linear-scan argmax text — it instantiates argmax_rt instead."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, scoring_mode="templated")
    artifacts = HardwiredBackend().generate(tmir, cfg)
    text = [p for p in artifacts.sources if p.name == "hw_score.v"][0].read_text()
    assert "mx_score_" not in text  # full_hardwired's hand-rolled comparator chain
    assert "argmax_rt" in text
    assert "adder_tree" in text


# ---------------------------------------------------------------------------
# RTL simulation (real iverilog, skipped if unavailable)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scoring_mode", ["full_hardwired", "templated"])
@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_rtl_simulation_matches_reference(tmp_path, scoring_mode):
    # _gen_tb_system() caps the system testbench at 8 embedded vectors
    # regardless of how many verification.test_vectors the model carries.
    tmir = _make_tmir(n_features=20, n_classes=3, n_clauses_pc=6, seed=3, n_samples=12)
    cfg = _make_cfg(tmp_path, tmir=tmir, scoring_mode=scoring_mode)
    backend = HardwiredBackend()
    artifacts = backend.generate(tmir, cfg)

    sim_dir = artifacts.rtl_dir / "sim"
    tb_path = [p for p in artifacts.testbenches if p.name == "tb_hw_system.v"][0]
    output = _run_iverilog(artifacts.sources, tb_path, sim_dir)
    assert "SUMMARY: 8 passed, 0 failed" in output, output
    assert "FAIL" not in output


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_full_hardwired_and_templated_predict_identically(tmp_path):
    """Same math, two implementations — the real correctness bar for
    scoring_mode, not just 'each one runs on its own'."""
    tmir = _make_tmir(n_features=15, n_classes=4, n_clauses_pc=10, seed=7, n_samples=20)
    backend = HardwiredBackend()

    preds_by_mode = {}
    for mode in ("full_hardwired", "templated"):
        out_dir = tmp_path / mode
        out_dir.mkdir()
        cfg = _make_cfg(out_dir, tmir=tmir, scoring_mode=mode)
        artifacts = backend.generate(tmir, cfg)
        sim_dir = artifacts.rtl_dir / "sim"
        tb_path = [p for p in artifacts.testbenches if p.name == "tb_hw_system.v"][0]
        output = _run_iverilog(artifacts.sources, tb_path, sim_dir)
        assert "FAIL" not in output, output
        preds_by_mode[mode] = [
            line for line in output.splitlines() if line.startswith("vec ")
        ]

    assert preds_by_mode["full_hardwired"] == preds_by_mode["templated"]


@pytest.mark.skipif(not _HAVE_VERILATOR, reason="verilator not installed")
def test_templated_verilator_harness_builds_and_passes(tmp_path):
    # n_samples=10 available, default n_test_vectors=8 caps what gets
    # embedded -- this harness used to embed the model's ENTIRE
    # verification set unconditionally (the one inconsistency across an
    # otherwise-uniform set of testbenches); it now shares the exact same
    # cap as tb_hw_system.v, so 8 (not 10) is the correct expectation.
    tmir = _make_tmir(n_features=15, n_classes=2, n_clauses_pc=8, seed=9, n_samples=10)
    cfg = _make_cfg(tmp_path, tmir=tmir, scoring_mode="templated")
    artifacts = HardwiredBackend().generate(tmir, cfg)

    ver_dir = artifacts.rtl_dir / "sim" / "verilator"
    build_cp = subprocess.run(["make", "-C", str(ver_dir), "all"], capture_output=True, text=True)
    assert build_cp.returncode == 0, build_cp.stdout + build_cp.stderr
    bin_path = ver_dir / "obj_dir" / "Vhw_tm_accelerator"
    assert bin_path.exists()
    run_cp = subprocess.run([str(bin_path)], capture_output=True, text=True, cwd=str(ver_dir))
    output = run_cp.stdout + run_cp.stderr
    assert "ALL 8 TESTS PASSED" in output, output


def test_n_test_vectors_controls_verilator_embedded_count_too(tmp_path):
    # The same config field that controls tb_hw_system.v's embedded count
    # must control tb_top.cpp's too -- one knob, both harnesses, not two
    # independently-drifting caps.
    tmir = _make_tmir(n_features=15, n_classes=2, n_clauses_pc=8, seed=3, n_samples=10)
    cfg = _make_cfg(tmp_path, tmir=tmir, scoring_mode="templated", n_test_vectors=4)
    artifacts = HardwiredBackend().generate(tmir, cfg)

    cpp_text = (artifacts.rtl_dir / "sim" / "verilator" / "tb_top.cpp").read_text()
    assert "static const int N_TESTS    = 4;" in cpp_text

    ver_dir = artifacts.rtl_dir / "sim" / "verilator"
    build_cp = subprocess.run(["make", "-C", str(ver_dir), "all"], capture_output=True, text=True)
    assert build_cp.returncode == 0, build_cp.stdout + build_cp.stderr
    bin_path = ver_dir / "obj_dir" / "Vhw_tm_accelerator"
    run_cp = subprocess.run([str(bin_path)], capture_output=True, text=True, cwd=str(ver_dir))
    output = run_cp.stdout + run_cp.stderr
    assert "ALL 4 TESTS PASSED" in output, output
