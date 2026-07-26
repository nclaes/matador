"""Tests for matador.backends.gp_tiled — the runtime-reprogrammable
GP-tiled backend (vendored from GP_TM_Inference_Accelerator)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from matador.backends.base import ReprogramStep
from matador.backends.gp_tiled.config import GPTiledAcceleratorConfig
from matador.backends.gp_tiled.emulator import GPTiledEmulator
from matador.backends.gp_tiled.reprogram_config import ReprogramStepConfig, ReprogramSuiteConfig
from matador.backends.gp_tiled.rtl import GPTiledBackend
from matador.backends.gp_tiled.tmir_bridge import (
    GPCapacityError,
    check_capacity,
    tmir_to_tmmodel,
)
from matador.backends.registry import config_class_for, get, list_backends
from matador.ir.tm_ir import (
    Architecture, EvalVector, Hyperparameters, Provenance, Representation, TMIR, Verification,
)
from matador.inference.reference import predict

INCLUDE = 200
EXCLUDE = 50

_HAVE_IVERILOG = shutil.which("iverilog") is not None
_HAVE_VERILATOR = shutil.which("verilator") is not None


def _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4, with_vectors=True) -> TMIR:
    n_literals = 2 * n_features
    ta = np.full((n_classes, n_clauses_pc, n_literals), EXCLUDE, dtype=np.int32)
    ta[0, 0, 0] = INCLUDE          # Class 0, clause 0: include f0 (pos)
    ta[0, 1, 1] = INCLUDE          # Class 0, clause 1: include f1 (pos)
    ta[1, 0, 0] = INCLUDE          # Class 1, clause 0: include f0 AND f1 (pos)
    ta[1, 0, 1] = INCLUDE
    ta[1, 2, n_features] = INCLUDE  # Class 1, clause 2: include NOT-f0 (neg)

    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=n_literals,
            n_classes=n_classes, n_clauses_per_class=n_clauses_pc,
            n_clauses_total=n_classes * n_clauses_pc,
            clause_organization="per_class", threshold=threshold,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
    )

    if with_vectors:
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(8, n_features)).astype(np.uint8)
        preds, scores = predict(tmir, X)
        tmir.verification = Verification(test_vectors=[
            EvalVector(input=X[i].tolist(), expected_class=int(preds[i]), expected_scores=scores[i].tolist())
            for i in range(len(X))
        ])
    return tmir


# Defaults chosen to comfortably fit the smallest known device (xc7z020) at
# feat_slice=clause_slice=32: required_bits = 2*512*256 = 262,144 (<< budget).
def _make_cfg(tmp_path, tmir=None, **overrides) -> GPTiledAcceleratorConfig:
    tmir = tmir or _make_tmir()
    tmir_path = tmp_path / "model.yaml"
    tmir.to_yaml(tmir_path)
    payload = {
        "model_path": str(tmir_path),
        "output_dir": str(tmp_path),
        "fifo_depth": 16,
        "target_fpga": "xc7z020",
        "max_features": 512,
        "max_clauses_total": 256,
    }
    payload.update(overrides)
    return GPTiledAcceleratorConfig.model_validate(payload)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registered_in_backend_list():
    assert "vanilla_gp_tiled" in list_backends()


def test_registry_resolves_backend_and_config_class():
    cls = get("vanilla_gp_tiled")
    assert cls is GPTiledBackend
    assert config_class_for("vanilla_gp_tiled") is GPTiledAcceleratorConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_config_rejects_missing_model_path(tmp_path):
    with pytest.raises(Exception):
        GPTiledAcceleratorConfig.model_validate({
            "model_path": str(tmp_path / "does_not_exist.yaml"),
            "output_dir": str(tmp_path),
            "target_fpga": "xc7z020",
            "max_features": 512,
            "max_clauses_total": 256,
        })


def test_config_defaults(tmp_path):
    cfg = _make_cfg(tmp_path)
    assert cfg.feat_slice == 32
    assert cfg.clause_slice == 32
    assert cfg.max_classes == 32


def test_config_derives_slice_counts_and_bram_bits(tmp_path):
    cfg = _make_cfg(tmp_path, max_features=2000, max_clauses_total=2000,
                    max_classes=20, target_fpga="xcku040")
    assert cfg.max_feat_slices == 63       # ceil(2000/32)
    assert cfg.max_clause_slices == 63     # ceil(2000/32)
    assert cfg.n_tiles_max == 63 * 63
    assert cfg.required_bram_bits == 2 * (63 * 32) * (63 * 32)


def test_config_rejects_capacity_exceeding_device_budget(tmp_path):
    """2000x2000 needs ~8.13 Mbit; xc7z020 only has 4.9 Mbit total BRAM."""
    with pytest.raises(Exception, match="exceeds the .* budget"):
        _make_cfg(tmp_path, max_features=2000, max_clauses_total=2000,
                 max_classes=20, target_fpga="xc7z020")


def test_config_accepts_capacity_fitting_bigger_device(tmp_path):
    cfg = _make_cfg(tmp_path, max_features=2000, max_clauses_total=2000,
                    max_classes=20, target_fpga="xcku040")
    assert cfg.required_bram_bits < cfg.required_bram_bits + 1  # sanity: no raise above


def test_config_rejects_unknown_device_without_override(tmp_path):
    with pytest.raises(Exception, match="Unknown target_fpga"):
        _make_cfg(tmp_path, target_fpga="some_custom_part")


def test_config_accepts_unknown_device_with_explicit_budget(tmp_path):
    cfg = _make_cfg(tmp_path, target_fpga="some_custom_part", bram_bits_budget=100_000_000)
    assert cfg.target_fpga == "some_custom_part"


def test_config_rejects_slice_count_beyond_protocol_ceiling(tmp_path):
    """word2's n_feat_slices header field is 8 bits (max 255)."""
    with pytest.raises(Exception, match="protocol's hard limit"):
        _make_cfg(tmp_path, feat_slice=1, max_features=1000,
                 target_fpga="xcku040", max_clauses_total=256)


# ---------------------------------------------------------------------------
# TMIR <-> TMModel bridge
# ---------------------------------------------------------------------------

def test_tmir_to_tmmodel_round_trips_geometry():
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    model = tmir_to_tmmodel(tmir)
    assert model.n_features == 8
    assert model.n_classes == 2
    assert model.clauses_per_class == 4
    assert model.threshold == 4
    assert len(model.include) == 8


def test_tmir_to_tmmodel_bit_convention_matches_reference():
    """Bit f of include[g] (positive literal) and bit n_features+f (negated)
    must match TMIR's own literal ordering, cross-checked via predict()."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4, with_vectors=False)
    model = tmir_to_tmmodel(tmir)

    rng = np.random.default_rng(1)
    X = rng.integers(0, 2, size=(20, 8)).astype(np.uint8)
    ref_preds, _ = predict(tmir, X)

    for i in range(len(X)):
        fv_int = sum(int(b) << f for f, b in enumerate(X[i]))
        pred, _ = model.infer(fv_int)
        assert pred == ref_preds[i]


def test_tmir_to_tmmodel_rejects_coalesced():
    tmir = _make_tmir()
    tmir.architecture.clause_organization = "coalesced"
    with pytest.raises(GPCapacityError):
        tmir_to_tmmodel(tmir)


def test_check_capacity_rejects_model_too_big(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir, max_classes=1)
    with pytest.raises(GPCapacityError):
        check_capacity(tmir, cfg)


def test_check_capacity_accepts_model_that_fits(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    check_capacity(tmir, cfg)  # should not raise


# ---------------------------------------------------------------------------
# Emulator
# ---------------------------------------------------------------------------

def test_emulator_matches_embedded_test_vectors(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    emul = GPTiledEmulator(tmir, cfg)

    for tv in tmir.verification.test_vectors:
        trace = emul.run(tv.input)
        assert trace.predicted_class == tv.expected_class


def test_emulator_rejects_wrong_feature_count(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    emul = GPTiledEmulator(tmir, cfg)
    with pytest.raises(ValueError):
        emul.run([0, 1])


def test_emulator_unit_testbenches_returns_vendored_sources(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    emul = GPTiledEmulator(tmir, cfg)
    tbs = emul.unit_testbenches([])
    assert "tb_system_gp" in tbs
    assert "module tb_system_gp" in tbs["tb_system_gp"]


# ---------------------------------------------------------------------------
# RTL generation
# ---------------------------------------------------------------------------

def test_generate_writes_expected_files(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()

    artifacts = backend.generate(tmir, cfg)

    assert artifacts.rtl_dir.exists()
    src_names = {p.name for p in artifacts.sources}
    assert src_names == {
        "axis_fifo.v", "clause_eval.v", "tile_mem.v",
        "score_acc_rt.v", "argmax_rt.v", "tm_accel_gp.v",
    }
    tb_names = {p.name for p in artifacts.testbenches}
    assert "tb_system_gp.v" in tb_names
    assert "tb_system_gp_model.v" in tb_names
    sim_names = {p.name for p in artifacts.sim_scripts}
    assert "model_stimulus.memh" in sim_names
    assert "model_expected.memh" in sim_names
    assert "run_iverilog.sh" in sim_names


def test_generate_bakes_capacity_into_top_module(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, max_classes=4, max_clauses_total=64,
                    max_features=64, feat_slice=32, clause_slice=32)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    top_text = (artifacts.rtl_dir / "src" / "tm_accel_gp.v").read_text()
    assert "parameter MAX_CLASSES       = 4" in top_text
    assert "parameter MAX_CLAUSES_TOTAL = 64" in top_text
    assert "parameter MAX_FEAT_SLICES   = 2" in top_text   # ceil(64/32)
    assert "parameter MAX_CLAUSE_SLICES = 2" in top_text   # ceil(64/32)
    assert "parameter FEAT_SLICE        = 32" in top_text
    assert "parameter CLAUSE_SLICE      = 32" in top_text


def test_generate_bakes_nondefault_slice_width(tmp_path):
    """feat_slice/clause_slice != 32 must flow through to TILE_WIDTH and
    WORD_CNT_W, not just the slice-count parameters."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, feat_slice=16, clause_slice=16,
                    max_features=64, max_clauses_total=64, max_classes=4)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    top_text = (artifacts.rtl_dir / "src" / "tm_accel_gp.v").read_text()
    assert "parameter FEAT_SLICE        = 16" in top_text
    assert "parameter CLAUSE_SLICE      = 16" in top_text
    # TILE_WIDTH = CLAUSE_SLICE * 2 * FEAT_SLICE = 16*2*16 = 512
    assert "parameter TILE_WIDTH        = 512" in top_text
    # WORD_CNT_W = clog2(512/32) = clog2(16) = 4
    assert "parameter WORD_CNT_W        = 4" in top_text


def test_generate_rejects_model_exceeding_capacity(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir, max_classes=1)
    backend = GPTiledBackend()
    with pytest.raises(GPCapacityError):
        backend.generate(tmir, cfg)


def test_generate_flattens_vectors_includes(tmp_path):
    """tb_system_gp.v's `include and $readmemh paths must be flattened to
    sim_dir-relative filenames (no nested vectors/ subdirectory) — see
    GPTiledBackend._flatten_includes."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    proto_tb = [p for p in artifacts.testbenches if p.name == "tb_system_gp.v"][0]
    text = proto_tb.read_text()
    assert "`include" not in text
    assert '$readmemh("vectors/' not in text
    assert '$readmemh("gp_stimulus.memh"' in text


def test_run_iverilog_sh_has_no_generate_time_absolute_paths(tmp_path):
    """run_iverilog.sh must resolve source/testbench paths relative to its
    own location at runtime (like waves.sh and the Verilator Makefile
    already do), not bake in absolute paths derived from output_dir at
    `matador generate` time -- those only happen to work if the bundle
    stays at the exact path it was generated at (e.g. inside a container
    where output_dir is a bind-mount point like /work; breaks the moment
    the RTL folder is copied elsewhere, including running outside the
    container it was generated in)."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    script = (artifacts.rtl_dir / "sim" / "run_iverilog.sh").read_text()
    assert str(tmp_path) not in script
    assert str(cfg.output_dir) not in script
    assert "SCRIPT_DIR=" in script


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_run_iverilog_sh_works_after_moving_the_rtl_bundle(tmp_path):
    """The concrete portability proof: generate at one path, delete that
    path entirely, run the script from a brand-new location -- must still
    pass. This is the exact bug report reproduced: run_iverilog.sh used to
    only work from the directory it was generated under."""
    tmir = _make_tmir()
    orig_root = tmp_path / "original_location"
    orig_root.mkdir()
    cfg = _make_cfg(orig_root, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    moved_root = tmp_path / "moved_elsewhere"
    moved_root.mkdir()
    moved_rtl = moved_root / "RTL"
    shutil.move(str(artifacts.rtl_dir), str(moved_rtl))
    shutil.rmtree(orig_root)  # the generate-time path no longer exists at all
    assert not orig_root.exists()

    cp = subprocess.run(
        ["bash", str(moved_rtl / "sim" / "run_iverilog.sh")],
        capture_output=True, text=True,
    )
    output = cp.stdout + cp.stderr
    assert cp.returncode == 0, output
    assert "FAIL" not in output, output
    assert "tb_system_gp_model: ALL TESTS PASSED" in output, output


# ---------------------------------------------------------------------------
# RTL-level boundary-crossing test — proves the widened FSM registers are
# actually correct in simulation, not just that Python-side config
# validation accepts large numbers. n_clauses_total=600 and n_tiles>255
# both exceed the OLD (pre-widening) register ceilings (511 and 255
# respectively); n_features is kept small so the LOAD payload (n_tiles*64
# beats) stays fast to simulate under iverilog.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_rtl_simulation_at_capacity_exceeding_old_ceilings(tmp_path):
    n_features = 448      # 14 feat_slices @ feat_slice=32
    n_classes = 3
    n_clauses_pc = 200    # n_clauses_total = 600 > old 511 ceiling
    n_clause_slices = 19  # ceil(600/32)
    # n_tiles = 14 * 19 = 266 > old 255 tile ceiling
    assert 14 * n_clause_slices > 255

    n_literals = 2 * n_features
    rng = np.random.default_rng(7)
    ta = np.full((n_classes, n_clauses_pc, n_literals), EXCLUDE, dtype=np.int32)
    # A handful of deterministic discriminative clauses per class, including
    # one at a high clause index (class 2, clause 150 -> global index
    # 2*200+150=550) to exercise indexing past the old 511-entry ceiling.
    ta[0, 0, 0] = INCLUDE
    ta[0, 0, 1] = INCLUDE
    ta[1, 0, 2] = INCLUDE
    ta[1, 0, n_features + 3] = INCLUDE
    ta[2, 150, 4] = INCLUDE
    ta[2, 150, 5] = INCLUDE

    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=n_literals,
            n_classes=n_classes, n_clauses_per_class=n_clauses_pc,
            n_clauses_total=n_classes * n_clauses_pc,
            clause_organization="per_class", threshold=4,
        ),
        hyperparameters=Hyperparameters(s=3.9, n_states=256),
        representation=Representation(ta_states=ta),
    )
    X = rng.integers(0, 2, size=(3, n_features)).astype(np.uint8)
    preds, scores = predict(tmir, X)
    tmir.verification = Verification(test_vectors=[
        EvalVector(input=X[i].tolist(), expected_class=int(preds[i]), expected_scores=scores[i].tolist())
        for i in range(len(X))
    ])

    cfg = _make_cfg(
        tmp_path, tmir=tmir,
        max_features=n_features, max_clauses_total=n_classes * n_clauses_pc,
        max_classes=n_classes, target_fpga="xc7z020",
    )
    assert cfg.n_tiles_max > 255
    assert cfg.max_clauses_total > 511

    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    src_dir = artifacts.rtl_dir / "src"
    tb_dir = artifacts.rtl_dir / "tb"
    model_tb = tb_dir / "tb_system_gp_model.v"
    sources = [
        src_dir / "axis_fifo.v", src_dir / "clause_eval.v", src_dir / "tile_mem.v",
        src_dir / "score_acc_rt.v", src_dir / "argmax_rt.v", src_dir / "tm_accel_gp.v",
    ]

    sim_dir = artifacts.rtl_dir / "sim"
    out_bin = sim_dir / "tb_system_gp_model"
    compile_cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(out_bin)]
        + [str(s) for s in sources] + [str(model_tb)],
        capture_output=True, text=True,
    )
    assert compile_cp.returncode == 0, compile_cp.stderr

    run_cp = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(sim_dir))
    output = run_cp.stdout + run_cp.stderr
    assert "FAIL" not in output, output
    assert "TIMEOUT" not in output, output
    assert "ALL TESTS PASSED" in output, output


# ---------------------------------------------------------------------------
# Exported standalone infrastructure: gen_vectors.py / tm_emulator.py /
# capacity.json / model_reference.json (try a NEW model without matador
# installed, against an already-synthesized bitstream) and provenance.json
# (software manifest of which trained model's weights are currently packed
# into model_stimulus.memh).
# ---------------------------------------------------------------------------

def test_generate_exports_standalone_tooling_and_manifests(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    sim_dir = artifacts.rtl_dir / "sim"
    for fname in ("gen_vectors.py", "tm_emulator.py", "capacity.json", "model_reference.json"):
        assert (sim_dir / fname).exists(), fname
    assert (artifacts.rtl_dir / "provenance.json").exists()

    sim_names = {p.name for p in artifacts.sim_scripts}
    assert {"gen_vectors.py", "tm_emulator.py", "capacity.json", "model_reference.json"} <= sim_names
    assert "provenance.json" in sim_names


def test_generated_capacity_json_matches_config(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, max_features=64, max_clauses_total=64, max_classes=4)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    cap = json.loads((artifacts.rtl_dir / "sim" / "capacity.json").read_text())
    assert cap["feat_slice"] == cfg.feat_slice
    assert cap["clause_slice"] == cfg.clause_slice
    assert cap["max_features"] == 64
    assert cap["max_clauses_total"] == 64
    assert cap["max_classes"] == 4
    assert cap["max_feat_slices"] == cfg.max_feat_slices
    assert cap["max_clause_slices"] == cfg.max_clause_slices
    assert cap["n_tiles_max"] == cfg.n_tiles_max


def test_generated_model_reference_json_matches_source_model(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    ref = json.loads((artifacts.rtl_dir / "sim" / "model_reference.json").read_text())
    assert ref["n_features"] == 8
    assert ref["n_classes"] == 2
    assert ref["clauses_per_class"] == 4
    assert ref["threshold"] == 4
    assert len(ref["include"]) == 8


def test_provenance_json_records_fingerprint_and_geometry(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir.provenance = Provenance(
        framework="tmu", framework_version="1.2.3", dataset_id="my_dataset",
        trained_at=datetime(2026, 7, 10, tzinfo=timezone.utc), epochs=50,
    )
    cfg = _make_cfg(tmp_path, tmir=tmir, target_fpga="xcku040")
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    prov = json.loads((artifacts.rtl_dir / "provenance.json").read_text())
    assert prov["model_fingerprint"] == tmir.fingerprint()
    assert prov["architecture"]["n_features"] == 8
    assert prov["architecture"]["n_classes"] == 2
    assert prov["synthesized_capacity"]["target_fpga"] == "xcku040"
    assert prov["training_provenance"]["framework"] == "tmu"
    assert prov["training_provenance"]["dataset_id"] == "my_dataset"
    assert prov["training_provenance"]["epochs"] == 50


def test_provenance_json_handles_missing_provenance_gracefully(tmp_path):
    tmir = _make_tmir()
    assert tmir.provenance is None
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    prov = json.loads((artifacts.rtl_dir / "provenance.json").read_text())
    assert prov["training_provenance"] is None
    assert prov["model_fingerprint"]  # still present regardless
    assert "_note" in prov


def test_gen_vectors_py_has_no_matador_dependency(tmp_path):
    """The exported tool must run with just itself + the sibling
    tm_emulator.py — no matador install required."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"

    text = (sim_dir / "gen_vectors.py").read_text()
    assert "import matador" not in text
    assert "from matador" not in text


def test_gen_vectors_py_combined_produces_valid_vectors_for_a_new_model(tmp_path):
    """Exercises the actual exported tool as a subprocess (matching how a
    real standalone user would invoke it), building vectors for a model
    that was never seen by matador's own generate() call."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"

    new_model = {
        "name": "brand_new_model",
        "n_features": 8, "n_classes": 2, "clauses_per_class": 4, "threshold": 4,
        "include": [1, 0, 0, 0, 4, 0, 0, 0],
    }
    (sim_dir / "new_model.json").write_text(json.dumps(new_model))

    cp = subprocess.run(
        [sys.executable, "gen_vectors.py", "combined", "new_model.json", "--random",
         "-n", "4", "--seed", "1", "-o", "new_stim.memh", "--expected", "new_expected.memh"],
        cwd=str(sim_dir), capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stderr
    assert (sim_dir / "new_stim.memh").exists()
    assert (sim_dir / "new_expected.memh").exists()
    assert "wrote" in cp.stdout


def test_gen_vectors_py_rejects_model_exceeding_this_bundles_capacity(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir, max_classes=2, max_clauses_total=16, max_features=64)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"

    too_big_model = {
        "n_features": 8, "n_classes": 50, "clauses_per_class": 4, "threshold": 4,
        "include": [0] * 200,
    }
    (sim_dir / "too_big.json").write_text(json.dumps(too_big_model))

    cp = subprocess.run(
        [sys.executable, "gen_vectors.py", "load", "too_big.json", "-o", "x.memh"],
        cwd=str(sim_dir), capture_output=True, text=True,
    )
    assert cp.returncode != 0
    assert "error:" in cp.stderr


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_gen_vectors_py_testbench_output_passes_against_real_rtl(tmp_path):
    """End-to-end proof: a model built entirely standalone via gen_vectors.py
    (no matador involved in building the model or its vectors) actually
    classifies correctly when replayed against the already-synthesized RTL."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"
    src_dir = artifacts.rtl_dir / "src"

    new_model = {
        "name": "standalone_new_model",
        "n_features": 8, "n_classes": 2, "clauses_per_class": 4, "threshold": 4,
        "include": [1, 0, 0, 0, 4, 0, 0, 0],
    }
    (sim_dir / "standalone_model.json").write_text(json.dumps(new_model))

    gen_cp = subprocess.run(
        [sys.executable, "gen_vectors.py", "combined", "standalone_model.json", "--random",
         "-n", "4", "--seed", "5", "-o", "sa_stim.memh", "--expected", "sa_expected.memh",
         "--testbench", "tb_standalone.v"],
        cwd=str(sim_dir), capture_output=True, text=True,
    )
    assert gen_cp.returncode == 0, gen_cp.stderr
    assert (sim_dir / "tb_standalone.v").exists()

    sources = [
        src_dir / "axis_fifo.v", src_dir / "clause_eval.v", src_dir / "tile_mem.v",
        src_dir / "score_acc_rt.v", src_dir / "argmax_rt.v", src_dir / "tm_accel_gp.v",
    ]
    out_bin = sim_dir / "tb_standalone"
    compile_cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(out_bin)]
        + [str(s) for s in sources] + [str(sim_dir / "tb_standalone.v")],
        capture_output=True, text=True,
    )
    assert compile_cp.returncode == 0, compile_cp.stderr

    run_cp = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(sim_dir))
    output = run_cp.stdout + run_cp.stderr
    assert "FAIL" not in output, output
    assert "TIMEOUT" not in output, output
    assert "ALL TESTS PASSED" in output, output


# ---------------------------------------------------------------------------
# Verilator harness: generic memh-replayer, shipped alongside the iverilog
# path. matador.models.validator.validate_rtl (backend-agnostic) picks up
# sim/verilator/Makefile automatically once it exists.
# ---------------------------------------------------------------------------

def test_generate_exports_verilator_harness(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    verilator_dir = artifacts.rtl_dir / "sim" / "verilator"
    assert (verilator_dir / "Makefile").exists()
    assert (verilator_dir / "tb_top.cpp").exists()
    sim_names = {p.name for p in artifacts.sim_scripts}
    assert {"Makefile", "tb_top.cpp"} <= sim_names


def test_verilator_tb_top_has_no_matador_dependency(tmp_path):
    """The harness's own explanatory comments may mention matador (it
    documents which matador method copies the file in) -- what actually
    matters is that it has no #include of anything matador-specific, since
    it must compile standalone against just the Verilator-generated model."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    text = (artifacts.rtl_dir / "sim" / "verilator" / "tb_top.cpp").read_text()
    includes = [line for line in text.splitlines() if line.strip().startswith("#include")]
    assert includes, "expected at least the Verilator/stdlib #include lines"
    assert not any("matador" in line.lower() for line in includes)


@pytest.mark.skipif(not _HAVE_VERILATOR, reason="verilator not installed")
def test_verilator_harness_passes_against_model_specific_vectors(tmp_path):
    """make -C sim/verilator run must build + replay model_stimulus.memh /
    model_expected.memh (tb_top.cpp's defaults) and report PASS, exercising
    the real synthesized RTL through Verilator, not just iverilog."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    verilator_dir = artifacts.rtl_dir / "sim" / "verilator"

    cp = subprocess.run(
        ["make", "-C", str(verilator_dir), "run"],
        capture_output=True, text=True,
    )
    output = cp.stdout + cp.stderr
    assert cp.returncode == 0, output
    assert "FAIL" not in output, output
    assert "TIMEOUT" not in output, output
    assert "ALL TESTS PASSED" in output, output


@pytest.mark.skipif(not _HAVE_VERILATOR, reason="verilator not installed")
def test_verilator_harness_replays_a_gen_vectors_new_model_via_stim_exp_flags(tmp_path):
    """The same compiled Verilator binary must replay a brand-new model's
    vectors (built standalone via gen_vectors.py) when pointed at them with
    --stim/--exp — no rebuild needed, proving the harness is genuinely
    generic rather than hardcoded to the generation-time model."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"
    verilator_dir = sim_dir / "verilator"

    build_cp = subprocess.run(["make", "-C", str(verilator_dir), "all"], capture_output=True, text=True)
    assert build_cp.returncode == 0, build_cp.stderr
    bin_candidates = [
        p for p in (verilator_dir / "obj_dir").glob("V*")
        if p.is_file() and os.access(p, os.X_OK) and p.suffix == ""
    ]
    assert bin_candidates, "verilator binary not found in obj_dir"
    bin_path = bin_candidates[0]

    new_model = {
        "name": "verilator_new_model",
        "n_features": 8, "n_classes": 2, "clauses_per_class": 4, "threshold": 4,
        "include": [1, 0, 0, 0, 4, 0, 0, 0],
    }
    (sim_dir / "vnew_model.json").write_text(json.dumps(new_model))
    gen_cp = subprocess.run(
        [sys.executable, "gen_vectors.py", "combined", "vnew_model.json", "--random",
         "-n", "3", "--seed", "9", "-o", "vnew_stim.memh", "--expected", "vnew_expected.memh"],
        cwd=str(sim_dir), capture_output=True, text=True,
    )
    assert gen_cp.returncode == 0, gen_cp.stderr

    run_cp = subprocess.run(
        [str(bin_path), "--stim", "../vnew_stim.memh", "--exp", "../vnew_expected.memh"],
        cwd=str(verilator_dir), capture_output=True, text=True,
    )
    output = run_cp.stdout + run_cp.stderr
    assert "FAIL" not in output, output
    assert "TIMEOUT" not in output, output
    assert "ALL TESTS PASSED" in output, output


# ---------------------------------------------------------------------------
# gen_vectors.py sequence: chain multiple user-supplied models into one
# continuous reprogram-and-verify run.
# ---------------------------------------------------------------------------

def test_gen_vectors_sequence_builds_multi_model_stream(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"

    model_a = {"name": "seqA", "n_features": 8, "n_classes": 2, "clauses_per_class": 4,
               "threshold": 4, "include": [1, 0, 0, 0, 4, 0, 0, 0]}
    model_b = {"name": "seqB", "n_features": 8, "n_classes": 3, "clauses_per_class": 4,
               "threshold": 4, "include": [0] * 12}
    (sim_dir / "seqA.json").write_text(json.dumps(model_a))
    (sim_dir / "seqB.json").write_text(json.dumps(model_b))
    manifest = [
        {"model": "seqA.json", "random": True, "n": 2, "seed": 1},
        {"model": "seqB.json", "random": True, "n": 2, "seed": 2},
        {"model": "seqA.json", "random": True, "n": 2, "seed": 1},
    ]
    (sim_dir / "seq_manifest.json").write_text(json.dumps(manifest))

    cp = subprocess.run(
        [sys.executable, "gen_vectors.py", "sequence", "seq_manifest.json",
         "-o", "seq_stim.memh", "--expected", "seq_exp.memh"],
        cwd=str(sim_dir), capture_output=True, text=True,
    )
    assert cp.returncode == 0, cp.stderr
    assert "step 0" in cp.stdout and "step 1" in cp.stdout and "step 2" in cp.stdout
    assert (sim_dir / "seq_stim.memh").exists()
    assert (sim_dir / "seq_exp.memh").exists()

    # 3 LOAD acks expected — one per reprogram step
    exp_lines = (sim_dir / "seq_exp.memh").read_text().splitlines()
    ack_count = sum(1 for line in exp_lines if line.strip().upper().startswith("1A5000"))
    assert ack_count == 3, exp_lines


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_gen_vectors_sequence_testbench_passes_against_real_rtl(tmp_path):
    """The concrete proof of reprogrammability: two DIFFERENT user-authored
    models, chained via `sequence`, replayed against the real synthesized
    RTL in one continuous run -- must show a successful reload + correct
    predictions for each model in order."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"
    src_dir = artifacts.rtl_dir / "src"

    model_a = {"name": "seqA", "n_features": 8, "n_classes": 2, "clauses_per_class": 4,
               "threshold": 4, "include": [1, 0, 0, 0, 4, 0, 0, 0]}
    model_b = {"name": "seqB", "n_features": 8, "n_classes": 3, "clauses_per_class": 4,
               "threshold": 4, "include": [3, 0, 0, 0, 12, 0, 0, 0, 48, 0, 0, 0]}
    (sim_dir / "rseqA.json").write_text(json.dumps(model_a))
    (sim_dir / "rseqB.json").write_text(json.dumps(model_b))
    manifest = [
        {"model": "rseqA.json", "random": True, "n": 2, "seed": 3},
        {"model": "rseqB.json", "random": True, "n": 2, "seed": 4},
    ]
    (sim_dir / "rseq_manifest.json").write_text(json.dumps(manifest))

    gen_cp = subprocess.run(
        [sys.executable, "gen_vectors.py", "sequence", "rseq_manifest.json",
         "-o", "rseq_stim.memh", "--expected", "rseq_exp.memh", "--testbench", "tb_rseq.v"],
        cwd=str(sim_dir), capture_output=True, text=True,
    )
    assert gen_cp.returncode == 0, gen_cp.stderr
    assert (sim_dir / "tb_rseq.v").exists()

    sources = [
        src_dir / "axis_fifo.v", src_dir / "clause_eval.v", src_dir / "tile_mem.v",
        src_dir / "score_acc_rt.v", src_dir / "argmax_rt.v", src_dir / "tm_accel_gp.v",
    ]
    out_bin = sim_dir / "tb_rseq"
    compile_cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(out_bin)]
        + [str(s) for s in sources] + [str(sim_dir / "tb_rseq.v")],
        capture_output=True, text=True,
    )
    assert compile_cp.returncode == 0, compile_cp.stderr

    run_cp = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(sim_dir))
    output = run_cp.stdout + run_cp.stderr
    assert "FAIL" not in output, output
    assert "TIMEOUT" not in output, output
    assert "ALL TESTS PASSED" in output, output


# ---------------------------------------------------------------------------
# test_vectors.txt + curated GTKWave layout: makes "where are the test
# vectors" and "how do I compare predicted vs. expected" answerable without
# reading Verilog, and verified against the RTL's actual beat numbering.
# ---------------------------------------------------------------------------

def test_generate_exports_test_vectors_txt_and_gtkw(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    sim_dir = artifacts.rtl_dir / "sim"
    assert (sim_dir / "test_vectors.txt").exists()
    assert (sim_dir / "tb_system_gp_model.gtkw").exists()
    sim_names = {p.name for p in artifacts.sim_scripts}
    assert {"test_vectors.txt", "tb_system_gp_model.gtkw"} <= sim_names


def test_test_vectors_txt_lists_every_embedded_vector_correctly(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    text = (artifacts.rtl_dir / "sim" / "test_vectors.txt").read_text()
    vecs = tmir.verification.test_vectors
    assert f"{len(vecs)} test vector(s)" in text
    for i, tv in enumerate(vecs):
        bits = "".join(str(int(b) & 1) for b in tv.input)
        # row must show this index, its expected class, and its exact bit pattern
        assert f"{i:>4}  {tv.expected_class:>14}  {bits}" in text


def test_test_vectors_txt_handles_no_embedded_vectors(tmp_path):
    tmir = _make_tmir(with_vectors=False)
    assert tmir.verification is None
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    text = (artifacts.rtl_dir / "sim" / "test_vectors.txt").read_text()
    assert "none embedded" in text
    assert "0 test vector(s)" in text


def test_model_gtkw_signal_paths_match_generated_testbench_hierarchy(tmp_path):
    """The .gtkw's signal references must actually resolve against
    tb_system_gp_model.v's real module/instance names, not just look
    plausible -- checked against the generated testbench source directly."""
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    gtkw = (artifacts.rtl_dir / "sim" / "tb_system_gp_model.gtkw").read_text()
    tb_text = [p for p in artifacts.testbenches if p.name == "tb_system_gp_model.v"][0].read_text()

    assert "module tb_system_gp_model;" in tb_text
    assert ") dut (" in tb_text  # instance is named "dut"

    for sig in ("s_tvalid", "s_tready", "s_tdata", "s_tlast",
                "m_tvalid", "m_tready", "m_tdata", "m_tlast",
                "busy", "configured", "e", "sent", "fail_cnt"):
        assert f"tb_system_gp_model.{sig}" in gtkw
    assert "tb_system_gp_model.dut.state" in gtkw


def test_waves_sh_uses_matching_gtkw_when_present(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)

    script = (artifacts.rtl_dir / "sim" / "waves.sh").read_text()
    assert 'GTKW="${WF%.vcd}.gtkw"' in script
    assert 'gtkwave "$WF" "$GTKW"' in script


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_waveform_beat_numbering_matches_test_vectors_txt_exactly(tmp_path):
    """The concrete claim documented in the generated README (beat[0] is the
    LOAD ack, beat[k+1] corresponds to test_vectors.txt row k) verified
    against real simulation output, not just asserted."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    sim_dir = artifacts.rtl_dir / "sim"
    src_dir = artifacts.rtl_dir / "src"
    tb_dir = artifacts.rtl_dir / "tb"

    sources = [
        src_dir / "axis_fifo.v", src_dir / "clause_eval.v", src_dir / "tile_mem.v",
        src_dir / "score_acc_rt.v", src_dir / "argmax_rt.v", src_dir / "tm_accel_gp.v",
    ]
    out_bin = sim_dir / "tb_system_gp_model"
    compile_cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(out_bin)]
        + [str(s) for s in sources] + [str(tb_dir / "tb_system_gp_model.v")],
        capture_output=True, text=True,
    )
    assert compile_cp.returncode == 0, compile_cp.stderr
    run_cp = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(sim_dir))
    output = run_cp.stdout

    beat_data = {}
    for line in output.splitlines():
        m = re.match(r"PASS beat\[(\d+)\]: data=([0-9a-fA-F]+)", line)
        if m:
            beat_data[int(m.group(1))] = int(m.group(2), 16)

    vecs = tmir.verification.test_vectors
    assert beat_data[0] & 0xFF000000 == 0xA5000000, "beat[0] must be the LOAD ack"
    for k, tv in enumerate(vecs):
        assert beat_data[k + 1] == tv.expected_class, (
            f"beat[{k+1}] should be test_vectors.txt row {k}'s prediction "
            f"({tv.expected_class}), got {beat_data[k+1]}"
        )


# ---------------------------------------------------------------------------
# reprogram-suite: multi-model/dataset RTL reprogramming (build_reprogram_suite)
# ---------------------------------------------------------------------------

def test_supports_reprogramming_is_true_for_gp_tiled():
    assert GPTiledBackend().supports_reprogramming is True


def test_base_backends_do_not_support_reprogramming():
    from matador.backends.hardwired.rtl import HardwiredBackend
    from matador.backends.tiled.rtl import TiledBackend

    assert TiledBackend().supports_reprogramming is False
    assert HardwiredBackend().supports_reprogramming is False
    with pytest.raises(NotImplementedError):
        TiledBackend().build_reprogram_suite(None, [], None)


def test_build_reprogram_suite_requires_prior_generate(tmp_path):
    backend = GPTiledBackend()
    step = ReprogramStep(tmir_path=tmp_path / "model.yaml")
    with pytest.raises(FileNotFoundError, match="matador generate"):
        backend.build_reprogram_suite(tmp_path / "RTL", [step], _make_cfg(tmp_path))


def test_build_reprogram_suite_rejects_empty_steps(tmp_path):
    tmir = _make_tmir()
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir, cfg)
    with pytest.raises(ValueError, match="non-empty"):
        backend.build_reprogram_suite(artifacts.rtl_dir, [], cfg)


def test_build_reprogram_suite_writes_expected_files(tmp_path):
    tmir_a = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir_a)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir_a, cfg)

    tmir_b = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir_b_path = tmp_path / "model_b.yaml"
    tmir_b.to_yaml(tmir_b_path)
    tmir_a_path = tmp_path / "model_a.yaml"
    tmir_a.to_yaml(tmir_a_path)

    steps = [
        ReprogramStep(tmir_path=tmir_a_path, name="modelA"),
        ReprogramStep(tmir_path=tmir_b_path, name="modelB"),
    ]
    result = backend.build_reprogram_suite(artifacts.rtl_dir, steps, cfg)

    assert result.testbenches == [artifacts.rtl_dir / "tb" / "tb_reprogram_suite.v"]
    sim_names = {p.name for p in result.sim_scripts}
    assert sim_names == {
        "reprogram_stimulus.memh", "reprogram_expected.memh",
        "reprogram_manifest.txt", "tb_reprogram_suite.gtkw",
    }
    for p in [*result.testbenches, *result.sim_scripts]:
        assert p.exists()

    manifest = (artifacts.rtl_dir / "sim" / "reprogram_manifest.txt").read_text()
    assert "step 0: model='modelA'" in manifest
    assert "step 1: model='modelB'" in manifest
    assert "beat 0: LOAD ack" in manifest


def test_build_reprogram_suite_rejects_model_exceeding_capacity(tmp_path):
    tmir_small = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir_small, max_features=16, max_clauses_total=8)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir_small, cfg)

    tmir_huge = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir_huge.architecture.n_clauses_total = 999999   # deliberately exceeds cfg's capacity
    huge_path = tmp_path / "huge.yaml"
    tmir_huge.to_yaml(huge_path)

    steps = [ReprogramStep(tmir_path=huge_path)]
    with pytest.raises(GPCapacityError):
        backend.build_reprogram_suite(artifacts.rtl_dir, steps, cfg)


def test_reprogram_step_vectors_from_dataset_drops_label_and_uses_model_prediction(tmp_path):
    """dataset_path steps must check RTL against the MODEL's own prediction
    (model.infer_tiled), not the dataset's ground-truth label column --
    consistent with how every other testbench in this backend works."""
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4, with_vectors=False)
    cfg = _make_cfg(tmp_path, tmir=tmir)
    backend = GPTiledBackend()
    model = tmir_to_tmmodel(tmir)

    dataset_path = tmp_path / "ds_test.txt"
    rng = np.random.default_rng(3)
    rows = []
    fvs = []
    for _ in range(4):
        bits = rng.integers(0, 2, size=8)
        fvs.append(sum(int(b) << f for f, b in enumerate(bits)))
        wrong_label = 99   # deliberately NOT the model's real prediction
        rows.append(" ".join(map(str, bits.tolist())) + f" {wrong_label}")
    dataset_path.write_text("\n".join(rows) + "\n")

    step = ReprogramStep(tmir_path=tmp_path / "unused.yaml", dataset_path=dataset_path)
    pairs = backend._resolve_step_vectors(step, tmir, model)

    assert len(pairs) == 4
    for (fv, expected), real_fv in zip(pairs, fvs):
        assert fv == real_fv
        assert expected == model.infer_tiled(fv)[0]   # not 99


def test_reprogram_step_vectors_default_uses_embedded_test_vectors(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4, with_vectors=True)
    backend = GPTiledBackend()
    model = tmir_to_tmmodel(tmir)
    step = ReprogramStep(tmir_path=tmp_path / "unused.yaml")
    pairs = backend._resolve_step_vectors(step, tmir, model)

    assert len(pairs) == len(tmir.verification.test_vectors)
    for (fv, expected), tv in zip(pairs, tmir.verification.test_vectors):
        assert expected == tv.expected_class
        assert fv == sum((int(b) & 1) << f for f, b in enumerate(tv.input))


def test_reprogram_step_n_samples_subsamples_dataset(tmp_path):
    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4, with_vectors=False)
    backend = GPTiledBackend()
    model = tmir_to_tmmodel(tmir)

    dataset_path = tmp_path / "big_test.txt"
    rng = np.random.default_rng(4)
    rows = [" ".join(map(str, rng.integers(0, 2, size=8).tolist())) + " 0" for _ in range(20)]
    dataset_path.write_text("\n".join(rows) + "\n")

    step = ReprogramStep(tmir_path=tmp_path / "unused.yaml", dataset_path=dataset_path, n_samples=5, seed=0)
    pairs = backend._resolve_step_vectors(step, tmir, model)
    assert len(pairs) == 5


# --- ReprogramSuiteConfig schema ---

def test_reprogram_suite_config_rejects_empty_steps():
    with pytest.raises(Exception, match="non-empty"):
        ReprogramSuiteConfig.model_validate({"steps": []})


def test_reprogram_step_config_rejects_missing_model(tmp_path):
    with pytest.raises(Exception, match="does not exist"):
        ReprogramStepConfig.model_validate({"model": str(tmp_path / "nope.yaml")})


def test_reprogram_step_config_accepts_real_model(tmp_path):
    tmir = _make_tmir()
    p = tmp_path / "m.yaml"
    tmir.to_yaml(p)
    cfg = ReprogramStepConfig.model_validate({"model": str(p)})
    assert cfg.model == p
    assert cfg.dataset is None


# --- Real end-to-end RTL test: 2 models, 1 from a dataset file ---

@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_reprogram_suite_rtl_passes_across_two_models(tmp_path):
    tmir_a = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    cfg = _make_cfg(tmp_path, tmir=tmir_a)
    backend = GPTiledBackend()
    artifacts = backend.generate(tmir_a, cfg)

    tmir_b = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir_b_path = tmp_path / "model_b.yaml"
    tmir_b.to_yaml(tmir_b_path)
    tmir_a_path = tmp_path / "model_a.yaml"
    tmir_a.to_yaml(tmir_a_path)

    rng = np.random.default_rng(0)
    rows = [" ".join(map(str, rng.integers(0, 2, size=8).tolist())) + " 0" for _ in range(5)]
    dataset_path = tmp_path / "model_b_test.txt"
    dataset_path.write_text("\n".join(rows) + "\n")

    steps = [
        ReprogramStep(tmir_path=tmir_a_path, name="modelA"),
        ReprogramStep(tmir_path=tmir_b_path, dataset_path=dataset_path, n_samples=3, seed=1, name="modelB"),
    ]
    result = backend.build_reprogram_suite(artifacts.rtl_dir, steps, cfg)

    rtl_dir = artifacts.rtl_dir
    src_dir = rtl_dir / "src"
    sim_dir = rtl_dir / "sim"
    sources = [src_dir / f for f in
               ("axis_fifo.v", "clause_eval.v", "tile_mem.v", "score_acc_rt.v", "argmax_rt.v", "tm_accel_gp.v")]
    out_bin = sim_dir / "tb_reprogram_suite"
    compile_cp = subprocess.run(
        ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", str(out_bin)]
        + [str(s) for s in sources] + [str(result.testbenches[0])],
        capture_output=True, text=True,
    )
    assert compile_cp.returncode == 0, compile_cp.stderr
    run_cp = subprocess.run(["vvp", str(out_bin)], capture_output=True, text=True, cwd=str(sim_dir))
    output = run_cp.stdout + run_cp.stderr
    assert "FAIL" not in output, output
    assert "TIMEOUT" not in output, output
    assert "ALL TESTS PASSED (13 beats checked)" in output, output   # 1+8 (modelA embedded) + 1+3 (modelB dataset)


# --- CLI ---

def test_cli_reprogram_suite_end_to_end(tmp_path):
    import yaml

    from matador.cli import main

    tmir_a = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir_a_path = tmp_path / "model_a.yaml"
    tmir_a.to_yaml(tmir_a_path)
    tmir_b = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir_b_path = tmp_path / "model_b.yaml"
    tmir_b.to_yaml(tmir_b_path)

    accel_cfg_path = tmp_path / "vanilla_gp_tiled.yaml"
    accel_cfg_path.write_text(yaml.safe_dump({
        "model_path": str(tmir_a_path), "output_dir": str(tmp_path / "out"),
        "fifo_depth": 16, "target_fpga": "xc7z020", "max_features": 512, "max_clauses_total": 256,
    }))
    try:
        main(["generate", "--backend", "vanilla_gp_tiled", "--config", str(accel_cfg_path)], standalone_mode=False)
    except SystemExit:
        pass

    reprogram_cfg_path = tmp_path / "reprogram_config.yaml"
    reprogram_cfg_path.write_text(yaml.safe_dump({"steps": [
        {"model": str(tmir_a_path), "name": "modelA"},
        {"model": str(tmir_b_path), "name": "modelB"},
    ]}))

    try:
        main(["reprogram-suite", "--backend", "vanilla_gp_tiled", "--config", str(accel_cfg_path),
              "--reprogram-config", str(reprogram_cfg_path)], standalone_mode=False)
    except SystemExit:
        pass

    rtl_dir = tmp_path / "out" / "vanilla_gp_tiled" / "RTL"
    assert (rtl_dir / "tb" / "tb_reprogram_suite.v").exists()
    assert (rtl_dir / "sim" / "reprogram_stimulus.memh").exists()


def test_cli_reprogram_suite_rejects_unsupported_backend(tmp_path):
    import yaml

    from matador.cli import main
    from click.testing import CliRunner

    reprogram_cfg_path = tmp_path / "reprogram_config.yaml"
    reprogram_cfg_path.write_text(yaml.safe_dump({"steps": [{"model": str(tmp_path / "x.yaml")}]}))
    # x.yaml doesn't need to exist for this test -- it should fail on the
    # backend-support check before ever validating step contents.
    (tmp_path / "x.yaml").write_text("dummy: true")

    runner = CliRunner()
    result = runner.invoke(main, [
        "reprogram-suite", "--backend", "vanilla_tiled",
        "--config", str(tmp_path / "vanilla_tiled.yaml"),
        "--reprogram-config", str(reprogram_cfg_path),
    ])
    assert result.exit_code != 0
    assert "does not support runtime reprogramming" in result.output
