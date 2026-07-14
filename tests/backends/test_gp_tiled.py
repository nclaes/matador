"""Tests for matador.backends.gp_tiled — the runtime-reprogrammable
GP-tiled backend (vendored from GP_TM_Inference_Accelerator)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import pytest

from matador.backends.gp_tiled.config import GPTiledAcceleratorConfig
from matador.backends.gp_tiled.emulator import GPTiledEmulator
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
