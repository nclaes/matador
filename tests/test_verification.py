"""Tests for matador.verification.compare."""

from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from matador.cli import main
from matador.config.schema import TMAcceleratorConfig
from matador.ir.tm_ir import (
    Architecture,
    EvalVector,
    Hyperparameters,
    Representation,
    TMIR,
    Verification,
)
from matador.verification.compare import (
    ComparisonReport,
    compare_emulator_to_reference,
    run_regression,
)

INCLUDE = 200
EXCLUDE = 50


def _make_tmir(n_features=4, n_classes=2, n_clauses_pc=4, threshold=4) -> TMIR:
    n_literals = 2 * n_features
    ta = np.full((n_classes, n_clauses_pc, n_literals), EXCLUDE, dtype=np.int32)
    ta[0, 0, 0] = INCLUDE
    ta[0, 1, 1] = INCLUDE
    ta[1, 0, 0] = INCLUDE
    ta[1, 0, 1] = INCLUDE
    ta[1, 2, n_features] = INCLUDE
    return TMIR(
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


def _make_cfg(tmp_path) -> TMAcceleratorConfig:
    tmir_path = tmp_path / "model.yaml"
    _make_tmir().to_yaml(tmir_path)
    return TMAcceleratorConfig.model_validate({
        "model_path": str(tmir_path),
        "output_dir": str(tmp_path),
    })


# ---------------------------------------------------------------------------
# compare_emulator_to_reference
# ---------------------------------------------------------------------------

class TestCompareEmulatorToReference:
    def _tmir_with_vectors(self, tmp_path):
        tmir = _make_tmir()
        tmir.verification = Verification(test_vectors=[
            EvalVector(input=[0, 0, 0, 0], expected_class=0, expected_scores=[]),
            EvalVector(input=[1, 1, 0, 0], expected_class=0, expected_scores=[]),
            EvalVector(input=[1, 0, 0, 0], expected_class=0, expected_scores=[]),
            EvalVector(input=[0, 1, 0, 0], expected_class=0, expected_scores=[]),
        ])
        return tmir

    def test_all_pass_on_valid_model(self, tmp_path):
        tmir = self._tmir_with_vectors(tmp_path)
        cfg  = _make_cfg(tmp_path)
        report = compare_emulator_to_reference(tmir, cfg)
        assert report.all_passed, f"Expected all passed, got {report.n_failed} failures"
        assert report.n_vectors == 4

    def test_returns_comparison_report(self, tmp_path):
        tmir = self._tmir_with_vectors(tmp_path)
        cfg  = _make_cfg(tmp_path)
        report = compare_emulator_to_reference(tmir, cfg)
        assert isinstance(report, ComparisonReport)
        assert len(report.results) == 4

    def test_empty_vectors_returns_zero_count(self, tmp_path):
        tmir = _make_tmir()  # no verification block
        cfg  = _make_cfg(tmp_path)
        report = compare_emulator_to_reference(tmir, cfg)
        assert report.n_vectors == 0
        assert report.all_passed

    def test_vector_result_fields(self, tmp_path):
        tmir = self._tmir_with_vectors(tmp_path)
        cfg  = _make_cfg(tmp_path)
        report = compare_emulator_to_reference(tmir, cfg)
        r0 = report.results[0]
        assert r0.index == 0
        assert r0.input_features == [0, 0, 0, 0]
        assert r0.expected_class == 0
        assert isinstance(r0.emulator_class, int)
        assert isinstance(r0.reference_class, int)
        assert r0.emulator_matches_reference

    def test_pass_rate(self, tmp_path):
        tmir = self._tmir_with_vectors(tmp_path)
        cfg  = _make_cfg(tmp_path)
        report = compare_emulator_to_reference(tmir, cfg)
        assert report.pass_rate == 100.0


# ---------------------------------------------------------------------------
# run_regression
# ---------------------------------------------------------------------------

class TestRunRegression:
    def test_with_extra_vectors(self, tmp_path):
        tmir = _make_tmir()
        cfg  = _make_cfg(tmp_path)
        extras = [
            ([0, 0, 0, 0], 0),
            ([1, 1, 0, 0], None),
        ]
        report = run_regression(tmir, cfg, extra_vectors=extras)
        assert report.n_vectors == 2
        assert report.all_passed

    def test_with_embedded_and_extra(self, tmp_path):
        tmir = _make_tmir()
        tmir.verification = Verification(test_vectors=[
            EvalVector(input=[0, 0, 0, 0], expected_class=0, expected_scores=[]),
        ])
        cfg  = _make_cfg(tmp_path)
        extras = [([1, 0, 0, 0], None)]
        report = run_regression(tmir, cfg, extra_vectors=extras)
        assert report.n_vectors == 2

    def test_no_vectors_returns_zero(self, tmp_path):
        tmir = _make_tmir()
        cfg  = _make_cfg(tmp_path)
        report = run_regression(tmir, cfg)
        assert report.n_vectors == 0


# ---------------------------------------------------------------------------
# CLI — matador emulate
# ---------------------------------------------------------------------------

class TestEmulateCLI:
    def _setup(self, tmp_path):
        tmir = _make_tmir()
        tmir.verification = Verification(test_vectors=[
            EvalVector(input=[0, 0, 0, 0], expected_class=0, expected_scores=[]),
            EvalVector(input=[1, 1, 0, 0], expected_class=0, expected_scores=[]),
        ])
        tmir_path = tmp_path / "model.yaml"
        tmir.to_yaml(tmir_path)

        cfg_dict = {
            "model_path": str(tmir_path),
            "output_dir": str(tmp_path),
        }
        import yaml
        cfg_path = tmp_path / "accel.yaml"
        cfg_path.write_text(yaml.dump(cfg_dict))
        return cfg_path, tmir

    def test_basic_run(self, tmp_path):
        cfg_path, _ = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["emulate", "--config", str(cfg_path)])
        assert result.exit_code == 0, result.output
        assert "passed" in result.output.lower()

    def test_verbose_flag(self, tmp_path):
        cfg_path, _ = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["emulate", "--config", str(cfg_path), "--verbose"])
        assert result.exit_code == 0
        assert "ROM reads" in result.output

    def test_verify_flag(self, tmp_path):
        cfg_path, _ = self._setup(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["emulate", "--config", str(cfg_path), "--verify"])
        assert result.exit_code == 0
        assert "matches reference" in result.output

    def test_trace_output(self, tmp_path):
        cfg_path, _ = self._setup(tmp_path)
        trace_out = tmp_path / "trace.json"
        runner = CliRunner()
        result = runner.invoke(main, [
            "emulate", "--config", str(cfg_path),
            "--trace", str(trace_out),
        ])
        assert result.exit_code == 0
        assert trace_out.exists()
        import json
        data = json.loads(trace_out.read_text())
        assert isinstance(data, list)
        assert len(data) == 2  # 2 test vectors

    def test_no_vectors_fails(self, tmp_path):
        tmir = _make_tmir()
        tmir_path = tmp_path / "model.yaml"
        tmir.to_yaml(tmir_path)
        import yaml
        cfg_path = tmp_path / "accel.yaml"
        cfg_path.write_text(yaml.dump({
            "model_path": str(tmir_path),
            "output_dir": str(tmp_path),
        }))
        runner = CliRunner()
        result = runner.invoke(main, ["emulate", "--config", str(cfg_path)])
        assert result.exit_code != 0
        assert "No test vectors" in result.output

    def test_missing_config_fails(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["emulate", "--config", str(tmp_path / "ghost.yaml")])
        assert result.exit_code != 0
