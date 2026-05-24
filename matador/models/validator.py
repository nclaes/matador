"""TMIR model validation — schema checks, test-vector verification, accuracy."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)


@dataclass
class RTLTestResult:
    name: str
    passed: bool
    summary: str       # last status line printed by the testbench
    failures: list     # FAIL / TIMEOUT / compile-error lines


@dataclass
class RTLValidationReport:
    results: list      # list[RTLTestResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def n_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)


@dataclass
class ValidationReport:
    model_path: Path
    schema_ok: bool = False
    schema_error: str = ""
    vectors_total: int = 0
    vectors_passed: int = 0
    vectors_ran: bool = False
    accuracy_pct: Optional[float] = None
    accuracy_correct: Optional[int] = None
    accuracy_total: Optional[int] = None

    @property
    def all_passed(self) -> bool:
        if not self.schema_ok:
            return False
        if self.vectors_ran and self.vectors_passed < self.vectors_total:
            return False
        return True


def validate_model(config) -> ValidationReport:
    """Run all validation checks against a TMIR model.

    Args:
        config: ValidationConfig instance.

    Returns:
        ValidationReport with results of every check performed.
    """
    from matador.ir.tm_ir import TMIR
    from matador.inference.reference import predict, verify_against_vectors

    path = config.model_path
    report = ValidationReport(model_path=path)

    # ── Load ─────────────────────────────────────────────────────────────────
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        tmir = TMIR.from_yaml(path)
    else:
        tmir = TMIR.from_npz(path)

    # ── Schema validation ─────────────────────────────────────────────────────
    try:
        tmir.validate_self()
        report.schema_ok = True
    except ValueError as exc:
        report.schema_ok = False
        report.schema_error = str(exc)
        return report

    # ── Test-vector verification ──────────────────────────────────────────────
    if tmir.verification and tmir.verification.test_vectors:
        result = verify_against_vectors(tmir)
        report.vectors_ran = True
        report.vectors_total = result.total
        report.vectors_passed = result.passed

    # ── Held-out test-set accuracy ────────────────────────────────────────────
    if config.test_data is not None:
        raw = np.genfromtxt(config.test_data, delimiter=" ", dtype=np.uint32)
        X_test = raw[:, :-1].astype(np.uint8)
        y_test = raw[:, -1]

        preds, _ = predict(tmir, X_test)
        correct = int((preds == y_test).sum())
        total = len(y_test)
        report.accuracy_correct = correct
        report.accuracy_total = total
        report.accuracy_pct = 100.0 * correct / total if total > 0 else 0.0

    return report


def validate_rtl(config) -> RTLValidationReport:
    """Compile and run the testbench suite against the generated RTL.

    Unit tests (axis_fifo, clause_eval, score_acc, argmax) always run under iverilog.
    The system test (tb_system) runs under Verilator when available (produces FST
    waveforms at sim/verilator/tb_system.fst), falling back to iverilog otherwise.

    Args:
        config: TMAcceleratorConfig instance (needs output_dir).

    Returns:
        RTLValidationReport with per-testbench pass/fail results.

    Raises:
        FileNotFoundError: RTL not generated yet, or neither iverilog nor verilator in PATH.
    """
    import shutil
    import subprocess

    rtl_dir      = config.output_dir / "RTL"
    src_dir      = rtl_dir / "src"
    tb_dir       = rtl_dir / "tb"
    sim_dir      = rtl_dir / "sim"
    verilator_dir = sim_dir / "verilator"

    if not rtl_dir.exists():
        raise FileNotFoundError(
            f"RTL directory not found: {rtl_dir}\n"
            "  Run 'matador generate --config <accelerator_config.yaml>' first."
        )

    have_iverilog  = shutil.which("iverilog") is not None
    have_verilator = shutil.which("verilator") is not None

    if not have_iverilog and not have_verilator:
        raise FileNotFoundError(
            "Neither iverilog nor verilator found in PATH.\n"
            "  Install one: apt-get install iverilog   OR   apt-get install verilator"
        )

    sources = [
        str(src_dir / "axis_fifo.v"),
        str(src_dir / "clause_eval.v"),
        str(src_dir / "score_acc.v"),
        str(src_dir / "argmax.v"),
        str(src_dir / "tm_accelerator.v"),
    ]
    results: list = []

    # ── Unit tests — always iverilog ──────────────────────────────────────────
    unit_tbs = ["tb_axis_fifo", "tb_clause_eval", "tb_score_acc", "tb_argmax"]
    if have_iverilog:
        for tb_name in unit_tbs:
            tb_file = str(tb_dir / f"{tb_name}.v")
            out_bin = str(sim_dir / tb_name)

            cp = subprocess.run(
                ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", out_bin]
                + sources + [tb_file],
                capture_output=True, text=True,
            )
            if cp.returncode != 0:
                err_lines = [l for l in (cp.stderr or cp.stdout).splitlines() if l.strip()]
                results.append(RTLTestResult(
                    name=tb_name, passed=False,
                    summary="compile error",
                    failures=err_lines[:10],
                ))
                continue

            rp = subprocess.run(["vvp", out_bin], capture_output=True, text=True,
                                cwd=str(sim_dir))
            all_lines = (rp.stdout + rp.stderr).splitlines()
            failures  = [l for l in all_lines if "FAIL" in l or "TIMEOUT" in l]
            passed    = not failures and rp.returncode == 0
            summary   = next((l for l in reversed(all_lines) if l.strip()), "")

            results.append(RTLTestResult(
                name=tb_name, passed=passed,
                summary=summary, failures=failures,
            ))
    else:
        _LOGGER.warning("iverilog not found — skipping unit tests")

    # ── System test — Verilator preferred, iverilog fallback ─────────────────
    have_verilator_harness = verilator_dir.exists() and (verilator_dir / "Makefile").exists()

    if have_verilator and have_verilator_harness:
        _LOGGER.debug("Running tb_system with Verilator (FST output)")
        cp = subprocess.run(
            ["make", "-C", str(verilator_dir), "all"],
            capture_output=True, text=True,
        )
        if cp.returncode != 0:
            err_lines = [l for l in (cp.stderr + cp.stdout).splitlines() if l.strip()]
            results.append(RTLTestResult(
                name="tb_system", passed=False,
                summary="verilator build error",
                failures=err_lines[:15],
            ))
        else:
            bin_path = verilator_dir / "obj_dir" / "Vtm_accelerator"
            rp = subprocess.run(
                [str(bin_path)], capture_output=True, text=True,
                cwd=str(verilator_dir),
            )
            all_lines = (rp.stdout + rp.stderr).splitlines()
            failures  = [l for l in all_lines if "FAIL" in l or "TIMEOUT" in l]
            passed    = not failures and rp.returncode == 0
            summary   = next((l for l in reversed(all_lines) if l.strip()), "")
            # Annotate to show Verilator was used
            results.append(RTLTestResult(
                name="tb_system", passed=passed,
                summary=f"[verilator] {summary}",
                failures=failures,
            ))

    elif have_iverilog:
        _LOGGER.debug("Verilator not available — running tb_system with iverilog")
        tb_name = "tb_system"
        tb_file = str(tb_dir / f"{tb_name}.v")
        out_bin = str(sim_dir / tb_name)

        cp = subprocess.run(
            ["iverilog", "-g2001", "-Wall", "-Wno-timescale", "-o", out_bin]
            + sources + [tb_file],
            capture_output=True, text=True,
        )
        if cp.returncode != 0:
            err_lines = [l for l in (cp.stderr or cp.stdout).splitlines() if l.strip()]
            results.append(RTLTestResult(
                name=tb_name, passed=False,
                summary="compile error",
                failures=err_lines[:10],
            ))
        else:
            rp = subprocess.run(["vvp", out_bin], capture_output=True, text=True,
                                cwd=str(sim_dir))
            all_lines = (rp.stdout + rp.stderr).splitlines()
            failures  = [l for l in all_lines if "FAIL" in l or "TIMEOUT" in l]
            passed    = not failures and rp.returncode == 0
            summary   = next((l for l in reversed(all_lines) if l.strip()), "")
            results.append(RTLTestResult(
                name=tb_name, passed=passed,
                summary=summary, failures=failures,
            ))
    else:
        _LOGGER.warning("No simulator available for tb_system")

    return RTLValidationReport(results=results)
