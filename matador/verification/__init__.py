"""Cross-reference verification: emulator vs. RTL simulation results."""

from matador.verification.compare import (
    ComparisonReport,
    VectorResult,
    compare_emulator_to_reference,
    run_regression,
)

__all__ = [
    "ComparisonReport",
    "VectorResult",
    "compare_emulator_to_reference",
    "run_regression",
]
