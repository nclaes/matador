from pathlib import Path

import numpy as np

from coal_tm.config import RTLConfig
from coal_tm.emulator import CoalescedEmulator
from coal_tm.validate import validate


def _config(tiny_model, tmp_path, test_data):
    return RTLConfig(
        output_dir=tmp_path / "out",
        tas=tiny_model["tas"],
        weights=tiny_model["weights"],
        classes=tiny_model["classes"],
        clauses=tiny_model["clauses"],
        features=tiny_model["features"],
        bus_width=8,
        test_data=test_data,
    )


def test_validate_reports_full_accuracy_when_labels_match_emulator(tiny_model, tmp_path):
    emu = CoalescedEmulator(
        tiny_model["tas"], tiny_model["weights"], tiny_model["classes"],
        tiny_model["clauses"], tiny_model["features"],
    )
    rows = []
    for f0, f1 in [(1, 0), (1, 1), (0, 0)]:
        x = [0] * 16
        x[0], x[1] = f0, f1
        label = emu.predict(np.array(x, dtype=np.uint8)).predicted_class
        rows.append(x + [label])
    test_data = tmp_path / "test_data.txt"
    test_data.write_text("\n".join(" ".join(map(str, r)) for r in rows) + "\n")

    config = _config(tiny_model, tmp_path, test_data)
    result = validate(config, n_vectors=3)
    assert result.correct == 3
    assert result.accuracy == 100.0


def test_validate_detects_mismatches(tiny_model, tmp_path):
    # x0=1,x1=0 fires clause 0 -> emulator predicts class 0 (see test_emulator.py).
    # Label it as class 1 on purpose so validate must report a mismatch.
    row = [1, 0] + [0] * 14 + [1]
    test_data = tmp_path / "test_data.txt"
    test_data.write_text(" ".join(map(str, row)) + "\n")

    config = _config(tiny_model, tmp_path, test_data)
    result = validate(config, n_vectors=1)
    assert result.correct == 0
    assert result.accuracy == 0.0


def test_validate_rejects_too_few_rows(tiny_model, tmp_path):
    row = [1, 0] + [0] * 14 + [0]
    test_data = tmp_path / "test_data.txt"
    test_data.write_text(" ".join(map(str, row)) + "\n")

    config = _config(tiny_model, tmp_path, test_data)
    try:
        validate(config, n_vectors=5)
    except ValueError as exc:
        assert "only 1 rows" in str(exc)
    else:
        raise AssertionError("expected ValueError for too few rows")
