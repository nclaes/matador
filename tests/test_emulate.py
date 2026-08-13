import numpy as np

from coal_tm.config import RTLConfig
from coal_tm.emulate import emulate_batch, emulate_single


def _config(tiny_model, tmp_path, test_data=None):
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


def test_emulate_single_matches_direct_emulator_call(tiny_model, tmp_path):
    config = _config(tiny_model, tmp_path)
    x = np.zeros(16, dtype=np.uint8)
    x[0], x[1] = 1, 0
    result = emulate_single(config, x)
    assert result.predicted_class == 0


def test_emulate_batch_defaults_to_config_test_data(tiny_model, tmp_path):
    # No trailing label column -- emulate_batch must not require one.
    row = [1, 0] + [0] * 14
    test_data = tmp_path / "test_data.txt"
    test_data.write_text(" ".join(map(str, row)) + "\n")

    config = _config(tiny_model, tmp_path, test_data=test_data)
    result = emulate_batch(config)
    assert list(result.predictions) == [0]


def test_emulate_batch_strips_trailing_label_column(tiny_model, tmp_path):
    # 17 columns = 16 features + 1 label -- must be auto-stripped.
    row = [1, 0] + [0] * 14 + [9]
    test_data = tmp_path / "test_data.txt"
    test_data.write_text(" ".join(map(str, row)) + "\n")

    config = _config(tiny_model, tmp_path, test_data=test_data)
    result = emulate_batch(config)
    assert list(result.predictions) == [0]


def test_emulate_batch_respects_n_vectors_and_test_data_override(tiny_model, tmp_path):
    rows = [[1, 0] + [0] * 14, [1, 1] + [0] * 14, [0, 0] + [0] * 14]
    test_data = tmp_path / "test_data.txt"
    test_data.write_text("\n".join(" ".join(map(str, r)) for r in rows) + "\n")

    config = _config(tiny_model, tmp_path)  # no test_data in the config itself
    result = emulate_batch(config, n_vectors=2, test_data=test_data)
    assert len(result.predictions) == 2


def test_emulate_batch_requires_a_data_source(tiny_model, tmp_path):
    config = _config(tiny_model, tmp_path)  # no test_data anywhere
    try:
        emulate_batch(config)
    except ValueError as exc:
        assert "no test_data" in str(exc)
    else:
        raise AssertionError("expected ValueError when no test_data is available")
