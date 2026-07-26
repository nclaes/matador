"""Tests for matador.preprocessing.booleanize — Stage 2 of the pipeline
(raw arrays -> Boolean train/test text files matador train already
consumes unchanged)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from matador.config.schema import BooleanisationConfig, FeatureEncoderSpec
from matador.preprocessing import encoders
from matador.preprocessing.booleanize import run_booleanize

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Per-encoder unit tests
# ---------------------------------------------------------------------------

def test_thermometer_monotonic_and_range():
    train_col = np.array([0.0, 4.0, 8.0, 12.0, 16.0])
    thresholds = encoders.fit_thermometer(train_col, bits=8, value_range=(0, 16), bins=None, quantile=False)
    assert thresholds.shape == (8,)
    bits = encoders.encode_thermometer(train_col, thresholds)
    assert bits.shape == (5, 8)
    # thermometer property: within each row, 1s form a prefix (non-increasing)
    for row in bits:
        assert list(row) == sorted(row, reverse=True)
    assert bits[0].sum() == 0     # value 0 <= every threshold
    assert bits[-1].sum() == 8    # value 16 > every threshold (thresholds strictly inside (0,16))


def test_thermometer_nan_encodes_to_zero_bits():
    train_col = np.array([0.0, 8.0, 16.0, np.nan])
    thresholds = encoders.fit_thermometer(train_col, bits=4, value_range=(0, 16), bins=None, quantile=False)
    bits = encoders.encode_thermometer(train_col, thresholds)
    assert bits[3].sum() == 0


def test_thermometer_quantile_fits_from_train_only():
    train_col = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    thresholds = encoders.fit_thermometer(train_col, bits=2, value_range=None, bins=None, quantile=True)
    assert thresholds.shape == (2,)
    assert 1.0 <= thresholds[0] <= 5.0


def test_threshold_encoder():
    col = np.array([-1.0, 0.0, 1.0, np.nan])
    bits = encoders.encode_threshold(col, threshold=0.0)
    assert bits.tolist() == [[0], [0], [1], [0]]


def test_onehot_encoder_explicit_categories():
    train_col = np.array([1, 2, 3])
    cats = encoders.fit_onehot(train_col, categories=[1, 2, 3])
    test_col = np.array([2, 3, 1, 99])
    bits = encoders.encode_onehot(test_col, cats)
    assert bits.tolist() == [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 0]]  # unseen category -> all-zero


def test_onehot_encoder_auto_detected_categories():
    train_col = np.array([5, 1, 3, 1])
    cats = encoders.fit_onehot(train_col, categories=None)
    assert cats == [1, 3, 5]   # sorted, from train split only


def test_passthrough_encoder():
    col = np.array([0.0, 1.0, 2.0, np.nan])
    bits = encoders.encode_passthrough(col)
    assert bits.tolist() == [[0], [1], [1], [0]]


# ---------------------------------------------------------------------------
# run_booleanize: column resolution, split handling, output format
# ---------------------------------------------------------------------------

def _make_raw_npz(tmp_path: Path, **arrays) -> Path:
    p = tmp_path / "raw.npz"
    np.savez(p, **arrays)
    return p


def test_booleanize_writes_train_data_compatible_format(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.integers(0, 17, size=(20, 3))
    x_test = rng.integers(0, 17, size=(5, 3))
    y_train = rng.integers(0, 2, size=20)
    y_test = rng.integers(0, 2, size=5)
    npz = _make_raw_npz(tmp_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    cfg = BooleanisationConfig(
        raw_npz=npz, name="synth", output_dir=tmp_path / "out",
        default_encoder=FeatureEncoderSpec(encoder="thermometer", bits=8, range=(0, 16)),
    )
    report = run_booleanize(cfg)

    assert report.n_train == 20 and report.n_test == 5
    assert report.n_features_bool == 24   # 3 columns x 8 bits

    # This is exactly the format matador.models.trainer.load_data() parses:
    # space-separated uint, last column = label.
    train_txt = np.genfromtxt(report.output_paths["train"], delimiter=" ", dtype=np.uint32)
    assert train_txt.shape == (20, 25)
    assert set(np.unique(train_txt[:, :-1]).tolist()) <= {0, 1}   # every feature column is Boolean
    assert (train_txt[:, -1] == y_train).all()

    report_json = report.output_paths["report"]
    assert report_json.exists()


def test_booleanize_mixed_per_column_encoders_and_default(tmp_path):
    """column ranges, explicit single columns, and a default_encoder fallback
    can all coexist -- e.g. digits' "column 0-63, thermometer 8 bits" style
    plus a categorical column with its own onehot spec."""
    x_train = np.array([[0, 5, 1], [16, 5, 2], [8, 5, 3]], dtype=float)
    x_test = np.array([[4, 5, 1]], dtype=float)
    y_train, y_test = np.array([0, 1, 0]), np.array([1])
    npz = _make_raw_npz(tmp_path, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    cfg = BooleanisationConfig(
        raw_npz=npz, name="mixed", output_dir=tmp_path / "out",
        features=[
            FeatureEncoderSpec(column="0-0", encoder="thermometer", bits=4, range=(0, 16)),
            FeatureEncoderSpec(column=2, encoder="onehot", categories=[1, 2, 3]),
        ],
        default_encoder=FeatureEncoderSpec(encoder="passthrough"),
    )
    report = run_booleanize(cfg)
    # col0: 4 thermometer bits, col1: 1 passthrough bit (default), col2: 3 onehot bits
    assert report.n_features_bool == 4 + 1 + 3
    encoders_used = {c["column"]: c["encoder"] for c in report.per_column}
    assert encoders_used == {0: "thermometer", 1: "passthrough", 2: "onehot"}


def test_booleanize_column_covered_twice_raises(tmp_path):
    x = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
    npz = _make_raw_npz(tmp_path, x_train=x, y_train=y, x_test=x, y_test=y)
    cfg = BooleanisationConfig(
        raw_npz=npz, name="dup", output_dir=tmp_path / "out",
        features=[
            FeatureEncoderSpec(column=0, encoder="passthrough"),
            FeatureEncoderSpec(column="0-1", encoder="passthrough"),
        ],
    )
    with pytest.raises(ValueError, match="more than one"):
        run_booleanize(cfg)


def test_booleanize_uncovered_column_without_default_raises(tmp_path):
    x = np.zeros((4, 2)); y = np.array([0, 1, 0, 1])
    npz = _make_raw_npz(tmp_path, x_train=x, y_train=y, x_test=x, y_test=y)
    cfg = BooleanisationConfig(
        raw_npz=npz, name="uncovered", output_dir=tmp_path / "out",
        features=[FeatureEncoderSpec(column=0, encoder="passthrough")],
    )
    with pytest.raises(ValueError, match="not covered"):
        run_booleanize(cfg)


def test_booleanize_splits_unsplit_raw_xy(tmp_path):
    """When raw_npz only has x/y (no pre-existing split -- e.g. a user's own
    raw data fed straight in, bypassing matador ingest), booleanize applies
    its own seeded split."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, size=(50, 2)).astype(float)
    y = rng.integers(0, 2, size=50)
    npz = _make_raw_npz(tmp_path, x=x, y=y)

    cfg = BooleanisationConfig(
        raw_npz=npz, name="unsplit", output_dir=tmp_path / "out",
        default_encoder=FeatureEncoderSpec(encoder="passthrough"),
        test_size=0.2, seed=0, stratify=True,
    )
    report = run_booleanize(cfg)
    assert report.n_train + report.n_test == 50
    assert 5 <= report.n_test <= 15


# ---------------------------------------------------------------------------
# Network-gated end-to-end: real digits ingest -> booleanize
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("MATADOR_NETWORK_TESTS") != "1",
    reason="set MATADOR_NETWORK_TESTS=1 to run real-network ingestion tests",
)
def test_booleanize_digits_end_to_end_matches_committed_shape(tmp_path):
    """Real ingest('digits') -> booleanize with the catalog's own documented
    encoding ("64 features x 8 bits, values 0..16 thermometer") -- checked
    against data/Digits/*.txt (already committed). Byte-exact row
    reproduction isn't achievable (the committed copy's exact random split
    and bit-threshold convention were never recorded — see the catalog's
    own notes), but shape, bit width, and internal thermometer consistency
    are real, verifiable correctness properties."""
    from matador.preprocessing.ingest import run_ingest
    from matador.preprocessing.sources import RawDataCatalog

    catalog = RawDataCatalog.model_validate(yaml.safe_load((REPO_ROOT / "data" / "Raw_Data_Bank.yaml").read_text()))
    spec = catalog.get("digits")
    ingest_report = run_ingest(spec, tmp_path / "raw", defaults=catalog.defaults)

    cfg = BooleanisationConfig(
        raw_npz=ingest_report.output_paths["npz"], name="digits", output_dir=tmp_path / "bool",
        default_encoder=FeatureEncoderSpec(encoder="thermometer", bits=8, range=(0, 16)),
    )
    report = run_booleanize(cfg)

    committed_train = np.genfromtxt(REPO_ROOT / "data" / "Digits" / "Digits_train.txt", dtype=np.uint32)
    committed_test = np.genfromtxt(REPO_ROOT / "data" / "Digits" / "Digits_test.txt", dtype=np.uint32)

    assert (report.n_train, report.n_features_bool + 1) == committed_train.shape
    assert (report.n_test, report.n_features_bool + 1) == committed_test.shape
    assert report.n_features_bool == 512   # 64 features x 8 bits, per the catalog's own encoding note

    mine_train = np.genfromtxt(report.output_paths["train"], dtype=np.uint32)
    mine_classes, mine_counts = np.unique(mine_train[:, -1], return_counts=True)
    committed_classes, committed_counts = np.unique(committed_train[:, -1], return_counts=True)
    assert mine_classes.tolist() == committed_classes.tolist() == list(range(10))
    assert np.all(np.abs(mine_counts - committed_counts) <= 15)   # same ballpark stratified split

    for block_start in range(0, 512, 8):
        block = mine_train[:, block_start:block_start + 8]
        assert np.all(np.diff(block.astype(int), axis=1) <= 0)   # thermometer: 1s form a prefix
