"""Tests for matador.preprocessing — raw-data ingestion (Stage 1 of the
booleanization pipeline: raw data -> matador booleanize -> matador train).

Covers every distinct (source.kind, extract.archive, parse.reader,
split.mode) combination actually used in data/Raw_Data_Bank.yaml, using
small synthetic fixtures (no network) plus one real network-gated
end-to-end test against the already-committed data/Digits/*.txt.
"""

from __future__ import annotations

import gzip
import os
import pickle
import struct
import tarfile
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from matador.preprocessing.ingest import run_ingest
from matador.preprocessing.sources import RawDataCatalog, RawDataSourceSpec, resolve_source_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "data" / "Raw_Data_Bank.yaml"


# ---------------------------------------------------------------------------
# Schema / config resolution
# ---------------------------------------------------------------------------

def test_real_catalog_validates():
    raw = yaml.safe_load(CATALOG_PATH.read_text())
    catalog = RawDataCatalog.model_validate(raw)
    assert len(catalog.datasets) == 11
    assert {d.key for d in catalog.datasets} == {
        "digits", "sports", "statlog", "gesture_phase", "human_activity",
        "mammographic", "emg", "sensorless_drive", "mnist", "cifar2", "kws2",
    }


def test_resolve_source_config_standalone_file(tmp_path):
    p = tmp_path / "data_source_config.yaml"
    p.write_text(yaml.safe_dump({
        "key": "standalone",
        "source": {"kind": "local", "path": str(tmp_path)},
        "parse": {"reader": "csv"},
        "split": {"mode": "random"},
    }))
    spec, defaults = resolve_source_config(p)
    assert spec.key == "standalone"
    assert defaults.retries == 3  # library default, since no catalog defaults were given


def test_resolve_source_config_catalog_pointer(tmp_path):
    p = tmp_path / "data_source_config.yaml"
    p.write_text(yaml.safe_dump({"catalog": str(CATALOG_PATH), "key": "digits"}))
    spec, defaults = resolve_source_config(p)
    assert spec.key == "digits"
    assert spec.parse.reader == "csv"


def test_resolve_source_config_rejects_bare_catalog(tmp_path):
    p = tmp_path / "data_source_config.yaml"
    p.write_text(CATALOG_PATH.read_text())
    with pytest.raises(ValueError, match="multi-dataset catalog"):
        resolve_source_config(p)


# ---------------------------------------------------------------------------
# Random split + plain csv reader (digits/statlog/mammographic/... shape)
# ---------------------------------------------------------------------------

def test_ingest_local_csv_random_split_stratified(tmp_path):
    csvf = tmp_path / "d.csv"
    rng = np.random.default_rng(0)
    rows = [",".join(map(str, rng.integers(0, 10, 4).tolist())) + f",{i % 3}" for i in range(100)]
    csvf.write_text("\n".join(rows))

    spec = RawDataSourceSpec.model_validate({
        "key": "t1", "source": {"kind": "local", "path": str(csvf)}, "extract": {"archive": "none"},
        "parse": {"reader": "csv", "delimiter": ",", "header": False, "label_column": -1, "dtype": "float32"},
        "split": {"mode": "random", "test_size": 0.2, "seed": 0, "stratify": True},
        "expect": {"n_rows": 100, "n_classes": 3},
        "export": {"formats": ["npz", "csv"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")

    assert report.x_train.shape[1] == 4
    assert report.x_train.shape[0] + report.x_test.shape[0] == 100
    assert 15 <= report.x_test.shape[0] <= 25
    assert not report.warnings
    npz = np.load(report.output_paths["npz"])
    assert set(npz.files) == {"x_train", "x_test", "y_train", "y_test"}
    assert report.output_paths["csv_train"].exists()
    assert report.output_paths["csv_test"].exists()


def test_ingest_csv_header_label_map_na_drop_columns(tmp_path):
    """Mammographic-style: header row, '?' missing values, string label_map,
    dropping a named column."""
    csvf = tmp_path / "b.csv"
    csvf.write_text("birads,age,shape,severity\n1,60,?,yes\n2,50,3,no\n")
    spec = RawDataSourceSpec.model_validate({
        "key": "t2", "source": {"kind": "local", "path": str(csvf)}, "extract": {"archive": "none"},
        "parse": {"reader": "csv", "delimiter": ",", "header": True, "na_values": ["?"], "label_column": -1,
                  "label_map": {"yes": 1, "no": 0}, "drop_columns": ["birads"], "dtype": "float32"},
        "split": {"mode": "random", "test_size": 0.5, "seed": 0},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")
    all_y = np.concatenate([report.y_train, report.y_test])
    assert set(all_y.tolist()) == {0, 1}
    all_x = np.concatenate([report.x_train, report.x_test])
    assert all_x.shape[1] == 2   # age, shape (birads dropped, severity is the label)
    assert np.isnan(all_x).any()  # the '?' row


def test_ingest_source_member_restricts_random_split_pool(tmp_path):
    """Digits-style: two files extracted, but the random split is derived
    from only one of them (split.source_member)."""
    src_dir = tmp_path / "src"; src_dir.mkdir()
    (src_dir / "a.tra").write_text("1,1,0\n2,2,0\n")   # would corrupt the split if included
    (src_dir / "b.tes").write_text("\n".join(f"{i},{i},{i % 2}" for i in range(20)))

    spec = RawDataSourceSpec.model_validate({
        "key": "t3", "source": {"kind": "local", "path": str(src_dir)},
        "extract": {"archive": "none", "members": ["*.tra", "*.tes"]},
        "parse": {"reader": "csv", "delimiter": ",", "header": False, "label_column": -1, "dtype": "float32"},
        "split": {"mode": "random", "source_member": "b.tes", "test_size": 0.2, "seed": 0},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")
    assert report.x_train.shape[0] + report.x_test.shape[0] == 20   # only b.tes's 20 rows


# ---------------------------------------------------------------------------
# Predefined split: glob-based + label_file (human_activity shape)
# ---------------------------------------------------------------------------

def test_ingest_predefined_glob_split_with_label_file(tmp_path):
    zpath = tmp_path / "har.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("HAR/train/X_train.txt", "1 2 3\n4 5 6\n7 8 9\n")
        zf.writestr("HAR/train/y_train.txt", "1\n2\n1\n")
        zf.writestr("HAR/test/X_test.txt", "9 9 9\n")
        zf.writestr("HAR/test/y_test.txt", "2\n")

    spec = RawDataSourceSpec.model_validate({
        "key": "t4", "source": {"kind": "local", "path": str(zpath)},
        "extract": {"archive": "zip", "members": [
            "**/train/X_train.txt", "**/train/y_train.txt", "**/test/X_test.txt", "**/test/y_test.txt",
        ]},
        "parse": {"reader": "csv", "delimiter": "whitespace", "header": False, "label_file": True, "dtype": "float32"},
        "split": {"mode": "predefined", "train": ["**/train/X_train.txt"], "test": ["**/test/X_test.txt"]},
        "expect": {"n_train": 3, "n_test": 1},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")

    assert report.x_train.tolist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert report.y_train.tolist() == [0, 1, 0]   # raw labels 1,2,1 -> shifted to 0-indexed
    assert report.y_test.tolist() == [1]
    assert report.label_offset == 1
    assert report.warnings == ["labels were not 0-indexed; shifted by -1 (raw min was 1)"]


# ---------------------------------------------------------------------------
# Predefined split: role-based (mnist shape) + idx reader
# ---------------------------------------------------------------------------

def _make_idx_images(path: Path, n: int, h: int, w: int, seed: int) -> np.ndarray:
    header = (2051).to_bytes(4, "big") + n.to_bytes(4, "big") + h.to_bytes(4, "big") + w.to_bytes(4, "big")
    pix = np.random.default_rng(seed).integers(0, 255, size=(n, h, w), dtype=np.uint8)
    with gzip.open(path, "wb") as f:
        f.write(header + pix.tobytes())
    return pix.reshape(n, -1)


def _make_idx_labels(path: Path, labels: list[int]) -> None:
    header = (2049).to_bytes(4, "big") + len(labels).to_bytes(4, "big")
    with gzip.open(path, "wb") as f:
        f.write(header + bytes(labels))


def test_ingest_role_based_predefined_split_idx(tmp_path):
    src = tmp_path / "src"; src.mkdir()
    train_pix = _make_idx_images(src / "train-images-idx3-ubyte.gz", 5, 2, 2, seed=1)
    _make_idx_labels(src / "train-labels-idx1-ubyte.gz", [0, 1, 2, 0, 1])
    _make_idx_images(src / "t10k-images-idx3-ubyte.gz", 2, 2, 2, seed=2)
    _make_idx_labels(src / "t10k-labels-idx1-ubyte.gz", [1, 0])

    spec = RawDataSourceSpec.model_validate({
        "key": "t5",
        "source": {"kind": "http_files", "urls": [
            {"url": f"file://{src}/train-images-idx3-ubyte.gz", "filename": "train-images-idx3-ubyte.gz", "role": "train_X"},
            {"url": f"file://{src}/train-labels-idx1-ubyte.gz", "filename": "train-labels-idx1-ubyte.gz", "role": "train_y"},
            {"url": f"file://{src}/t10k-images-idx3-ubyte.gz", "filename": "t10k-images-idx3-ubyte.gz", "role": "test_X"},
            {"url": f"file://{src}/t10k-labels-idx1-ubyte.gz", "filename": "t10k-labels-idx1-ubyte.gz", "role": "test_y"},
        ]},
        "extract": {"archive": "gzip"}, "parse": {"reader": "idx", "dtype": "uint8"},
        "split": {"mode": "predefined", "train": ["train_X", "train_y"], "test": ["test_X", "test_y"]},
        "expect": {"n_train": 5, "n_test": 2},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out", cache_dir=tmp_path / "cache")

    assert report.y_train.tolist() == [0, 1, 2, 0, 1]
    assert report.y_test.tolist() == [1, 0]
    assert np.array_equal(report.x_train, train_pix)


# ---------------------------------------------------------------------------
# Predefined split: glob-based (cifar2 shape) + cifar_pickle reader
# ---------------------------------------------------------------------------

def test_ingest_cifar_pickle_relabel_and_greyscale(tmp_path):
    pkl_dir = tmp_path / "src" / "cifar-10-batches-py"; pkl_dir.mkdir(parents=True)

    def make_batch(path, n, labels, seed):
        data = np.random.default_rng(seed).integers(0, 255, size=(n, 3 * 32 * 32), dtype=np.uint8)
        with open(path, "wb") as f:
            pickle.dump({b"data": data, b"labels": labels}, f)

    make_batch(pkl_dir / "data_batch_1", 4, [0, 1, 8, 9], seed=3)
    make_batch(pkl_dir / "test_batch", 2, [2, 3], seed=4)
    tar_path = tmp_path / "cifar.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(pkl_dir, arcname="cifar-10-batches-py")

    spec = RawDataSourceSpec.model_validate({
        "key": "t6", "source": {"kind": "local", "path": str(tar_path)},
        "extract": {"archive": "tar.gz", "members": [
            "cifar-10-batches-py/data_batch_*", "cifar-10-batches-py/test_batch",
        ]},
        "parse": {"reader": "cifar_pickle", "dtype": "uint8", "to_greyscale": True,
                  "relabel": {0: {"vehicles": [0, 1, 8, 9]}, 1: {"animals": [2, 3, 4, 5, 6, 7]}}},
        "split": {"mode": "predefined", "train": ["data_batch_*"], "test": ["test_batch"]},
        "expect": {"n_train": 4, "n_test": 2},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")

    assert report.x_train.shape == (4, 1024)   # 32x32 greyscale, flattened
    assert report.y_train.tolist() == [0, 0, 0, 0]   # all in {0,1,8,9} -> vehicles
    assert report.y_test.tolist() == [1, 1]          # {2,3} -> animals


# ---------------------------------------------------------------------------
# Random split + file-level dir_of_txt reader (sports shape)
# ---------------------------------------------------------------------------

def test_ingest_file_level_dir_of_txt_label_from_path(tmp_path):
    zpath = tmp_path / "sports.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for act in range(1, 4):
            for seg in range(1, 4):
                content = "\n".join(f"{i} {i + 1}" for i in range(3))
                zf.writestr(f"dsa/a{act:02d}/p1/s{seg:02d}.txt", content)

    spec = RawDataSourceSpec.model_validate({
        "key": "t7", "source": {"kind": "local", "path": str(zpath)},
        "extract": {"archive": "zip", "members": ["**/a[0-9][0-9]/p[1-8]/s[0-9][0-9].txt"]},
        "parse": {"reader": "dir_of_txt", "delimiter": " ", "header": False,
                  "label_from_path": r"a(\d+)", "dtype": "float32"},
        "split": {"mode": "random", "test_size": 0.34, "seed": 0, "stratify": True},
        "expect": {"n_rows": 9, "n_classes": 3},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")

    assert report.x_train.shape[1] == 6   # 3 rows x 2 cols flattened per file
    assert report.x_train.shape[0] + report.x_test.shape[0] == 9


# ---------------------------------------------------------------------------
# Predefined split: spec_files (kws2 shape) + wav_dir reader
# ---------------------------------------------------------------------------

def _make_wav(path: Path, n_samples: int, value: int) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(16000)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([value] * n_samples)))


def test_ingest_wav_dir_spec_files_split(tmp_path):
    wdir = tmp_path / "wavsrc"
    (wdir / "yes").mkdir(parents=True)
    (wdir / "no").mkdir(parents=True)
    for i in range(4):
        _make_wav(wdir / "yes" / f"clip{i}.wav", 100, 100 + i)
    for i in range(3):
        _make_wav(wdir / "no" / f"clip{i}.wav", 100, -100 - i)
    (wdir / "testing_list.txt").write_text("yes/clip0.wav\nno/clip0.wav\n")

    spec = RawDataSourceSpec.model_validate({
        "key": "t8", "source": {"kind": "local", "path": str(wdir)}, "extract": {"archive": "none"},
        "parse": {"reader": "wav_dir", "sample_rate": 16000, "bit_depth": 16, "channels": 1,
                  "label_from_path": "^(yes|no)/"},
        "split": {"mode": "predefined", "spec_files": ["testing_list.txt"]},
        "export": {"formats": ["npz"], "path": "{output_dir}/{key}"},
    })
    report = run_ingest(spec, tmp_path / "out")

    assert report.x_train.shape[0] == 5
    assert report.x_test.shape[0] == 2
    # held-out clips (clip0 of each class) must land in test, not train
    assert report.x_train.shape[1] == report.x_test.shape[1] == 100


# ---------------------------------------------------------------------------
# Network-gated end-to-end correctness test against committed ground truth
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("MATADOR_NETWORK_TESTS") != "1",
    reason="set MATADOR_NETWORK_TESTS=1 to run real-network ingestion tests",
)
def test_ingest_digits_matches_committed_row_and_class_counts(tmp_path):
    """Real end-to-end fetch of the 'digits' catalog entry, checked against
    data/Digits/Digits_train.txt / Digits_test.txt (already committed to the
    repo) -- a true correctness check, not just "it ran"."""
    catalog = RawDataCatalog.model_validate(yaml.safe_load(CATALOG_PATH.read_text()))
    spec = catalog.get("digits")

    report = run_ingest(spec, tmp_path / "out", defaults=catalog.defaults)

    committed_train = np.genfromtxt(REPO_ROOT / "data" / "Digits" / "Digits_train.txt", dtype=np.uint32)
    committed_test = np.genfromtxt(REPO_ROOT / "data" / "Digits" / "Digits_test.txt", dtype=np.uint32)

    assert report.x_train.shape[0] == committed_train.shape[0] == 1437
    assert report.x_test.shape[0] == committed_test.shape[0] == 360
    assert len(np.unique(np.concatenate([report.y_train, report.y_test]))) == 10
