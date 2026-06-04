"""Training logic for Tsetlin Machine models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from matador.ir.tm_ir import EvalVector, Verification
from matador.models.tmu_adapter import from_tmu, import_tmu_classifier

if TYPE_CHECKING:
    from matador.config.schema import TrainingConfig

_LOGGER = logging.getLogger(__name__)


def _import_tmu():
    return import_tmu_classifier()


def load_data(config: TrainingConfig) -> dict[str, np.ndarray]:
    """Load and split space-separated Boolean dataset files."""
    train_raw = np.genfromtxt(config.train_data, delimiter=" ", dtype=np.uint32)
    test_raw = np.genfromtxt(config.test_data, delimiter=" ", dtype=np.uint32)

    return {
        "x_train": train_raw[:, :-1],
        "y_train": train_raw[:, -1],
        "x_test": test_raw[:, :-1],
        "y_test": test_raw[:, -1],
    }


def train_model(config: TrainingConfig, data: dict[str, np.ndarray]):
    """Train a TMClassifier and return the fitted model."""
    TMClassifier, BenchmarkTimer = _import_tmu()

    tm = TMClassifier(
        type_iii_feedback=False,
        number_of_clauses=config.clauses,
        T=config.T,
        s=config.s,
        max_included_literals=config.max_included_literals,
        weighted_clauses=False,
        seed=config.seed,
    )

    for epoch in range(config.epochs):
        timer_epoch = BenchmarkTimer(logger=None, text="Epoch Time")
        with timer_epoch:
            timer_train = BenchmarkTimer(logger=None, text="Training Time")
            with timer_train:
                tm.fit(data["x_train"], data["y_train"], metrics=["update_p"])

            timer_test = BenchmarkTimer(logger=None, text="Testing Time")
            with timer_test:
                accuracy = 100 * (tm.predict(data["x_test"]) == data["y_test"]).mean()

        _LOGGER.info(
            "Epoch %d/%d — accuracy: %.2f%%  train: %.2fs  test: %.2fs",
            epoch + 1,
            config.epochs,
            accuracy,
            timer_train.elapsed(),
            timer_test.elapsed(),
        )

    return tm


def _embed_test_vectors(tmir, config: "TrainingConfig", n: int = 10) -> None:
    """Sample the first *n* rows of the test set and embed them into tmir.verification."""
    from matador.inference.reference import predict

    raw = np.genfromtxt(config.test_data, delimiter=" ", dtype=np.uint32)
    n = min(n, len(raw))
    X = raw[:n, :-1].astype(np.uint8)
    preds, scores = predict(tmir, X)
    tmir.verification = Verification(
        test_vectors=[
            EvalVector(
                input=list(int(v) for v in X[i]),
                expected_class=int(preds[i]),
                expected_scores=[int(scores[i, j]) for j in range(scores.shape[1])],
            )
            for i in range(n)
        ]
    )
    _LOGGER.debug("Embedded %d test vectors into TMIR", n)


def export_tmir(tm, config: TrainingConfig) -> tuple[Path, Path, Path]:
    """Convert a trained TMU model to TMIR, validate, and write to disk.

    Files are written to ``<output_dir>/TMIR/``:
      - ``<stem>.yaml``              — human-readable model
      - ``<stem>.npz``               — compact binary, preferred for tooling
      - ``validation_config.yaml``   — ready-to-use config for ``matador validate``

    Returns:
        (yaml_path, npz_path, validation_config_path)

    Raises:
        ValueError: if the resulting TMIR fails self-validation.
    """
    import yaml as _yaml

    tmir_dir = config.output_dir / "TMIR"
    tmir_dir.mkdir(parents=True, exist_ok=True)

    s_tag = str(int(config.s)) if config.s == int(config.s) else str(config.s)
    stem = (
        f"TM_TMIR"
        f"_Clauses_{config.clauses}"
        f"_s_value_{s_tag}"
        f"_T_value_{config.T}"
        f"_epochs_{config.epochs}"
        f"_max_literals_{config.max_included_literals}"
    )

    tmir = from_tmu(tm)

    # Patch provenance with the actual epoch count from the training run.
    tmir.provenance.epochs = config.epochs

    # Embed the first N test samples as ground-truth eval vectors so the TMIR
    # is self-validating without needing the test dataset on disk.
    _embed_test_vectors(tmir, config, n=10)

    tmir.validate_self()
    _LOGGER.debug("TMIR self-validation passed")

    yaml_path = tmir_dir / f"{stem}.yaml"
    npz_path = tmir_dir / f"{stem}.npz"

    tmir.to_yaml(yaml_path)
    tmir.to_npz(npz_path)

    # Write a validation config that points straight at this model and the
    # test dataset used during training, so the user can immediately run:
    #   matador validate --config <validation_config_path>
    val_cfg = {"model_path": str(yaml_path), "test_data": str(config.test_data)}
    val_cfg_path = tmir_dir / "validation_config.yaml"
    val_cfg_path.write_text(_yaml.dump(val_cfg, default_flow_style=False, sort_keys=False))

    # Write standalone provenance script + companion data files so the
    # model can be verified without a matador installation.
    from matador.inference.script_gen import write_provenance_artifacts
    script_path = write_provenance_artifacts(tmir, tmir_dir, test_data_path=config.test_data)

    _LOGGER.info("TMIR written to %s  (+.npz)", yaml_path)
    _LOGGER.info("Validation config written to %s", val_cfg_path)
    _LOGGER.info("Provenance script written to %s", script_path)
    return yaml_path, npz_path, val_cfg_path
