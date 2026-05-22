"""Training logic for Tsetlin Machine models."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matador.config.schema import TrainingConfig

_LOGGER = logging.getLogger(__name__)


def _import_tmu():
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
        from tmu.tools import BenchmarkTimer

        return TMClassifier, BenchmarkTimer
    except ImportError as exc:
        raise SystemExit(
            "The 'tmu' package is required for training but is not installed.\n"
            "Inside the container run:  pip install tmu\n"
            "Or rebuild with the ml extra:  pip install 'matador[ml]'"
        ) from exc


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


def export_ta_states(tm, config: TrainingConfig) -> Path:
    """Write TA state file to output_dir; return the file path."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    s_tag = str(int(config.s)) if config.s == int(config.s) else str(config.s)
    filename = (
        f"TM_TA_states"
        f"_Clauses_{config.clauses}"
        f"_s_value_{s_tag}"
        f"_T_value_{config.T}"
        f"_epochs_{config.epochs}"
        f"_max_literals_{config.max_included_literals}"
    )
    out_path = config.output_dir / filename

    clauses_half = config.clauses // 2

    with out_path.open("w") as fh:
        for cls in range(config.classes):
            for j in range(clauses_half):
                for polarity in (0, 1):
                    tas = [
                        int(tm.get_ta_action(j, k, the_class=cls, polarity=polarity))
                        for k in range(config.features * 2)
                    ]
                    for feat in range(config.features):
                        fh.write(f"{tas[feat]} {tas[config.features + feat]} ")

    _LOGGER.info("TA states written to %s", out_path)
    return out_path
