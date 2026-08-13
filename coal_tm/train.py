"""Train a Coalesced TM and export the TAs.txt/weights.txt that coal_tm.rtl
consumes.

The old flow (MATADOR_main.py's train_model()) fit the model and printed
accuracy but never exported anything — the README documented the export as
a by-hand snippet the developer had to run themselves afterwards, so
chaining train -> generate never actually worked. Here export is just the
last step of training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier
from tmu.tools import BenchmarkTimer

from coal_tm import artifacts
from coal_tm.config import TrainingConfig


@dataclass
class TrainResult:
    accuracy: float
    tas_path: Path
    weights_path: Path


def _load_dataset(config: TrainingConfig) -> dict[str, np.ndarray]:
    train_raw = np.genfromtxt(config.training_data, delimiter=" ")
    test_raw = np.genfromtxt(config.test_data, delimiter=" ")

    X_train, Y_train = train_raw[:, :-1], train_raw[:, -1]
    X_test, Y_test = test_raw[:, :-1], test_raw[:, -1]

    if X_train.shape[1] != config.features:
        raise ValueError(
            f"{config.training_data} has {X_train.shape[1]} feature columns, "
            f"config says features={config.features}"
        )
    if X_test.shape[1] != config.features:
        raise ValueError(
            f"{config.test_data} has {X_test.shape[1]} feature columns, "
            f"config says features={config.features}"
        )

    return {
        "X_train": X_train.astype(np.uint32),
        "Y_train": Y_train.astype(np.uint32),
        "X_test": X_test.astype(np.uint32),
        "Y_test": Y_test.astype(np.uint32),
    }


def train(config: TrainingConfig) -> TrainResult:
    data = _load_dataset(config)

    tm = TMCoalescedClassifier(
        type_iii_feedback=False,
        number_of_clauses=config.clauses,
        T=config.T,
        s=config.s,
        max_included_literals=config.max_included_literals,
        weighted_clauses=True,
        seed=config.seed,
    )

    accuracy = 0.0
    for epoch in range(config.epochs):
        timer = BenchmarkTimer(logger=None, text="Epoch Time")
        with timer:
            tm.fit(data["X_train"], data["Y_train"])
            accuracy = float(100 * (tm.predict(data["X_test"]) == data["Y_test"]).mean())
        print(f"[train] epoch {epoch + 1}/{config.epochs}  accuracy={accuracy:.2f}%  ({timer.elapsed():.2f}s)")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    tas_path = config.output_dir / "TAs.txt"
    weights_path = config.output_dir / "weights.txt"
    artifacts.write_tas(tas_path, tm, config.clauses, config.features)
    artifacts.write_weights(weights_path, tm, config.classes, config.clauses)
    print(f"[train] wrote {tas_path}")
    print(f"[train] wrote {weights_path}")

    return TrainResult(accuracy=accuracy, tas_path=tas_path, weights_path=weights_path)
