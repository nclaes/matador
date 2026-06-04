"""ROM-based provenance report generation.

Runs the full held-out test dataset through the pure-NumPy reference
inference engine (identical algorithm to the RTL ROM) and produces a
signed JSON report suitable for sharing with hardware collaborators.

The "ROM" referred to here is representation.includes — the boolean
Include-action matrix that the RTL tile ROM stores.  Inference never
touches the trained TMU object or the tmu C extension.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PerClassResult:
    correct: int
    total: int
    accuracy_pct: float


@dataclass
class ProvenanceReport:
    # Identity
    generated_at: str
    matador_version: str
    python_version: str
    platform: str

    # Model
    model_path: str
    model_fingerprint: str
    model_variant: str
    n_features: int
    n_classes: int
    n_clauses_total: int
    n_clauses_per_class: Optional[int]
    hyperparameters: dict

    # Training provenance (from TMIR if present)
    trained_at: Optional[str]
    training_epochs: Optional[int]
    framework: Optional[str]
    framework_version: Optional[str]

    # Dataset
    test_data_path: str
    test_data_sha256: str
    n_samples: int

    # Inference engine
    inference_engine: str

    # Results
    accuracy_pct: float
    accuracy_correct: int
    accuracy_total: int
    per_class: dict  # class_label -> PerClassResult as dict
    confusion_matrix: list  # list[list[int]]

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=str)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def generate(
    model_path: Path,
    test_data_path: Path,
) -> ProvenanceReport:
    """Load a TMIR model and run full-dataset ROM-based inference.

    Args:
        model_path:     Path to a TMIR .yaml or .npz file.
        test_data_path: Path to a space-separated Boolean dataset
                        (last column = integer class label, 0-indexed).

    Returns:
        ProvenanceReport — serialisable to JSON.
    """
    from matador import __version__
    from matador.inference.reference import predict
    from matador.ir.tm_ir import TMIR

    # ── Load model ────────────────────────────────────────────────────────────
    suffix = model_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        tmir = TMIR.from_yaml(model_path)
    else:
        tmir = TMIR.from_npz(model_path)

    tmir.validate_self()

    # Materialise includes so fingerprint is deterministic regardless of
    # whether the TMIR was saved with ta_states or includes.
    tmir.to_includes()

    arch = tmir.architecture
    hyper = tmir.hyperparameters
    prov = tmir.provenance

    # ── Load test data ────────────────────────────────────────────────────────
    raw = np.genfromtxt(test_data_path, delimiter=" ", dtype=np.uint32)
    X_test = raw[:, :-1].astype(np.uint8)
    y_test = raw[:, -1].astype(np.int64)
    n_samples = len(y_test)

    # Dataset fingerprint
    data_bytes = test_data_path.read_bytes()
    data_sha256 = hashlib.sha256(data_bytes).hexdigest()

    # ── Inference — from the ROM (Include-action matrix) ─────────────────────
    preds, scores = predict(tmir, X_test)
    preds = preds.astype(np.int64)

    correct_mask = preds == y_test
    accuracy_correct = int(correct_mask.sum())
    accuracy_pct = 100.0 * accuracy_correct / n_samples if n_samples > 0 else 0.0

    # Per-class breakdown
    classes = sorted(int(c) for c in np.unique(y_test))
    per_class: dict = {}
    for cls in classes:
        mask = y_test == cls
        cls_total = int(mask.sum())
        cls_correct = int((correct_mask & mask).sum())
        cls_pct = 100.0 * cls_correct / cls_total if cls_total > 0 else 0.0
        per_class[str(cls)] = asdict(PerClassResult(
            correct=cls_correct,
            total=cls_total,
            accuracy_pct=round(cls_pct, 4),
        ))

    # Confusion matrix
    n_classes = arch.n_classes
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(y_test, preds):
        if 0 <= int(true) < n_classes and 0 <= int(pred) < n_classes:
            cm[int(true), int(pred)] += 1

    return ProvenanceReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        matador_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),

        model_path=str(model_path.resolve()),
        model_fingerprint=tmir.fingerprint(),
        model_variant=tmir.variant,
        n_features=arch.n_features,
        n_classes=arch.n_classes,
        n_clauses_total=arch.n_clauses_total,
        n_clauses_per_class=arch.n_clauses_per_class,
        hyperparameters={
            "s": hyper.s,
            "T": arch.threshold,
            "n_states": hyper.n_states,
            "seed": hyper.seed,
        },

        trained_at=prov.trained_at.isoformat() if prov else None,
        training_epochs=prov.epochs if prov else None,
        framework=prov.framework if prov else None,
        framework_version=prov.framework_version if prov else None,

        test_data_path=str(test_data_path.resolve()),
        test_data_sha256=data_sha256,
        n_samples=n_samples,

        inference_engine=(
            "matador.inference.reference — pure NumPy, ROM-equivalent "
            "(Include-action matrix, no tmu dependency)"
        ),

        accuracy_pct=round(accuracy_pct, 4),
        accuracy_correct=accuracy_correct,
        accuracy_total=n_samples,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )
