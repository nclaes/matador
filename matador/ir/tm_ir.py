"""Tsetlin Machine Intermediate Representation (TMIR) — canonical schema.

See docs/tmir_spec.md for the full normative conventions.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from matador.ir._numpy_compat import NumpyArray


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


# ---------------------------------------------------------------------------
# Nested schemas
# ---------------------------------------------------------------------------

class Architecture(_Base):
    n_features: int
    n_literals: int
    n_classes: int
    n_clauses_per_class: Optional[int] = None
    n_clauses_total: int
    clause_organization: Literal["per_class", "coalesced"]
    threshold: int

    @field_validator("n_literals", mode="after")
    @classmethod
    def _check_literals(cls, v: int, info) -> int:
        n_features = info.data.get("n_features")
        if n_features is not None and v != 2 * n_features:
            raise ValueError(
                f"n_literals ({v}) must equal 2 * n_features ({n_features})"
            )
        return v


class Hyperparameters(_Base):
    s: float
    n_states: int = 256
    seed: Optional[int] = None


class CompressedPayload(_Base):
    scheme: Literal["redress", "rle", "bitpacked"]
    block_size: Optional[int] = None
    payload: bytes
    decompressed_shape: tuple


class Representation(_Base):
    ta_states: Optional[NumpyArray] = None
    includes: Optional[NumpyArray] = None
    compressed: Optional[CompressedPayload] = None

    @model_validator(mode="after")
    def _at_least_one(self) -> "Representation":
        if (
            self.ta_states is None
            and self.includes is None
            and self.compressed is None
        ):
            raise ValueError(
                "Representation requires at least one of: "
                "ta_states, includes, compressed"
            )
        return self


class WeightRange(_Base):
    min: int
    max: int
    abs_max: int


class Weights(_Base):
    values: NumpyArray
    signed: bool
    bit_width: int
    range: WeightRange


class Derived(_Base):
    score_accumulator_width: int
    avg_clause_length: float
    max_clause_length: int
    min_clause_length: int
    clause_length_distribution: dict[int, int]
    empty_clause_count: int


class PreprocessingFingerprint(_Base):
    name: str
    config: dict[str, Any] = {}
    fingerprint: str


class EvalVector(_Base):
    """One inference test vector stored inside a TMIR Verification block."""
    input: list[int]
    expected_class: int
    expected_scores: list[int]


# Alias so external code can use the spec-mandated name without triggering
# pytest's "test class" collection heuristic.
TestVector = EvalVector


class Verification(_Base):
    test_vectors: list[EvalVector] = []
    software_reference_hash: str = ""


class Provenance(_Base):
    framework: str
    framework_version: str
    dataset_id: Optional[str] = None
    trained_at: datetime
    epochs: int
    hyperparameter_log: dict[str, Any] = {}


class Telemetry(_Base):
    clause_activation_frequency: Optional[NumpyArray] = None
    per_class_clause_usage: Optional[NumpyArray] = None
    marginality: Optional[NumpyArray] = None


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

class TMIR(_Base):
    tmir_version: Literal["0.2"] = "0.2"
    variant: Literal[
        "vanilla", "weighted", "coalesced",
        "convolutional", "regression", "graph",
    ]
    architecture: Architecture
    hyperparameters: Hyperparameters
    representation: Representation
    weights: Optional[Weights] = None
    derived: Optional[Derived] = None
    preprocessing: Optional[PreprocessingFingerprint] = None
    verification: Optional[Verification] = None
    provenance: Optional[Provenance] = None
    telemetry: Optional[Telemetry] = None

    # Compute derived stats lazily on first construction if absent.
    @model_validator(mode="after")
    def _auto_derived(self) -> "TMIR":
        if self.derived is not None:
            return self
        inc = self._resolve_includes()
        if inc is None:
            return self
        self.derived = _compute_derived(self, inc)
        return self

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def to_includes(self) -> None:
        """Derive ``representation.includes`` from ``ta_states`` in-place.

        Idempotent — does nothing if includes is already present.
        """
        if self.representation.includes is not None:
            return
        if self.representation.ta_states is None:
            raise ValueError(
                "Cannot derive includes: ta_states is absent and no includes present"
            )
        threshold = self.hyperparameters.n_states // 2
        self.representation.includes = (
            self.representation.ta_states > threshold
        )

    def validate_self(self) -> None:
        """Cross-field consistency checks; raises ValueError on violation."""
        arch = self.architecture
        rep = self.representation

        # Literal count
        if arch.n_literals != 2 * arch.n_features:
            raise ValueError(
                f"n_literals={arch.n_literals} != 2*n_features={arch.n_features}"
            )

        # Per-class clause totals
        if arch.clause_organization == "per_class":
            if arch.n_clauses_per_class is None:
                raise ValueError(
                    "n_clauses_per_class is required for per_class organisation"
                )
            expected = arch.n_classes * arch.n_clauses_per_class
            if arch.n_clauses_total != expected:
                raise ValueError(
                    f"n_clauses_total={arch.n_clauses_total} != "
                    f"n_classes*n_clauses_per_class={expected}"
                )
        else:  # coalesced
            if self.weights is None:
                raise ValueError(
                    "weights are required for coalesced clause_organization"
                )

        # Shape checks on ta_states
        if rep.ta_states is not None:
            _check_ta_shape(rep.ta_states, arch)

        # Shape checks on includes
        if rep.includes is not None:
            _check_ta_shape(rep.includes, arch)

        # Weights shape
        if self.weights is not None:
            _check_weight_shape(self.weights.values, arch)

    def fingerprint(self) -> str:
        """SHA-256 fingerprint over the canonical model definition."""
        canonical = self.model_dump_json(
            include={
                "variant", "architecture", "hyperparameters", "representation"
            }
        )
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        return f"sha256:{digest}"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible Python dict (numpy arrays as b64 dicts)."""
        return json.loads(self.model_dump_json())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TMIR":
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """Write model to a YAML file (arrays serialised as base64).

        Uses ruamel.yaml when available; falls back to pyyaml otherwise.
        """
        d = self.to_dict()
        try:
            from ruamel.yaml import YAML
            yml = YAML()
            yml.default_flow_style = False
            with open(path, "w") as fh:
                yml.dump(d, fh)
        except ImportError:
            import yaml
            with open(path, "w") as fh:
                yaml.dump(d, fh, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TMIR":
        """Load model from a YAML file."""
        try:
            from ruamel.yaml import YAML
            yml = YAML()
            with open(path) as fh:
                data = yml.load(fh)
            return cls.from_dict(_ruamel_to_plain(data))
        except ImportError:
            import yaml
            with open(path) as fh:
                data = yaml.safe_load(fh)
            return cls.from_dict(data)

    def to_npz(self, path: str | Path) -> None:
        """Save model to compressed .npz (arrays stored raw, metadata as JSON).

        This format is preferred for large models — arrays are stored without
        base64 overhead, so it is smaller and faster than YAML.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        arrays: dict[str, np.ndarray] = {}
        meta = _extract_arrays_for_npz(self.model_dump(), arrays, prefix="")
        meta_bytes = json.dumps(meta, default=_json_default).encode()
        arrays["__meta__"] = np.frombuffer(meta_bytes, dtype=np.uint8)
        np.savez_compressed(str(path.with_suffix("")), **arrays)

    @classmethod
    def from_npz(cls, path: str | Path) -> "TMIR":
        """Load model from a .npz file saved with :meth:`to_npz`."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")

        with np.load(str(path), allow_pickle=False) as f:
            meta_bytes = bytes(f["__meta__"])
            meta = json.loads(meta_bytes.decode())
            data = _restore_arrays_from_npz(meta, f)
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_includes(self) -> Optional[np.ndarray]:
        """Return the best available boolean include mask, or None."""
        rep = self.representation
        if rep.ta_states is not None:
            thresh = self.hyperparameters.n_states // 2
            return rep.ta_states > thresh
        if rep.includes is not None:
            return rep.includes
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ta_shape(arr: np.ndarray, arch: Architecture) -> None:
    if arch.clause_organization == "per_class":
        expected = (
            arch.n_classes,
            arch.n_clauses_per_class,
            arch.n_literals,
        )
        if arr.shape != expected:
            raise ValueError(
                f"ta_states/includes shape {arr.shape} != expected {expected} "
                f"for per_class organisation"
            )
    else:
        expected = (arch.n_clauses_total, arch.n_literals)
        if arr.shape != expected:
            raise ValueError(
                f"ta_states/includes shape {arr.shape} != expected {expected} "
                f"for coalesced organisation"
            )


def _check_weight_shape(values: np.ndarray, arch: Architecture) -> None:
    if arch.clause_organization == "per_class":
        expected = (arch.n_classes, arch.n_clauses_per_class)
    else:
        expected = (arch.n_classes, arch.n_clauses_total)
    if values.shape != expected:
        raise ValueError(
            f"weights.values shape {values.shape} != expected {expected}"
        )


def _compute_derived(tmir: TMIR, includes: np.ndarray) -> Derived:
    """Compute Derived statistics from the boolean include mask."""
    # Flatten to 2-D: (total_clauses, n_literals)
    flat = includes.reshape(-1, includes.shape[-1])
    clause_lengths = flat.sum(axis=1).astype(int)

    dist: dict[int, int] = {}
    for length in clause_lengths:
        dist[int(length)] = dist.get(int(length), 0) + 1

    arch = tmir.architecture
    abs_weight_max = 1
    if tmir.weights is not None:
        abs_weight_max = max(1, int(tmir.weights.range.abs_max))

    if arch.clause_organization == "per_class" and arch.n_clauses_per_class:
        acc_raw = arch.n_clauses_per_class * abs_weight_max
    else:
        acc_raw = arch.n_clauses_total * abs_weight_max

    acc_width = max(1, math.ceil(math.log2(acc_raw + 1))) + 1

    return Derived(
        score_accumulator_width=acc_width,
        avg_clause_length=float(np.mean(clause_lengths)),
        max_clause_length=int(clause_lengths.max()),
        min_clause_length=int(clause_lengths.min()),
        clause_length_distribution=dist,
        empty_clause_count=int((clause_lengths == 0).sum()),
    )


def _extract_arrays_for_npz(
    obj: Any, arrays: dict[str, np.ndarray], prefix: str
) -> Any:
    """Walk a model_dump() result, lift out numpy arrays into ``arrays``."""
    if isinstance(obj, np.ndarray):
        key = prefix.lstrip("_").replace(".", "_").replace("[", "_").replace("]", "")
        # Ensure unique keys
        base_key = key or "arr"
        k = base_key
        n = 0
        while k in arrays:
            n += 1
            k = f"{base_key}_{n}"
        arrays[k] = obj
        return {"__npz__": k}
    if isinstance(obj, dict):
        return {
            kk: _extract_arrays_for_npz(vv, arrays, f"{prefix}.{kk}" if prefix else kk)
            for kk, vv in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [
            _extract_arrays_for_npz(item, arrays, f"{prefix}[{i}]")
            for i, item in enumerate(obj)
        ]
    return obj


def _restore_arrays_from_npz(obj: Any, npz: Any) -> Any:
    if isinstance(obj, dict):
        if "__npz__" in obj:
            return npz[obj["__npz__"]]
        return {k: _restore_arrays_from_npz(v, npz) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays_from_npz(v, npz) for v in obj]
    return obj


def _ruamel_to_plain(obj: Any) -> Any:
    """Recursively convert ruamel.yaml CommentedMap/Seq to plain dict/list."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    if isinstance(obj, CommentedMap):
        return {k: _ruamel_to_plain(v) for k, v in obj.items()}
    if isinstance(obj, CommentedSeq):
        return [_ruamel_to_plain(v) for v in obj]
    return obj


def _json_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
