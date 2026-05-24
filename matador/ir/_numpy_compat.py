"""Pydantic v2 + NumPy interop.

Provides ``NumpyArray``, an Annotated type that:
  - Accepts ``np.ndarray`` values directly (pass-through).
  - Accepts ``{"dtype": str, "shape": [int, ...], "data_b64": str}`` dicts
    (used when loading from JSON / YAML).
  - Serialises to that same dict when ``model_dump(mode="json")`` or
    ``model_dump_json()`` is called; leaves arrays as-is for
    ``model_dump()`` (Python mode).
"""

from __future__ import annotations

import base64
from typing import Annotated, Any

import numpy as np
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import PlainValidator


def _validate_numpy(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        return v
    if isinstance(v, dict) and "data_b64" in v:
        dtype = np.dtype(v["dtype"])
        shape = tuple(v["shape"])
        raw = base64.b64decode(v["data_b64"])
        return np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    raise ValueError(
        f"Cannot coerce {type(v).__name__!r} to np.ndarray — "
        "expected ndarray or {dtype, shape, data_b64} dict"
    )


def _serialize_numpy_json(v: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": str(v.dtype),
        "shape": list(v.shape),
        "data_b64": base64.b64encode(v.tobytes()).decode(),
    }


NumpyArray = Annotated[
    np.ndarray,
    PlainValidator(_validate_numpy),
    PlainSerializer(_serialize_numpy_json, when_used="json"),
]
