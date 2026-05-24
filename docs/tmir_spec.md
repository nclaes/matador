# TMIR Specification — version 0.2

**Tsetlin Machine Intermediate Representation**

TMIR is the canonical model representation for Tsetlin Machines in the
matador project.  It uses TMU's storage layout as the canonical convention
so that models extracted from TMU and models deployed through the matador
hardware backends share a single authoritative schema.

---

## Scope

This document is normative.  All TMIR producers (e.g., `tmu_adapter.py`)
and consumers (e.g., `inference/reference.py`, hardware back-ends) must
conform to the conventions below.

---

## Literals

For a model with **F features**, there are **L = 2F literals** per clause.
Literal index `i` within any TA-state or include array is defined as:

| Range          | Meaning                    |
|----------------|----------------------------|
| 0 ≤ i < F      | positive literal: `x[i]`    |
| F ≤ i < 2F     | negated literal: `NOT x[i−F]` |

This ordering matches TMU's internal convention.

---

## Clause Organisation

TMIR supports two clause organisations, selected by
`architecture.clause_organization`.

### `"per_class"` — Vanilla and Weighted TM

Each class owns its own clause bank.

- **TA states shape:** `(n_classes, n_clauses_per_class, n_literals)`
- **Weights shape:** `(n_classes, n_clauses_per_class)` — optional for
  vanilla (default weight = 1), required for weighted.
- **Polarity by index:** clause `k` contributes positively to its class when
  `k < n_clauses_per_class // 2`, negatively when `k ≥ half`.
- **Weights** are **non-negative** integers; the sign is carried by the
  clause index.
- **Score formula:**

  ```
  score[c] = Σ_k  (  (+1 if k < half else −1)  ×  weight[c,k]  ×  clause_out[k]  )
  ```

  where `weight[c,k]` defaults to 1 if `weights` is absent.

### `"coalesced"` — Coalesced TM

Clauses are shared across all classes; each class has its own signed weight
vector.

- **TA states shape:** `(n_clauses, n_literals)`
- **Weights shape:** `(n_classes, n_clauses)` — **REQUIRED** for coalesced.
- **Polarity by weight sign:** a positive weight is a positive vote, a
  negative weight is a negative vote.
- **Score formula:**

  ```
  score[c] = Σ_k  weight[c,k]  ×  clause_out[k]
  ```

---

## TA States

- Integer values in the range **[1, n_states]**.
- **Include decision:** literal `i` is included in clause `k` iff
  `ta_state[k, i] > n_states // 2`.
- Default `n_states = 256` (8-bit TAs, the TMU default).  The actual value
  is stored in `hyperparameters.n_states`.
- Stored as `int16` or `int32`.

---

## Weights

| Organisation | Sign of stored values    |
|--------------|--------------------------|
| `per_class`  | Non-negative (≥ 0)       |
| `coalesced`  | Signed (arbitrary)       |

Weights are always integers — no fractional quantisation.

`weights.range.abs_max` is the maximum absolute value; hardware backends
use this to size weight memory.

---

## Empty Clause Convention (Inference vs. Training)

> **At inference time, a clause with zero included literals outputs 0.**

This differs from TMU's training behaviour, where an empty clause outputs 1
so the regulator can shrink it.  Applying the training convention at
inference time causes empty (untrained) clauses to contribute spurious
votes and is a common porting bug.

All TMIR-compliant inference engines (including `inference/reference.py`)
must follow the inference convention.

---

## TMIR Schema (v0.2)

```
TMIR
├── tmir_version:     "0.2"                    (literal)
├── variant:          vanilla | weighted | coalesced
│                     | convolutional | regression | graph
├── architecture:     Architecture
│   ├── n_features:            int
│   ├── n_literals:            int  (= 2 * n_features, validated)
│   ├── n_classes:             int
│   ├── n_clauses_per_class:   int | None  (None for coalesced)
│   ├── n_clauses_total:       int  (n_classes*n_cpc for per_class)
│   ├── clause_organization:   "per_class" | "coalesced"
│   └── threshold:             int  (T in TMU parlance)
│
├── hyperparameters:  Hyperparameters
│   ├── s:     float  (specificity)
│   ├── n_states: int  (TA state range, default 256)
│   └── seed:  int | None
│
├── representation:   Representation          (at least one required)
│   ├── ta_states:   NDArray[int32]  (shape per clause_organization)
│   ├── includes:    NDArray[bool]   (lossy quantised view of ta_states)
│   └── compressed:  CompressedPayload | None
│       ├── scheme:           "redress" | "rle" | "bitpacked"
│       ├── block_size:       int | None
│       ├── payload:          bytes
│       └── decompressed_shape: tuple[int, ...]
│
├── weights:          Weights | None
│   ├── values:    NDArray[int32]  (shape per clause_organization)
│   ├── signed:    bool
│   ├── bit_width: int             (ceil(log2(abs_max+1)) + sign_bit)
│   └── range:     WeightRange
│       ├── min:     int
│       ├── max:     int
│       └── abs_max: int
│
├── derived:          Derived | None          (computed at construction)
│   ├── score_accumulator_width:    int
│   ├── avg_clause_length:          float
│   ├── max_clause_length:          int
│   ├── min_clause_length:          int
│   ├── clause_length_distribution: {length: count}
│   └── empty_clause_count:         int
│
├── preprocessing:    PreprocessingFingerprint | None
│   ├── name:        str
│   ├── config:      dict
│   └── fingerprint: str  (sha256:<hex>)
│
├── verification:     Verification | None
│   ├── test_vectors: list[TestVector]
│   │   ├── input:           list[int]
│   │   ├── expected_class:  int
│   │   └── expected_scores: list[int]
│   └── software_reference_hash: str
│
├── provenance:       Provenance | None
│   ├── framework:          str
│   ├── framework_version:  str
│   ├── dataset_id:         str | None
│   ├── trained_at:         datetime
│   ├── epochs:             int
│   └── hyperparameter_log: dict
│
└── telemetry:        Telemetry | None
    ├── clause_activation_frequency: NDArray[int] | None
    ├── per_class_clause_usage:      NDArray[int] | None
    └── marginality:                 NDArray[float] | None
                  (|state − threshold| / threshold per TA)
```

---

## Serialisation Formats

| Format           | API                       | Notes                              |
|------------------|---------------------------|------------------------------------|
| Python dict      | `to_dict()` / `from_dict()` | NumPy arrays as `{dtype,shape,data_b64}` |
| YAML             | `to_yaml()` / `from_yaml()` | Same encoding; human-readable      |
| NumPy `.npz`     | `to_npz()` / `from_npz()` | Arrays stored raw; preferred for large models |

---

## Invariants enforced by `validate_self()`

1. `n_literals == 2 * n_features`
2. At least one of `ta_states`, `includes`, `compressed` is present.
3. `per_class` organisation requires `n_clauses_per_class` to be set and
   `n_clauses_total == n_classes * n_clauses_per_class`.
4. `coalesced` organisation requires `weights` to be present.
5. Shape of `ta_states` / `includes` matches `clause_organization`.
6. Shape of `weights.values` matches `clause_organization`.

---

## Fingerprint

`TMIR.fingerprint()` returns a `sha256:<hex>` string computed over a
canonical JSON serialisation of `{variant, architecture, hyperparameters,
representation}`.  Use it to:
- Cache inference results without re-running the model.
- Detect model identity in provenance logs.

---

## Versioning

Bump `tmir_version` whenever the schema changes in a backward-incompatible
way.  Older readers should reject unknown versions.

Current version: **0.2**

---

## Source Locality Rule

> The file `matador/models/tmu_adapter.py` is the **only** module in the
> matador package that may contain `import tmu` or `from tmu` statements.

This invariant is enforced by an automated test in
`tests/test_tmu_adapter.py::test_only_tmu_adapter_imports_tmu`.
