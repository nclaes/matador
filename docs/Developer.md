# Developer Guide — Extending Matador

Matador has three independent, registry-based extension points, one per pipeline stage —
adding to any of them means writing new code/config in one place and registering it,
never touching the others:

| Extension point | Pipeline stage | Registry |
|---|---|---|
| [Adding a new dataset](#adding-a-new-dataset) | `matador ingest` | `data/Raw_Data_Bank.yaml` (`matador/preprocessing/registry.py`) |
| [Adding a new booleanisation technique](#adding-a-new-booleanisation-technique) | `matador booleanize` | `matador/preprocessing/encoders.py`'s `ENCODERS` dict |
| [Adding a new accelerator backend](#adding-a-new-accelerator-backend) | `matador generate`/`simulate`/`emulate` | `matador/backends/registry.py` |

```
data_source_config.yaml   matador ingest      raw arrays (npz)
   (or --dataset NAME)  ------------------->
booleanisation_config.yaml  matador booleanize   <name>_train.txt / <name>_test.txt
   (or --dataset NAME)    ------------------->
training_config.yaml        matador train        TMIR model
                          ------------------->
<backend>.yaml               matador generate     Verilog RTL
                          ------------------->
                              matador simulate / emulate
```

Deeper design rationale for the ingest/booleanize stages (why things are shaped the way
they are, not just how to add to them) is in [Preprocessing pipeline — design
notes](#preprocessing-pipeline--design-notes) at the end of this document.

---

## Adding a new dataset

A "dataset" is one entry in `data/Raw_Data_Bank.yaml` (or any catalog file shaped like
it) — it tells `matador ingest` where to fetch raw data from and how to parse it into
`(X, y)` arrays. Once added, `matador ingest --dataset <key>` and `matador booleanize
--dataset <key>` both work automatically, with **zero code changes** needed anywhere
else — the whole pipeline (CLI, registry, splash-screen status dashboard) is driven
purely off the catalog file. That file's own `FIELD REFERENCE` header (top of
`data/Raw_Data_Bank.yaml`) documents every field; `matador/preprocessing/sources.py`'s
`RawDataSourceSpec` (and its nested `SourceSpec`/`ExtractSpec`/`ParseSpec`/`SplitSpec`/
`ExportSpec`) is the pydantic schema it validates against.

### Worked example: registering `my_dataset`

#### 1. Pick the shape that matches your data

- **`source.kind`**: `http_zip` / `http_tar` / `http_files` (separate URLs, e.g. MNIST's
  four idx files) / `local` (already on disk — for a one-off, not usually for the shared
  catalog).
- **`extract.archive`**: `none` / `zip` / `tar.gz` / `gzip`, plus one optional level of
  `inner_archive` nesting (a zip inside a zip — see `human_activity` for a real example,
  and its own `notes` for a pitfall: the outer archive's `members` describes what to keep
  from the *inner* archive, not the outer one).
- **`parse.reader`**: `csv` (one row = one sample, single file or a fixed pool) /
  `dir_of_csv` / `dir_of_txt` (many files, either row-level via `label_column` or
  file-level via `label_from_path` — see `sports` for the file-level case, where each
  whole file flattens into one sample) / `idx` (MNIST ubyte format) / `cifar_pickle` /
  `wav_dir`. Pick the reader whose existing catalog entry looks most like your data and
  copy its `parse:` block as a starting point.
- **`split.mode`**: `random` (fit a train/test split yourself, optionally restricted to
  one extracted file via `source_member`) or `predefined` (the upstream data already
  ships its own train/test files — matched via `train`/`test` glob patterns, exact
  `source.urls[].role` names, or `spec_files` line-lists of held-out paths).

#### 2. Write the catalog entry

Add a new `- key: my_dataset` block under `datasets:` in `data/Raw_Data_Bank.yaml` (or
your own catalog file, if this isn't going into the shared one). Fill in `source`/
`extract`/`parse`/`split`/`export` per the field reference; `expect` is optional but
worth setting (a sanity-check dict — `matador ingest` warns, not fails, on a mismatch,
via `ingest.py::_check_expect`). Set `export.formats` to `[npz, csv]` unconditionally —
csv is only actually written when it's estimated to stay under
`ingest.py::_CSV_SIZE_THRESHOLD_MB` (100MB by default); a large real dataset (e.g. an
image dataset, or per-sample sensor data with no windowing applied) skips it
automatically with a warning stating the estimate, rather than needing you to
pre-judge the size by hand (`_export()`'s `_estimate_csv_bytes` formats a small row
sample and extrapolates, rather than writing the whole thing just to find out).

#### 3. (Optional) attach a verified booleanization recipe

If your raw-ingest output mechanically maps to a known, reproducible Boolean encoding
(a fixed value range, or thresholds you can fit from quantiles — no feature-engineering
step in between), add a `booleanization:` block:

```yaml
booleanization:
  default_encoder: {encoder: thermometer, bits: 8, range: [0, 16]}
```

This is what makes `matador booleanize --dataset my_dataset` (without `--show-recipe`)
actually run. If there's a feature-engineering step this pipeline doesn't automate
(windowing, MFCC, ...), leave it unset — `--show-recipe` still works (see [Adding a new
booleanisation technique](#adding-a-new-booleanisation-technique) below), it just won't
run without an explicit `--config`. Document *why* in the entry's `notes:` field either
way — every other catalog entry does this.

#### 4. Verify against the real source

```bash
matador list-datasets                                          # my_dataset should appear
matador ingest --dataset my_dataset --output-dir /tmp/raw_test  # real fetch — must succeed
```

Check the printed `train samples`/`test samples` counts (and any warnings) against your
`expect` block. **This step matters more than it looks** — three real catalog entries
(`emg`, `gesture_phase`, `human_activity`) shipped with bugs (a malformed row, a
schema/label mismatch, a nested-archive member-filtering bug) that only surfaced the
first time each was actually fetched for real; the schema validates fine regardless,
since it has no way to check the entry against the real upstream file.

#### 5. Add tests

Mirror the existing patterns in `tests/preprocessing/test_ingest.py`:
- A **synthetic fixture test** exercising your `(source.kind, extract.archive,
  parse.reader, split.mode)` combination with a small local zip/csv you construct in the
  test — fast, runs in CI with no network. Look for the combination closest to yours
  already covered (each is grouped under a `# --- Predefined split: ... ---` /
  `# --- Random split: ... ---` comment banner) before writing a new one from scratch.
- A **network-gated real-fetch test** (`@pytest.mark.skipif(os.environ.get(
  "MATADOR_NETWORK_TESTS") != "1", ...)`), checking the real upstream data ingests to the
  shape your `expect` block claims. This is the only thing that would have caught the
  three bugs mentioned above — a synthetic fixture is written by you, so it can't
  disagree with your own assumptions about the real file.
- If you added a `booleanization:` block, add it to `tests/preprocessing/
  test_registry.py`'s `test_get_default_booleanization_matches_expected_availability`/
  `test_default_recipe_produces_documented_bit_width` parametrize lists.

#### 6. Update docs if dataset counts changed

`docs/Usage.md`'s Step 2 lists which datasets have verified recipes vs. not by name, and
`docs/Developer.md` (this file) states counts (`N/M entries`) in a couple of places —
grep for the old count before committing.

---

## Adding a new booleanisation technique

A "booleanisation technique" is a new **encoder**: a `(fit_fn, apply_fn)` pair in
`matador/preprocessing/encoders.py`'s `ENCODERS` registry that turns one raw column into
a fixed-width block of 0/1 bits. `matador booleanize` looks encoders up purely by name
string (`ENCODERS[spec.encoder]` in `booleanize.py`), so adding one needs no changes to
the orchestration logic (`run_booleanize`) at all.

### Worked example: adding a `gray_code` encoder

#### 1. Write `fit_fn`/`apply_fn` in `encoders.py`

`fit_fn(train_col, spec) -> params` is computed from the **train split only** (never
test — this is what prevents test-set leakage; every existing encoder follows this),
`apply_fn(col, params) -> uint8[n, bits]` applies those fitted params to any column
(train or test). Follow the existing encoders' convention of treating non-finite (NaN)
values as all-zero bits rather than raising — `encode_threshold`/`encode_thermometer`'s
`valid = np.isfinite(col)` masking is the pattern to copy.

```python
def fit_gray_code(train_col: np.ndarray, bits: int) -> int:
    return bits   # no data-dependent fitting needed for this one -- bits is enough

def encode_gray_code(col: np.ndarray, bits: int) -> np.ndarray:
    valid = np.isfinite(col)
    ints = np.zeros(col.shape[0], dtype=np.uint32)
    ints[valid] = col[valid].astype(np.uint32)
    gray = ints ^ (ints >> 1)
    out = np.zeros((col.shape[0], bits), dtype=np.uint8)
    for i in range(bits):
        out[:, i] = (gray >> i) & 1
    out[~valid] = 0
    return out
```

#### 2. Register it in `ENCODERS`

```python
ENCODERS: dict[str, tuple[Callable, Callable]] = {
    ...
    "gray_code": (
        lambda train_col, spec: fit_gray_code(train_col, spec.bits or 8),
        encode_gray_code,
    ),
}
```

#### 3. Add it to `FeatureEncoderSpec` (`matador/config/schema.py`)

```python
class FeatureEncoderSpec(BaseModel):
    ...
    encoder: Literal["thermometer", "threshold", "onehot", "passthrough", "gray_code"]
    # add any gray_code-specific params fields here, following e.g. `threshold`'s
    # own `threshold: Optional[float]` pattern if your encoder needs one beyond `bits`
```

#### 4. Update `n_bits_for` if the bit-width isn't already covered

`encoders.py::n_bits_for(encoder, params)` tells `booleanize.py` how many bits a fitted
encoder produced (for the report and column-offset bookkeeping). `gray_code`'s width is
fixed at fit time exactly like `thermometer`'s, but its `params` is a plain `int`, not an
array — add a branch (`if encoder == "gray_code": return params`) rather than relying on
`thermometer`'s `.shape[0]`.

#### 5. Add tests

Mirror `tests/preprocessing/test_booleanize.py`'s per-encoder unit tests
(`test_thermometer_monotonic_and_range`, `test_threshold_encoder`, ...) — fit on a known
train column, assert the output bits directly, then a NaN-handling case. Add a
`test_gray_code_encoder` alongside them, and optionally extend
`test_booleanize_mixed_per_column_encoders_and_default` to cover mixing it with an
existing encoder in one config.

#### 6. (Optional) attach it as a dataset's verified default

If a technique reproduces a specific catalog entry's documented `booleanised.shape.bits`
exactly, you can now set that entry's `booleanization:` block to use it (see [Adding a
new dataset](#adding-a-new-dataset) step 3) — verified the same way the existing
defaults are, via `test_default_recipe_produces_documented_bit_width`.

---

## Adding a new accelerator backend

Matador uses a plugin registry so new accelerator architectures can be added without
touching existing code.

### Backend naming convention

```
<tm_variant>_<architecture>
```

Examples:
- `vanilla_tiled` — vanilla TM, tiled FSM + tile ROM
- `vanilla_hardwired` — vanilla TM, HCB streaming + adder tree
- `coalesced_tiled` — coalesced TM, tiled architecture (future)
- `weighted_hardwired` — weighted vanilla TM, hardwired (future)

### File structure

```
matador/backends/
  base.py              — AcceleratorSpec, RTLBackend, CycleAccurateModel ABCs
  registry.py          — plugin registry + descriptions
  vanilla_tiled/
    __init__.py
    config.py          — TiledAcceleratorConfig (Pydantic)
    rtl.py             — TiledBackend(RTLBackend)
    emulator.py         — TiledEmulator alias
  vanilla_hardwired/
    __init__.py
    config.py          — HardwiredAcceleratorConfig
    rtl.py             — HardwiredBackend(RTLBackend)
    emulator.py         — HardwiredEmulator
  gp_tiled/
    __init__.py
    config.py          — GPTiledAcceleratorConfig
    rtl.py             — GPTiledBackend(RTLBackend)
    emulator.py         — GPTiledEmulator
    tmir_bridge.py      — TMIR <-> vendored TMModel conversion
    vendor/             — vendored GP_TM_Inference_Accelerator sources
```

### Worked example: adding a `coalesced_tiled` backend

#### 1. Create the directory

```
matador/backends/coalesced_tiled/
  __init__.py
  config.py
  rtl.py
  emulator.py
```

#### 2. Write the config (`config.py`)

```python
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
from pathlib import Path

class CoalescedTiledConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model_path:      Path
    output_dir:      Path
    axis_data_width: Literal[32, 64] = 32
    fifo_depth:      int = 16
    feat_slice:      int = 4
    clause_slice:    int = 4
    # Coalesced-specific: weight bit width
    weight_bits:     int = Field(default=8, ge=1, le=16)
```

#### 3. Write the RTL generator (`rtl.py`)

```python
from matador.backends.base import RTLBackend, RTLArtifacts

class CoalescedTiledBackend(RTLBackend):

    @property
    def name(self) -> str:
        return "coalesced_tiled"

    @property
    def config_class(self):
        from matador.backends.coalesced_tiled.config import CoalescedTiledConfig
        return CoalescedTiledConfig

    @property
    def emulator_class(self):
        from matador.backends.coalesced_tiled.emulator import CoalescedTiledEmulator
        return CoalescedTiledEmulator

    def generate(self, tmir, config) -> RTLArtifacts:
        # Generate Verilog for the coalesced architecture
        # Must produce: src/*.v, tb/*.v, sim/*.sh, sim/verilator/Makefile, README.md
        ...
```

#### 4. Write the emulator (`emulator.py`)

Implement `CycleAccurateModel` from `matador.backends.base`.  The emulator must agree with:
1. The `AcceleratorSpec` — same predictions as `matador.inference.reference.predict()`
2. The RTL — verified by `matador emulate --verify`

#### 5. Register in `registry.py`

```python
_BACKENDS: dict[str, tuple[str, str]] = {
    "vanilla_tiled":     ("matador.backends.vanilla_tiled.rtl",     "TiledBackend"),
    "vanilla_hardwired": ("matador.backends.vanilla_hardwired.rtl", "HardwiredBackend"),
    "coalesced_tiled":   ("matador.backends.coalesced_tiled.rtl",   "CoalescedTiledBackend"),  # ← add
}

_DESCRIPTIONS: dict[str, str] = {
    ...
    "coalesced_tiled": "Coalesced TM — shared clause bank, signed weights, tiled FSM.",
}
```

#### 6. Add an example config

```bash
cp examples/vanilla_tiled.yaml examples/coalesced_tiled.yaml
# Edit to add weight_bits field
```

#### 7. Verify

```bash
matador list-backends           # should show coalesced_tiled
matador generate --backend coalesced_tiled --config /work/coalesced_tiled.yaml
matador simulate --backend coalesced_tiled --config /work/coalesced_tiled.yaml
matador emulate  --backend coalesced_tiled --config /work/coalesced_tiled.yaml --verify
```

### The three-layer contract

Every backend must satisfy:

```
AcceleratorSpec.simulate(x).predicted_class
  == CycleAccurateModel.run(x).predicted_class
  == RTL_simulation(x).output

for all x in the test set.
```

`matador emulate --verify` enforces the second equality at runtime.  The first equality is enforced by the reference inference engine in `matador.inference.reference`.

> **Known limitation:** `matador emulate --verify`'s cross-check (`matador.verification.compare.compare_emulator_to_reference`) hardcodes `TMAcceleratorEmulator` and reads `cfg.feat_slice`/`cfg.clause_slice` — it only works for `vanilla_tiled` today. It raises `AttributeError` for `vanilla_hardwired` and `vanilla_gp_tiled` configs, which don't have those fields. This predates the `vanilla_gp_tiled` backend (confirmed by reproducing the same failure against `vanilla_hardwired` on a clean checkout) and is not backend-specific plumbing, so it wasn't fixed as part of adding `vanilla_gp_tiled` — it needs `compare_emulator_to_reference` to go through `backend.emulator_class` generically instead of importing the tiled emulator directly. Plain `matador emulate` (without `--verify`) already exercises the first equality correctly for every backend, since it compares each backend's own emulator against the TMIR's embedded `expected_class` (itself produced by the reference engine at training time).

### Runtime-reprogrammable backends (vendoring pre-existing RTL)

`vanilla_gp_tiled` (`matador/backends/gp_tiled/`) is a different shape of backend from `vanilla_tiled`/`vanilla_hardwired`: instead of generating fresh RTL parameterized to one specific trained model, it wraps a **pre-existing, hand-verified RTL core** (vendored from GP_TM_Inference_Accelerator) that is synthesized once and reprogrammed at runtime over `s_axis` with an AXI-Stream `CMD_LOAD` packet. This pattern differs from the worked example above in a few ways worth knowing before adding another one:

- **Compile-time capacity vs. runtime geometry, sized against a real device.** The config (`GPTiledAcceleratorConfig`) separates per-cycle SIMD width (`feat_slice`/`clause_slice` — "how many rounds of computation" one inference takes), capacity in real units (`max_features`/`max_clauses_total`/`max_classes`), and a `target_fpga` (or explicit `bram_bits_budget`). `max_feat_slices`/`max_clause_slices`/`n_tiles_max` are *derived* properties, not stored fields — `ceil(max_features/feat_slice)` etc. — so they can't drift out of sync. A `model_validator` computes `tile_mem`'s actual bit cost (`2 x features_padded x clauses_padded`, see `matador.backends.gp_tiled.fpga_budget`) and rejects the config outright if it doesn't fit the target device's BRAM budget — this replaced an earlier version of this backend that checked four independent, arbitrary per-field ceilings with no connection to real hardware feasibility. Any model whose actual geometry fits within the configured capacity loads without resynthesis. `generate()` still requires `model_path` (to build the `CMD_LOAD` artifact and a model-specific verification testbench), but the RTL sources it emits only vary with the capacity/device fields, not the model.
- **The vendored core's internal register widths are a second, independent ceiling.** Even after the BRAM-budget check above, `tm_accel_gp.v`'s internal FSM counters (`score_cnt`, `cfg_n_tiles`, `feat_pos_base`, etc.) impose their own hard limits — these were widened once (to exactly match the AXI-Stream header's own 8-bit/16-bit sub-field ceilings, see `tmir_bridge.HARD_MAX_*`) rather than left at whatever the shipped demo capacity happened to need. If you ever need to go *beyond* those (e.g. `clauses_per_class > 255`), that requires changing the wire protocol's header word layout itself, not just a register width — a genuinely bigger, breaking change, out of scope for a config knob.
- **Vendored, not templated, sources.** `matador/backends/gp_tiled/vendor/` holds the upstream RTL/Python files close to verbatim. `generate()` only does targeted regex substitution of the top module's capacity `parameter` declarations (see `rtl.py::_sub_param` / `_capacity_params`, shared between the generated `src/tm_accel_gp.v` and the model-specific testbench's DUT instantiation so they can't disagree) — it does not re-derive the datapath from TMIR the way `TMAccelerator`/`HardwiredBackend`'s generators do. This keeps the vendored core easy to diff against upstream if it's updated.
- **Two testbenches, two jobs.** The vendored `tb_system_gp.v` is a generic protocol-conformance suite (LOAD/INFER framing, error injection/recovery) that is identical regardless of which model is loaded — it does not close the three-layer contract by itself. `generate()` additionally emits a model-specific `tb_system_gp_model.v` (built in `rtl.py::_gen_model_testbench`) that loads *this* TMIR model and replays its embedded `verification.test_vectors`, which is what actually exercises `AcceleratorSpec.simulate(x) == RTL_simulation(x)` for the trained model.
- **`` `include ``-based vector files must be flattened.** The vendored testbench pulls in vector files via `` `include "vectors/gp_sizes.vh" `` and `$readmemh("vectors/*.memh", ...)`, assuming it's compiled with a `vectors/` subdirectory alongside it. `matador.models.validator.validate_rtl` doesn't set a compile-time working directory, so `generate()` rewrites these to sim_dir-flat paths (`rtl.py::_flatten_includes`) rather than relying on Icarus's `` `include `` search-path behavior.
- **Bridging conventions.** `matador/backends/gp_tiled/tmir_bridge.py` converts between TMIR's `(n_classes, n_clauses_per_class, n_literals)` boolean include array and the vendored core's own `TMModel.include[g]` integer-bitmask convention — this is the general shape to follow when vendoring any external golden model that has its own bespoke in-memory representation. `sync_capacity()` also repoints the vendored `tm_emulator` module's `FEAT_SLICE`/`CLAUSE_SLICE`/`TILE_WIDTH`/`WORDS_PER_ROW`/`MAX_*` globals at the configured (not the vendored file's default) values before every `TMModel` operation, since those are read as module globals at call time rather than bound once at import.
- **Exported standalone tooling, because RTL != model on this backend.** Since the same synthesized bitstream can be reprogrammed with a *different* model later via a fresh `CMD_LOAD` (no resynthesis), `generate()` also writes into `RTL/sim/`: a verbatim copy of `tm_emulator.py`, and `gen_vectors.py` (`matador/backends/gp_tiled/vendor/gen_vectors.py`, copied unmodified — stdlib-only, imports nothing from matador) which builds fresh `CMD_LOAD`/`CMD_INFER` `.memh` streams — and, with `--testbench`, a matching ready-to-compile Verilog testbench — for *any* new model described in a small JSON schema, entirely standalone (no matador install needed on the machine that has the exported RTL folder). `capacity.json` (this bitstream's actual synthesized ceilings) and `model_reference.json` (the generation-time model, doubling as a working schema example) are exported alongside it so `gen_vectors.py` validates new models against the real hardware, not its own built-in defaults. `RTL/provenance.json` is a separate, simpler concern: a software manifest (`rtl.py::_gen_provenance`) recording the *generation-time* model's TMIR fingerprint/geometry/training provenance — it identifies what's in `model_stimulus.memh` right now, but goes stale the moment a different model's `CMD_LOAD` is replayed against the live hardware (there's no on-chip readback to re-verify what's actually sitting in `tile_mem`; this is deliberately scoped as identification, not a hardware memory-integrity check).
- **Verilator support needs no per-config templating, for the same reason `gen_vectors.py` doesn't.** `matador.models.validator.validate_rtl` already runs whatever it finds at `sim/verilator/Makefile` for any backend — the gap was simply that no such Makefile existed for this backend. `matador/backends/gp_tiled/vendor/verilator/{Makefile,tb_top.cpp}` are static files, copied verbatim by `generate()` (mirroring `gen_vectors.py`'s own vendoring). `tb_top.cpp` reads its `--stim`/`--exp` `.memh` paths *at runtime* (`std::ifstream`, matching `tm_emulator.write_memh()`'s `{tlast,data}` line format) rather than embedding compile-time C arrays the way the original vendored `tb_top.cpp` (and `TiledBackend`/`HardwiredBackend`'s own `_gen_verilator_tb_cpp` generators) do — so the one compiled binary replays the generation-time model by default, or any `gen_vectors.py`-produced vectors for a brand-new model, with no rebuild. Verilator's `-Wall` flagged `UNUSEDSIGNAL` on the widened registers from the capacity-headroom work (upper bits of header-staging/derived-width registers that are legitimately unused at smaller capacities) — suppressed via `--Wno-UNUSEDSIGNAL`, the same category of "expected, not a bug" suppression the existing `--Wno-WIDTH`/`--Wno-BLKSEQ` flags already represent. Separately: fixed a latent bug in `matador.models.validator.validate_rtl`'s Verilator binary discovery (`glob("V*")` also matches Verilator's companion build artifacts — `.mk`, `.h`, `.a`, ... — and glob order isn't guaranteed, so it could pick a non-executable file; now filtered on the executable bit). That fix is shared infrastructure and applies to every backend's Verilator harness, not just this one.
- **`gen_vectors.py sequence` proves reprogrammability with the user's own models**, not just the two hardcoded demo models baked into `tb_system_gp.v`'s own vectors. It's the same `encode_load_packet()`/`encode_infer_packet()`/`stream_with_tlast()` pattern `combined` already uses for one model, generalized to a manifest (ordered list) of steps — each step's own model/vectors, concatenated into one continuous multi-reprogram stream. `--testbench` reuses `_emit_testbench()` unchanged (it only needs a display name and `tm_emulator`'s already-synced capacity globals, not anything model-specific), so no protocol-aware C++ or Verilog changes were needed to support it either.
- **Generated shell scripts must resolve paths at runtime, not bake them in at `generate()` time.** `_gen_sim_scripts`'s `run_iverilog.sh` used to interpolate absolute `src`/`tb` paths derived from `config.output_dir` directly into the script text (`f"{src}/{f}"`) — this happened to work as long as the bundle stayed at the exact filesystem path it was generated at (e.g. inside the dev container, where `output_dir` is typically the `/work` bind-mount point), but broke the moment the RTL folder was copied elsewhere or run outside that container, since the baked-in `/work/...` paths no longer resolved. Fixed to compute `SCRIPT_DIR`/`SRC_DIR`/`TB_DIR` from `$(dirname "$0")` at runtime instead, matching the pattern `waves.sh` and the Verilator `Makefile` already used correctly. `_gen_sim_scripts()` no longer needs an `rtl_dir` argument at all as a result — there's nothing generation-time-specific left to bake in.
- **`sim/test_vectors.txt` + a curated `sim/tb_system_gp_model.gtkw` answer "where are the test vectors" and "how do I check predicted vs. expected" without reading Verilog.** The `.memh` stimulus/expected files are opaque hex, and the raw GTKWave signal tree is an unsorted flat list — neither tells a newcomer which wire holds the predicted class or what it should equal. `rtl.py::_gen_test_vectors_txt` renders `tmir.verification.test_vectors` as a plain index/expected-class/input-bits table, with an explicit beat-numbering note (beat 0 = the `CMD_LOAD` ack, beat `k+1` = row `k`'s prediction) that callers can check directly against simulation output rather than take on faith. `rtl.py::_gen_model_tb_gtkw` reuses `matador.waves.gtkwave`'s low-level `.gtkw` helpers (`_header`/`_group`/`_sig`, the same ones `TMAccelerator`'s own waveform generation already uses) to group `tb_system_gp_model`'s signals into AXI-Stream Input / AXI-Stream Output (where `m_tdata` — the predicted class — lives) / Status / Testbench bookkeeping (`e`/`sent`/`fail_cnt`, so a beat number in the console log can be found in the waveform), and `waves.sh` now opens it automatically when present (`${WF%.vcd}.gtkw` lookup) instead of falling back to GTKWave's default unsorted view. The generated `README.md`'s new `## 0. Orientation` section points at `sim/test_vectors.txt` first, before any other file, as the answer to "where are the test vectors."
- **`matador reprogram-suite` proves reprogrammability across multiple models/datasets as a first-class Matador flow**, not just via the exported `gen_vectors.py sequence` script. `RTLBackend` (`matador/backends/base.py`) gained an opt-in capability pair — `supports_reprogramming` (`False` by default; `vanilla_tiled`/`vanilla_hardwired` need no changes) and `build_reprogram_suite(rtl_dir, steps: list[ReprogramStep], config)` (raises `NotImplementedError` by default) — so this stays scoped to backends that actually support it rather than becoming a premature core concept. `GPTiledBackend.build_reprogram_suite()` (`rtl.py`) augments an **already-generated** bundle (same `rtl_dir` convention `simulate` resolves — requires `matador generate` to have run first) by reusing existing infra rather than duplicating it: `tmir_bridge.check_capacity`/`tmir_to_tmmodel` per step (the same calls the single-model path already makes) and `vendor.tm_emulator`'s `encode_load_packet`/`encode_infer_packet`/`stream_with_tlast`/`expected_load_ack` imported as a library (not copy-pasted from the vendored CLI script). `_render_model_testbench`/`_gen_model_tb_gtkw` were generalized to take a testbench-name/file-name parameters (defaulting to the original single-model values, so the existing call site is unchanged) so the ~140-line Verilog template and the `.gtkw` layout logic aren't duplicated for the multi-step case. Each `ReprogramStep`'s vectors resolve in priority order — `vectors_path` (raw bit file) > `dataset_path` (a `matador booleanize`-shaped `*_test.txt`, label column dropped, optionally subsampled via `n_samples`/`seed`) > the step's own TMIR's embedded `verification.test_vectors` (default) — and in every case the **expected class is the model's own `infer_tiled()` prediction**, never the dataset's ground-truth label, matching the three-layer contract's actual invariant (`RTL == software reference`, not `RTL == ground truth`). New `sim/reprogram_manifest.txt` (per-step beat-index accounting, same "how do I read this against the waveform" convention as `test_vectors.txt`) and `sim/tb_reprogram_suite.gtkw` round out the bundle. `ReprogramSuiteConfig` (`matador/backends/gp_tiled/reprogram_config.py`) deliberately has no `backend`/`output_dir` fields of its own — the `matador reprogram-suite` CLI command takes `--backend`/`--config` the same way `simulate` already does, so a reprogram config can't silently drift out of sync with a separately-recorded output directory.

### Generated file checklist

A production-ready backend should generate:

| File | Required |
|---|---|
| `src/*.v` — Verilog sources | ✓ |
| `tb/tb_system.v` — system testbench | ✓ |
| `sim/run_iverilog.sh` | ✓ |
| `sim/lint_verilator.sh` | ✓ |
| `sim/verilator/Makefile` + `tb_top.cpp` | ✓ |
| `sim/*.gtkw` — GTKWave save files | ✓ |
| `sim/waves.sh` | ✓ |
| `sim/run_xsim.sh` | ✓ |
| `RTL/README.md` | ✓ |

---

## Preprocessing pipeline — design notes

`matador/preprocessing/` (previously an empty stub package) implements the two stages upstream of `matador train`, which has always required pre-booleanized `train_data`/`test_data` files (space-separated 0/1, last column = label — `matador/models/trainer.py::load_data`). Both stages are independently invocable and neither changes `TrainingConfig`/`load_data()` at all — `matador booleanize`'s output is byte-for-byte the same format `matador train` already parses.

This section explains *why* the two stages above are shaped the way they are; see [Adding
a new dataset](#adding-a-new-dataset) / [Adding a new booleanisation
technique](#adding-a-new-booleanisation-technique) for the *how*.

- **Stage 1 (`matador ingest`) mirrors `data/Raw_Data_Bank.yaml`'s own schema** (`matador/preprocessing/sources.py`'s `RawDataSourceSpec`/`RawDataCatalog` — the file's own "FIELD REFERENCE" header is effectively the spec) rather than inventing a new one, plus one addition: `source.kind == "local"` for raw data that's already on disk (skips the fetch step; a local **archive** still goes through `extract.archive`, since `kind` only decides whether downloading was needed, not whether unpacking is — see `ingest.py::_fetch_and_extract`'s routing). `resolve_source_config()` accepts either a standalone single-dataset file (one `datasets[]` entry's shape, no wrapper — the per-project `data_source_config.yaml`) or a `{catalog, key}` pointer into a shared catalog like `data/Raw_Data_Bank.yaml` itself.
- **Full fetch automation for every `(source.kind, extract.archive, parse.reader, split.mode)` combination the real catalog uses** (all 9 entries validate against the schema — `test_real_catalog_validates`): `fetch.py` (stdlib `urllib`, retries + `mirrors[]` fallback + `checksum_policy`), `archives.py` (zip/tar.gz/gzip, glob `members[]`, one level of `inner_archive` nesting for `human_activity`'s zip-inside-zip — the outer extraction step is filtered against `[inner_archive]`, not `spec.members`, since `members` describes files inside the *inner* archive), `readers.py` (`csv`/`dir_of_csv`/`dir_of_txt`/`idx`/`cifar_pickle`/`wav_dir` — the last two aren't exercised by any current catalog entry since `cifar2`/`kws2` were dropped, but remain available generic readers for a custom `--config` dataset), `splitters.py` (`split_random` — sklearn's `train_test_split` when available, since its stratified rounding is what actually produced several of the catalog's own recorded split sizes like digits' 1437/360, with a plain-numpy fallback when sklearn isn't installed; `split_predefined`'s glob/role/`spec_files` resolution).
- **Two label conventions per reader, dispatched on which knob is set, not one reader per combination.** `label_column` = row-level (each row's own label, e.g. `emg`/`gesture_phase`); `label_from_path` = file-level (the whole file IS one sample, flattened, labeled from its path/filename — e.g. `sports`' per-segment files, or a Speech-Commands-style per-clip `.wav` layout). `label_from_path` patterns are matched against each file's path **relative to the common ancestor of the files being read** (`readers.py::_relative_strings`), not the absolute filesystem path — an anchored pattern like `"^(yes|no)/"` would never match an absolute path, which starts with `/tmp/...` or similar, not `yes/`.
- **A single file's rows aren't assumed uniform width.** `readers.py::_rows_to_xy` derives its expected column count from the file's first row, but a real raw file can still have one malformed/truncated line partway through (hit for real on `emg`'s raw data — one row out of ~80k was missing its trailing label token) — any row whose token count differs is skipped and counted into a warning (`IngestReport.warnings`) rather than crashing the whole ingest with an `IndexError`.
- **`export.formats`'s `csv` request is honored dynamically, not by a static per-dataset size guess.** Every catalog entry can request `[npz, csv]` uniformly; `ingest.py::_export()` estimates the real on-disk csv size first (`_estimate_csv_bytes` — formats a small row sample and extrapolates, rather than writing the whole thing just to find out) and skips writing it (with a warning stating the estimate) above `_CSV_SIZE_THRESHOLD_MB` (100MB). This replaced an earlier design where three entries (`mnist`, `sports`, and `emg`) were hardcoded `formats: [npz]` only — `mnist`/`sports` for a real reason (their csvs would be ~127MB/~436MB), but `emg`'s exclusion turned out to be based on an incorrect size assumption (its *raw*, un-windowed ingest output is ~4.2M rows, not the ~57k the booleanised/RMS-windowed reference copy has — see its own `notes`), which a static per-dataset guess had no way to catch or self-correct. `human_activity` (~64MB) is the largest csv currently written, comfortably under the threshold.
- **Labels are normalized to be 0-indexed** (`ingest.py::_normalize_labels`, shifting by the observed minimum across train+test) since several catalog entries have raw labels starting at 1 (`sensorless_drive`'s own notes flag this explicitly) — recorded in the ingest report, not silent.
- **Stage 2 (`matador booleanize`)** — `encoders.py`'s registry (`thermometer`/`threshold`/`onehot`/`passthrough`) fits parameters (bin edges, categories) on the **train split only** to avoid test-set leakage, then applies them identically to train and test. `BooleanisationConfig` (`matador/config/schema.py`, alongside `TrainingConfig`/`ValidationConfig`) maps raw columns to encoders via an ordered `features:` list (single index or an inclusive `"lo-hi"` range string, applying the same encoder independently per covered column) plus a `default_encoder` fallback; every raw column must resolve to exactly one encoder or `run_booleanize` raises. Byte-exact reproduction of `data/Digits/*.txt`'s specific row order/thermometer convention isn't achievable (the catalog's own notes repeatedly flag that several datasets' original split seeds/encoding conventions were never recorded) — the network-gated tests instead check what's actually verifiable: exact shape match, a close (not exact) class distribution, and thermometer-encoding self-consistency (see `test_booleanize_digits_end_to_end_matches_committed_shape`).
- **Network-gated tests** (`tests/preprocessing/test_ingest.py`, `test_booleanize.py`) follow the same skip-gate convention as `_HAVE_IVERILOG`/`_HAVE_VERILATOR` elsewhere, but on an env var (`MATADOR_NETWORK_TESTS=1`) rather than a binary check — real fetches of `digits`/`human_activity`/`gesture_phase`, checked against either already-committed ground truth (`data/Digits/Digits_train.txt`/`Digits_test.txt`) or the catalog's own `expect` block, for a true correctness check rather than "it ran." This is the only thing that has actually caught real catalog bugs so far (see the `emg`/`gesture_phase`/`human_activity` note in [Adding a new dataset](#adding-a-new-dataset) step 4) — a schema-validation-only test cannot.
- **`matador/preprocessing/registry.py` gives cataloged datasets the same by-name ergonomics `matador/backends/registry.py` gives backends** — `list_datasets()`/`get_dataset(name)`/`describe_dataset(name)` mirror `list_backends()`/`get(name)`/`describe(name)` exactly, except a dataset registration is pure data (a `RawDataCatalog` entry), not a lazily-imported class, so there's no module-loading step — just a name-indexed lookup over `RawDataCatalog.get()` (already implemented in `sources.py`). `DEFAULT_CATALOG` resolves via `Path(matador.__file__).resolve().parent.parent / "data" / "Raw_Data_Bank.yaml"` — relative to the **installed package**, not the CWD, so it works identically in a local dev checkout and inside the Docker container (where `make shell` bind-mounts the repo root at `/workspace`, the same directory `matador/__init__.py` lives one level under). `matador ingest --dataset NAME` / `matador booleanize --dataset NAME` sit alongside the pre-existing `--config <path>` flag on both commands (mutually exclusive, checked explicitly in `cli.py`) — mirrors `--backend NAME` sitting alongside `--config` on `generate`/`simulate`.
- **A registered dataset's `booleanization:` block is deliberately optional, not required** — this is the "linked but swappable" design: a catalog entry names *where the raw data comes from*; a *default* Boolean encoding recipe is attached only when the raw-ingest output mechanically maps to a known, verifiable encoding (checked by reproducing each recipe's documented bit width from `booleanised.shape.bits` — see `test_default_recipe_produces_documented_bit_width`), and left unset wherever the catalog's own `notes`/`booleanised.encoding` already documents a feature-engineering step (windowed statistics, MFCC, dropped/re-derived columns) this pipeline doesn't implement (`sports`, `gesture_phase`, `human_activity`, `emg` — `matador booleanize --dataset <one of these>` fails with a clear pointer at `--config` instead, rather than silently claiming a reproduction that isn't real). `BooleanizationRecipe` (`matador/config/schema.py`) is the encoding-only half of `BooleanisationConfig` — `BooleanisationConfig` now inherits from it, adding just the I/O fields (`raw_npz`/`name`/`output_dir`/split params), so existing `config.features`/`config.default_encoder` access didn't need to change anywhere.
- **`matador booleanize --dataset NAME --show-recipe` makes a recipe inspectable — and editable — for every registered dataset, not just the 5 with a verified default.** `registry.get_recipe_for_inspection(name)` (distinct from `get_default_booleanization()`, which stays limited to verified defaults) returns `(recipe, is_verified)`: for the 5 verified datasets, the real catalog default; for the other 4, a generic, dataset-agnostic `_generic_skeleton_recipe()` (an 8-bit quantile-fit thermometer — no assumed value range, since none is known) explicitly labeled `UNVERIFIED` in the printed comments, with that dataset's own `notes` quoted inline so the reason there's no verified default is right there, not a separate lookup. This is the "suggest your own encoding" path for the unverified 4: `--show-recipe`'s output is a real, editable `booleanisation_config.yaml` either way (`raw_npz`/`name`/`output_dir` filled in from `--raw-dir`/`--output-dir`, `features`/`default_encoder` from the recipe), printed **before** checking that `raw_npz` exists, since inspecting a recipe is legitimate before ever running `matador ingest` (a `# NOTE:` comment says so if it's missing, rather than erroring). The safety line is drawn at the *actual run* path, not inspection: bare `matador booleanize --dataset NAME` (no `--show-recipe`) still calls `get_default_booleanization()` and refuses outright for the unverified 4 — matador never silently applies the generic skeleton's guess, only shows it. `cli.py::_yaml_safe()` exists solely because `FeatureEncoderSpec.range` round-trips through pydantic's `model_dump()` as a real `tuple`, which `yaml.safe_dump` can't represent — converted to a list first. The round trip is real for both cases, not just cosmetic: `matador booleanize --dataset X --show-recipe > cfg.yaml` followed by `matador booleanize --config cfg.yaml` produces byte-identical output to running `--dataset X` directly when verified (`test_show_recipe_output_round_trips_as_a_working_config`), and the unverified skeleton round-trips into a genuinely runnable config too (`test_unverified_skeleton_round_trips_into_a_working_config`). `matador list-datasets` shows a compact one-line summary either way (`_format_encoder_spec()`) so browsing all datasets doesn't require invoking `--show-recipe` once per entry.
