# Developer Guide — Adding New Backends

Matador uses a plugin registry so new accelerator architectures can be added without touching existing code.

---

## Backend naming convention

```
<tm_variant>_<architecture>
```

Examples:
- `vanilla_tiled` — vanilla TM, tiled FSM + tile ROM  
- `vanilla_hardwired` — vanilla TM, HCB streaming + adder tree
- `coalesced_tiled` — coalesced TM, tiled architecture (future)
- `weighted_hardwired` — weighted vanilla TM, hardwired (future)

---

## File structure

```
matador/backends/
  base.py              — AcceleratorSpec, RTLBackend, CycleAccurateModel ABCs
  registry.py          — plugin registry + descriptions
  vanilla_tiled/
    __init__.py
    config.py          — TiledAcceleratorConfig (Pydantic)
    rtl.py             — TiledBackend(RTLBackend)
    emulator.py        — TiledEmulator alias
  vanilla_hardwired/
    __init__.py
    config.py          — HardwiredAcceleratorConfig
    rtl.py             — HardwiredBackend(RTLBackend)
    emulator.py        — HardwiredEmulator
  gp_tiled/
    __init__.py
    config.py          — GPTiledAcceleratorConfig
    rtl.py             — GPTiledBackend(RTLBackend)
    emulator.py        — GPTiledEmulator
    tmir_bridge.py     — TMIR <-> vendored TMModel conversion
    vendor/            — vendored GP_TM_Inference_Accelerator sources
```

---

## Adding a new backend — worked example

To add a `coalesced_tiled` backend:

### 1. Create the directory

```
matador/backends/coalesced_tiled/
  __init__.py
  config.py
  rtl.py
  emulator.py
```

### 2. Write the config (`config.py`)

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

### 3. Write the RTL generator (`rtl.py`)

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

### 4. Write the emulator (`emulator.py`)

Implement `CycleAccurateModel` from `matador.backends.base`.  The emulator must agree with:
1. The `AcceleratorSpec` — same predictions as `matador.inference.reference.predict()`
2. The RTL — verified by `matador emulate --verify`

### 5. Register in `registry.py`

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

### 6. Add an example config

```bash
cp examples/vanilla_tiled.yaml examples/coalesced_tiled.yaml
# Edit to add weight_bits field
```

### 7. Verify

```bash
matador list-backends           # should show coalesced_tiled
matador generate --backend coalesced_tiled --config /work/coalesced_tiled.yaml
matador simulate --backend coalesced_tiled --config /work/coalesced_tiled.yaml
matador emulate  --backend coalesced_tiled --config /work/coalesced_tiled.yaml --verify
```

---

## The three-layer contract

Every backend must satisfy:

```
AcceleratorSpec.simulate(x).predicted_class
  == CycleAccurateModel.run(x).predicted_class
  == RTL_simulation(x).output

for all x in the test set.
```

`matador emulate --verify` enforces the second equality at runtime.  The first equality is enforced by the reference inference engine in `matador.inference.reference`.

> **Known limitation:** `matador emulate --verify`'s cross-check (`matador.verification.compare.compare_emulator_to_reference`) hardcodes `TMAcceleratorEmulator` and reads `cfg.feat_slice`/`cfg.clause_slice` — it only works for `vanilla_tiled` today. It raises `AttributeError` for `vanilla_hardwired` and `vanilla_gp_tiled` configs, which don't have those fields. This predates the `vanilla_gp_tiled` backend (confirmed by reproducing the same failure against `vanilla_hardwired` on a clean checkout) and is not backend-specific plumbing, so it wasn't fixed as part of adding `vanilla_gp_tiled` — it needs `compare_emulator_to_reference` to go through `backend.emulator_class` generically instead of importing the tiled emulator directly. Plain `matador emulate` (without `--verify`) already exercises the first equality correctly for every backend, since it compares each backend's own emulator against the TMIR's embedded `expected_class` (itself produced by the reference engine at training time).

---

## Runtime-reprogrammable backends (vendoring pre-existing RTL)

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

---

## Preprocessing pipeline (raw data → booleanized data)

`matador/preprocessing/` (previously an empty stub package) implements the two stages upstream of `matador train`, which has always required pre-booleanized `train_data`/`test_data` files (space-separated 0/1, last column = label — `matador/models/trainer.py::load_data`). Both stages are independently invocable and neither changes `TrainingConfig`/`load_data()` at all — `matador booleanize`'s output is byte-for-byte the same format `matador train` already parses.

```
data_source_config.yaml  →  matador ingest      →  raw arrays (npz)
booleanisation_config.yaml → matador booleanize  →  <name>_train.txt / <name>_test.txt
                                                      (unchanged matador train from here)
```

- **Stage 1 (`matador ingest`) mirrors `data/Raw_Data_Bank.yaml`'s own schema** (`matador/preprocessing/sources.py`'s `RawDataSourceSpec`/`RawDataCatalog` — the file's own "FIELD REFERENCE" header is effectively the spec) rather than inventing a new one, plus one addition: `source.kind == "local"` for raw data that's already on disk (skips the fetch step; a local **archive** still goes through `extract.archive`, since `kind` only decides whether downloading was needed, not whether unpacking is — see `ingest.py::_fetch_and_extract`'s routing). `resolve_source_config()` accepts either a standalone single-dataset file (one `datasets[]` entry's shape, no wrapper — the per-project `data_source_config.yaml`) or a `{catalog, key}` pointer into a shared catalog like `data/Raw_Data_Bank.yaml` itself.
- **Full fetch automation for every `(source.kind, extract.archive, parse.reader, split.mode)` combination the real catalog uses** (all 11 entries validate against the schema — `test_real_catalog_validates`): `fetch.py` (stdlib `urllib`, retries + `mirrors[]` fallback + `checksum_policy`), `archives.py` (zip/tar.gz/gzip, glob `members[]`, one level of `inner_archive` nesting for `human_activity`'s zip-inside-zip), `readers.py` (`csv`/`dir_of_csv`/`dir_of_txt`/`idx`/`cifar_pickle`/`wav_dir`), `splitters.py` (`split_random` — sklearn's `train_test_split` when available, since its stratified rounding is what actually produced several of the catalog's own recorded split sizes like digits' 1437/360, with a plain-numpy fallback when sklearn isn't installed; `split_predefined`'s glob/role/`spec_files` resolution).
- **Two label conventions per reader, dispatched on which knob is set, not one reader per combination.** `label_column` = row-level (each row's own label, e.g. `emg`/`gesture_phase`); `label_from_path` = file-level (the whole file IS one sample, flattened, labeled from its path/filename — e.g. `sports`' per-segment files, `kws2`'s per-clip `.wav`s). `label_from_path` patterns are matched against each file's path **relative to the common ancestor of the files being read** (`readers.py::_relative_strings`), not the absolute filesystem path — an anchored pattern like `kws2`'s `"^(yes|no)/"` would never match an absolute path, which starts with `/tmp/...` or similar, not `yes/`.
- **Labels are normalized to be 0-indexed** (`ingest.py::_normalize_labels`, shifting by the observed minimum across train+test) since several catalog entries have raw labels starting at 1 (`sensorless_drive`'s own notes flag this explicitly) — recorded in the ingest report, not silent.
- **Stage 2 (`matador booleanize`)** — `encoders.py`'s registry (`thermometer`/`threshold`/`onehot`/`passthrough`) fits parameters (bin edges, categories) on the **train split only** to avoid test-set leakage, then applies them identically to train and test. `BooleanisationConfig` (`matador/config/schema.py`, alongside `TrainingConfig`/`ValidationConfig`) maps raw columns to encoders via an ordered `features:` list (single index or an inclusive `"lo-hi"` range string, applying the same encoder independently per covered column) plus a `default_encoder` fallback; every raw column must resolve to exactly one encoder or `run_booleanize` raises. Byte-exact reproduction of `data/Digits/*.txt`'s specific row order/thermometer convention isn't achievable (the catalog's own notes repeatedly flag that several datasets' original split seeds/encoding conventions were never recorded) — the network-gated tests instead check what's actually verifiable: exact shape match, a close (not exact) class distribution, and thermometer-encoding self-consistency (see `test_booleanize_digits_end_to_end_matches_committed_shape`).
- **Network-gated tests** (`tests/preprocessing/test_ingest.py`, `test_booleanize.py`) follow the same skip-gate convention as `_HAVE_IVERILOG`/`_HAVE_VERILATOR` elsewhere, but on an env var (`MATADOR_NETWORK_TESTS=1`) rather than a binary check — a real fetch of the `digits` UCI dataset, checked against the already-committed `data/Digits/Digits_train.txt`/`Digits_test.txt` for a true correctness check rather than "it ran."

---

## Generated file checklist

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
