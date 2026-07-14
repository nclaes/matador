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
