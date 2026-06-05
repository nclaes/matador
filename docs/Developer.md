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
