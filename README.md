### <img src="/banner.png" width=900/>
Automated ASIC and FPGA design for Tsetlin Machine Accelerators.

| Repo Branch  | Description |
| ------------ | ------------ |
| `main`       | Overview of each Matador flow branch. |
| `development`| Automated FPGA accelerator design (GUI). |
| `4adrian`    | Original non-GUI Coalesced-TM flow (JSON-config, single-script). |
| `cotm_only`  | **This branch.** Minimal, Coalesced-TM-only CLI + Docker + emulator + testbenches, built on top of `4adrian`. |

## What this branch is

A small, focused toolchain for one accelerator architecture — the
**Coalesced Tsetlin Machine** (shared clause bank, per-class weighted sum,
argmax) — nothing else. No GUI, no Vanilla TM, no synthesis/deployment
flow (those live in `legacy/`, retired but not deleted). Four commands:
train a model, generate RTL from a trained model's TAs/weights, generate a
self-checking testbench, and run it under iverilog or Verilator.

The `4adrian` branch's Coalesced flow existed but didn't actually work
end-to-end: the vendored `tmu` library couldn't be imported at all under
any `PYTHONPATH` setup, training never exported the files RTL generation
needed, and the RTL generator itself had several bugs (a scrambled weight
matrix, a missing `endmodule`, an elaboration-failing duplicate
declaration, an array-to-scalar port mismatch, an iverilog-incompatible
array-sliced port connection, an undriven adder enable signal, and a
reversed class-index array direction across one module boundary). All of
that is fixed here — see git log for the specific commits.

## Quickstart

```bash
make build                        # build the dev image (once)
make shell WORK_DIR=/path/to/data # drop into the container, /work mounted
```

Inside the container:

```bash
make tmu-build                    # compile the vendored tmu C extension (once)

# Train (optional -- skip straight to `generate` if you already have
# TAs.txt/weights.txt, e.g. the ones checked in at the repo root):
python3 -m coal_tm.cli train --config /work/training_config.yaml

# Generate RTL from TAs.txt/weights.txt:
python3 -m coal_tm.cli generate --config /work/rtl_config.yaml

# Generate a self-checking testbench (N vectors, expected outputs computed
# by the emulator -- not copied from the test data's label column):
python3 -m coal_tm.cli testbench --config /work/rtl_config.yaml --n-vectors 20

# Run it:
python3 -m coal_tm.cli sim --rtl-dir /work/mnist_model/RTL
python3 -m coal_tm.cli sim --rtl-dir /work/mnist_model/RTL --verilator

# Sanity-check a single input without touching RTL at all:
python3 -m coal_tm.cli emulate --config /work/rtl_config.yaml --input "1,0,1,1,0,..."
```

See `examples/training_config.yaml` and `examples/rtl_config.yaml` for the
config schemas (validated — real errors, not silent misbehavior, for
things like mismatched file sizes, non-existent paths, or an
`adder_stages` that doesn't evenly divide `clauses`). In particular, see
[Known limitation: `bus_width` must be less than `features`](#known-limitation-bus_width-must-be-less-than-features)
below before picking `bus_width` for a small model.

## `train` and `generate` are independent

`coal_tm generate` only needs a `TAs.txt`/`weights.txt` pair — it never
requires having run `coal_tm train` first. Supplying externally-produced
TAs/weights (e.g. `TAs.txt`/`weights.txt` at this repo's root, or ones you
hand-extracted from a TMU model yourself) and generating straight from
those is a first-class path, not a special case.

## How training's export works

`coal_tm train` fits a `TMCoalescedClassifier` and, at the end, writes
`TAs.txt`/`weights.txt` into `output_dir` itself (`coal_tm/artifacts.py`).
Two details worth knowing if you're hand-producing these files:

- **TAs.txt is raw automaton state** (0–255ish), not a pre-thresholded
  0/1 include bit. A literal is Included when the automaton's top state
  bit is set (`state >= 128`) — the same bit TMU's own `get_ta_action()`
  reads.
- **Literal order is interleaved per feature**: index `2i` = feature `i`'s
  positive literal, `2i+1` = feature `i`'s negated literal. TMU's own
  `clause_bank` stores them the other way (all positive literals, then all
  negated) — `coal_tm.artifacts.write_tas()` is the one place that
  reordering happens.
- **A clause with zero included literals is forced to `0`**, not
  vacuously `1` — matching TMU's own C reference
  (`ClauseBank.c`: "Make empty clauses false").

`coal_tm/emulator.py` implements all three of these independently (reading
the exported files, not TMU's in-memory state), so it functions as a
genuine second implementation to check against — both during development
and via `coal_tm emulate`/the generated testbench's expected values.

## Layout

```
coal_tm/            CLI + generators (this is the thing you actually run)
docker/              Dockerfile, docker-compose.yml
examples/            Example YAML configs
tests/               pytest suite (emulator correctness, config validation,
                      RTL generation + simulation regression tests)
utils/RTL_templates/ The 3 static IP blocks coal_tm.rtl copies verbatim
                      (AXI_Interface.sv, new_adder.sv, TM_argmax.sv)
utils/tmu/           Vendored Tsetlin Machine Unified library
legacy/              Everything from 4adrian not carried forward here:
                      the GUI, the old (Vanilla-only, Coalesced-disabled)
                      RTL generator, Vivado synth/deploy scripts, and a
                      few stale/duplicate template files. Kept for
                      reference, not wired into anything.
```

## Known limitation: `bus_width` must be less than `features`

**Pick a `bus_width` that gives each vector at least two packets — never
one where the whole feature vector fits in a single packet.**
`RTLConfig` enforces this (`features > bus_width`, checked at config-load
time, not at simulation time) precisely so a bad choice fails loudly with
a clear message instead of generating hardware that produces wrong
predictions.

Why: the generated core pipelines each inference through several
registered stages after the last packet arrives (`HCB_done` ->
`adder_en` -> `adder_done` -> `argmax`, ~4 cycles) before the predicted
class is ready. Normally the *next* packet's own multi-cycle arrival
naturally gives that pipeline time to drain. When a whole vector fits in
one packet, a new inference can start on literally the next cycle — faster
than the previous one finishes draining — and two inferences overlap in
the same pipeline stage. `axis_wrapper.sv` does assert real AXI-Stream
backpressure (`s00_axis_tready` deasserts while an inference is in flight,
see `Hard_Coded_Inference_Top`'s `busy_countdown` in the generated
`TM_top.sv`) and that closes most of the gap, but not all of it in the
single-packet case — it was observed to still let bad values through.
Genuinely fixing this would mean redesigning the pipeline's own dispatch
logic (not just how long it waits), which is more invasive than this
branch's scope — flagged here rather than attempted, so it's a known,
documented constraint instead of a silent trap.

**What this means in practice:** with `bus_width=64`, any model with more
than 64 features is fine (this is the common case — MNIST-scale models
with hundreds of features give you many packets per vector and comfortable
margin). Only unusually small models (`features <= 64` at that bus width)
hit the limit. If you have one, just pick a narrower `bus_width` (e.g. 32,
16, or 8 — must still divide evenly into a byte-aligned AXI `tkeep` width,
so keep it a multiple of 8) so `features > bus_width` holds; this costs a
few extra packets per inference, not correctness.

If you try to `generate` or `testbench` a config that violates this,
you'll see:

```
features (N) must be greater than bus_width (N) — a model whose features
fit in a single packet gives the generated core no natural gap...
```

That error is the fix working as intended — lower `bus_width`, not a bug
to work around.

## Vendored `tmu` import fix

`utils/tmu`'s own C-extension build script
(`utils/tmu/lib/tmulib_extension_build.py`) names the compiled module
`tmu.tmulib`, and most of the library's internal modules import
accordingly (`from tmu... import ...`). A handful of files had drifted to
`from utils.tmu... import ...` instead (added when the library was nested
under `utils/`, never made consistent) — which meant `import tmu` failed
under *any* `sys.path` setup, before even reaching the compiled extension.
Fixed by normalizing those imports back to the style the rest of the
library — and its own build script — already expect.
`coal_tm/__init__.py` puts `utils/` on `sys.path` so `import tmu` resolves
correctly from anywhere `coal_tm` is imported.

## Out of scope (see `legacy/`)

Vanilla TM training/RTL, the Tkinter GUI, and Vivado synthesis/PYNQ
deployment. All present in `4adrian`, none wired up here — this branch is
deliberately Coalesced-TM-only. Ask if you want any of them revived.

Contributors: Tousif Rahman, Gang Mao, Sidharth Maheshwari, Marcos Sartori, Shengyu Duan, Bob Pattison, Adrian Wheeldon
