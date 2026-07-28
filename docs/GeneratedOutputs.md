# Generated Outputs — a field guide to `/work`

Every `matador` command in the pipeline writes into a workspace directory
(conventionally `/work` inside the dev container — see [Setup.md](Setup.md)
and `make shell WORK_DIR=...`). This page is a map of exactly what shows up
there, stage by stage, what each file is for, and what's safe to delete.

If you just want to run the pipeline, see [Usage.md](Usage.md) instead —
this page assumes you've already run some commands and are looking at a
`/work` full of unfamiliar files.

**Quick answer if you're in a hurry:** almost everything under `/work` is
*regenerated output* — safe to delete and rebuild by re-running the command
that made it. The only things that are expensive to lose are `TMIR/` (a
trained model — retraining costs real time) and `booleanised/` (booleanized
data — cheap to rebuild if `raw/` is still there, otherwise needs a
re-ingest). `matador clean --dry-run` shows you exactly what a full reset
would remove before it removes anything.

---

## `matador ingest --dataset X --output-dir /work/raw`

```
raw/
├── _cache/<key>/        # downloaded archive(s), as fetched from the source URL
├── _extracted/<key>/    # unpacked contents of the cache archive
├── <key>.npz            # the actual output — what booleanize reads
└── <key>_train.csv      # optional human-readable duplicate of the .npz
    <key>_test.csv       # (skipped automatically if it'd be >100MB)
```

- **`_cache/` and `_extracted/`** are working scratch space, not the output
  you need downstream. Safe to delete; a re-run of `ingest` repopulates them
  from the source URL (or from `_cache/` if the download itself already
  succeeded, without re-fetching).
- **`<key>.npz`** is the real deliverable — this is what `matador booleanize
  --dataset X` reads. Don't delete it unless you're happy to re-fetch.
- **`.csv` files** are a convenience for eyeballing the data in a text
  editor/spreadsheet; matador itself never reads them back.

## `matador booleanize --dataset X --output-dir /work/booleanised`

```
booleanised/
├── <name>_train.txt    # 0/1 features + integer label column, space-separated
├── <name>_test.txt
└── <name>_report.json  # per-column encoder choices, class distribution,
                         #  n_features_bool, n_classes
```

`<name>_train.txt`/`<name>_test.txt` are what `matador train` consumes
(`train_data:`/`test_data:` in a training config), and their filename stem
(`digits` from `digits_train.txt`) is also the *default* `model_name` for
whatever gets trained from them — see the next stage. `<name>_report.json`
is worth opening once: it tells you the exact `features:`/`classes:` values
your training config needs.

## `matador train --config training_config.yaml`

Output lands under `<output_dir>/TMIR/<model_name>/` — **one subdirectory
per model**, not per dataset, so training several differently-configured
models from the same or different datasets into the same workspace never
overwrites an earlier one. `model_name` defaults to the training data's
filename stem; set `model_name:` explicitly in the config to train a second
model from the *same* dataset without colliding.

```
TMIR/<model_name>/
├── TM_TMIR_Clauses_<c>_s_value_<s>_T_value_<T>_epochs_<e>_max_literals_<m>.yaml
│                          # the trained model, human-readable (TMIR format)
├── ...same stem....npz    # the same model, compact binary — this is the
│                          #  one every other command actually loads
├── validation_config.yaml # {model_path, test_data} — feeds `matador validate`
├── ta_actions.npy         # the trained TA "Include" decisions as a plain
│                          #  numpy bool array (n_classes, clauses_per_class,
│                          #  n_literals) — the actual learned "program"
├── model_metadata.json    # geometry/hyperparameters rom_inference.py needs
│                          #  to interpret ta_actions.npy on its own
└── rom_inference.py       # standalone, numpy-only inference + verification
                           #  script — works with just this directory copied
                           #  elsewhere, no matador install required
```

This directory *is* the trained model. Everything downstream (`generate`,
`simulate`, `emulate`, `validate`, `provenance`, `reprogram-suite`) reads
from here — treat it like you would any other trained-model checkpoint:
worth keeping, not casually deleted, since re-creating it means retraining.

## `matador generate --backend X --config accel_config.yaml`

Output lands under `<output_dir>/<backend>/RTL/`. All three backends share
this shape:

```
RTL/
├── src/       synthesizable Verilog-2001 — the actual hardware deliverable
├── tb/        testbenches (one per src/ module, plus a full-system one)
├── sim/       run_iverilog.sh, lint_verilator.sh, run_xsim.sh, waves.sh,
│              *.gtkw (curated GTKWave waveform views), verilator/Makefile
└── README.md  self-contained explanation of THIS specific build — start
               here, it's generated fresh for every model/config and is
               more detailed than this page for backend-specific quirks
```

`RTL/README.md` is intentionally the most authoritative doc for anything
backend-specific (parameter tables, latency, memory map) — this page only
covers what's common across backends plus the differences worth knowing
before you go looking.

**`vanilla_tiled` / `vanilla_hardwired`** bake the trained model directly
into `src/` (e.g. `src/tm_accelerator.v`'s ROM, or `hcb_blocks.v`'s wired-in
TA states) — one RTL build *is* one model. Regenerating with a different
`model_path:` produces a different `src/`.

**`vanilla_gp_tiled`** is different: it's a *runtime-reprogrammable* core.
`src/` is capacity-parameterized (sized to fit a target FPGA's BRAM budget)
but generic — no model is baked in. The model is instead encoded as a
runtime payload, which is why `sim/` has extra files:

```
sim/
├── model_stimulus.memh   # this model's CMD_LOAD (weights) + CMD_INFER
│                         #  (its own embedded test vectors), as an AXI-
│                         #  Stream beat sequence ready to replay
├── model_expected.memh   # the predicted class for each test vector, i.e.
│                         #  what a passing simulation must produce
├── test_vectors.txt      # the same test vectors, human-readable
├── capacity.json         # THIS bitstream's synthesized capacity ceiling
│                         #  (max_classes/max_clauses_total/etc.) — any
│                         #  model you try later must fit inside these
├── model_reference.json  # this model's raw geometry/weights, in the
│                         #  schema gen_vectors.py expects
├── tm_emulator.py         # standalone tools (no matador install needed)
└── gen_vectors.py         #  to build CMD_LOAD/CMD_INFER vectors for a
                           #  DIFFERENT model against this SAME bitstream,
                           #  i.e. to try reprogramming it by hand
```

and `RTL/provenance.json` (root, not `sim/`) records which trained model's
weights `model_stimulus.memh` currently encodes — since the same bitstream
can be reprogrammed with many models over its life, this is what tells you
"what's actually loaded right now" without re-deriving it from the RTL.

You'll also see `gp_stimulus.memh`, `gp_expected.memh`, `gp_sizes.vh`,
`gp_small_load.memh`, and `gp_small_infer.memh` in `sim/` — these are **not**
about your model at all. They're a fixed, ships-with-every-build
protocol-conformance fixture (exercised by `tb_system_gp.v`) that proves the
CMD_LOAD/CMD_INFER wire protocol itself works, independent of what model you
trained. Safe to ignore unless you're debugging the protocol layer.

## `matador simulate --backend X --config accel_config.yaml`

Runs in-place inside the `RTL/` tree already written by `generate` — no new
top-level output directory. It does add a couple of things to `sim/`:
compiled testbench binaries (`sim/<tb_name>`, from iverilog) and, if
Verilator is installed, `sim/verilator/obj_dir/` (build artifacts plus a
`.fst` waveform). Everything else prints straight to stdout — there are no
separate log files to go hunting for.

## `matador emulate --backend X --config accel_config.yaml [--verify] [--trace out.json]`

Pure in-memory software emulation — writes **nothing** by default; results
print to stdout. The one exception is `--trace out.json`, which writes
exactly that one file (a cycle-by-cycle JSON trace of every architectural
event: FIFO, ROM reads, clause evaluation, score accumulation, argmax) —
useful for debugging a mismatch against the RTL, not something you need for
normal use.

## `matador reprogram-suite --backend vanilla_gp_tiled ...` (optional)

Requires `generate` to have already run — augments the *existing* `RTL/`
tree in place rather than creating a new one, since the point is proving
several models coexist on one already-synthesized core:

```
tb/  tb_reprogram_suite.v          # LOAD -> INFER -> LOAD -> INFER -> ...
sim/ reprogram_stimulus.memh
sim/ reprogram_expected.memh
sim/ reprogram_manifest.txt        # which beat belongs to which {model, dataset} step
sim/ tb_reprogram_suite.gtkw
```

`src/` is untouched — the whole point is that the same synthesized core is
reused, not rebuilt, across every model in the sequence.

---

## What's safe to delete

| Path | Safe to delete? | To rebuild |
|---|---|---|
| `raw/_cache/`, `raw/_extracted/` | Yes, always | re-run `matador ingest` |
| `raw/<key>.npz` / `.csv` | Yes, if you'll re-ingest | re-run `matador ingest` |
| `booleanised/` | Yes, if `raw/` is still there | re-run `matador booleanize` |
| `TMIR/<model_name>/` | **Costs real time to rebuild** | re-run `matador train` (retraining) |
| `<backend>/RTL/` | Yes | re-run `matador generate` (needs `TMIR/` to still exist) |
| `<backend>/RTL/sim/verilator/obj_dir/` | Yes, always | re-run `matador simulate` |

`matador clean --dry-run` walks the actual workspace and shows precisely
what it would remove, rather than relying on this table being current —
prefer that over deleting by hand.
