# Usage

All commands run **inside the container** (`make shell WORK_DIR=/path/to/your/data`).

Already have Boolean (0/1) training/test data files? Skip straight to
[Step 3 — Prepare training config](#step-3--prepare-training-config). Starting
from raw or external data instead? Steps 1–2 below get you there.
`matador faena` walks through this interactively and skips whichever steps
you say you don't need.

---

## Step 1 — Fetch/materialize raw data (optional)

**If the dataset is already registered** (see `matador list-datasets` — the
shared catalog at `data/Raw_Data_Bank.yaml`), reference it by name, no config
file needed:

```bash
matador ingest --dataset digits --output-dir /work/raw
```

**Otherwise**, describe where it is yourself:

```bash
cp examples/data_source_config.yaml /work/data_source_config.yaml
```

Either a standalone spec (fields copied out of one entry in a catalog like
`data/Raw_Data_Bank.yaml` — this is also how you'd add a *new* dataset to
the shared catalog, for everyone else to reference by name too; see
[Developer.md § Adding a new dataset](Developer.md#adding-a-new-dataset) for
a full worked example), or a pointer into a shared catalog:

```yaml
catalog: /work/data/Raw_Data_Bank.yaml   # a multi-dataset catalog
key: digits                               # one entry in it

# — or, for data you already have on disk —
# key: my_dataset
# source: {kind: local, path: /work/my_raw_data/data.csv}
# extract: {archive: none}
# parse: {reader: csv, delimiter: ",", header: false, label_column: -1, dtype: float32}
# split: {mode: random, test_size: 0.2, seed: 0, stratify: true}
# export: {formats: [npz], path: "{output_dir}/{key}"}
```

```bash
matador ingest --config /work/data_source_config.yaml --output-dir /work/raw
```

Either form downloads (or resolves a local path), extracts, parses, and
splits the raw data, writing `/work/raw/<key>.npz`.

---

## Step 2 — Booleanize (optional)

Turns the raw arrays above (or any x/y `.npz` of your own) into the exact
Boolean format `matador train` expects — no changes needed there.

**Registered datasets can carry a *verified* default encoding recipe**
(`matador list-datasets` shows which ones — `digits`, `statlog`,
`mammographic`, `sensorless_drive`, `mnist` as of the shared
catalog today; verified means it's been checked to reproduce the
documented Boolean shape exactly). Running one straight is safe:

```bash
matador booleanize --dataset digits --raw-dir /work/raw --output-dir /work/booleanised
```

The rest (`sports`, `gesture_phase`, `human_activity`, `emg`) have
no verified default — usually because a feature-engineering step (windowed
statistics, MFCC, ...) sits between raw ingest output and their documented
shape that this pipeline doesn't automate (each one's own `notes` in
`data/Raw_Data_Bank.yaml` explain why). `matador booleanize --dataset
<one of these>` (without `--show-recipe`) refuses to run, rather than
silently applying a guess.

**Every registered dataset can still be inspected, though** — `matador
list-datasets` shows a one-line summary either way; for the full picture:

```bash
matador booleanize --dataset digits --show-recipe
matador booleanize --dataset sports --show-recipe   # works too, prints an
                                                      # UNVERIFIED skeleton +
                                                      # that dataset's own notes
```

Both print a ready-to-use `booleanisation_config.yaml` and exit without
running anything — for a verified dataset it's the real default; for an
unverified one it's a generic quantile-fit thermometer skeleton, clearly
labeled as unvalidated, meant purely as an editable starting point (this is
also how you'd suggest your own encoding for an unverified dataset, or a
different one for a verified dataset — edit the shown YAML, save it, then
run it explicitly with `--config`).

**Either way, to actually use your own recipe** — starting from scratch,
or from a registered dataset's recipe (verified or not) as a base to edit:

```bash
cp examples/booleanisation_config.yaml /work/booleanisation_config.yaml
# — or —
matador booleanize --dataset digits --show-recipe > /work/booleanisation_config.yaml
```

```yaml
raw_npz: /work/raw/digits.npz
name: digits
output_dir: /work/booleanised

default_encoder:
  encoder: thermometer   # thermometer | threshold | onehot | passthrough
                         # (see Developer.md § Adding a new booleanisation
                         # technique to add another one)
  bits: 8
  range: [0, 16]

test_size: 0.2   # only used when raw_npz has unsplit x/y
seed: 0
stratify: true
```

```bash
matador booleanize --config /work/booleanisation_config.yaml
```

Either form outputs `/work/booleanised/digits_train.txt` / `digits_test.txt`
(space-separated 0/1, last column = label) and a `digits_report.json` — feed these
straight into `train_data`/`test_data` below.

---

## Step 3 — Prepare training config

```bash
cp examples/training_config.yaml /work/training_config.yaml
```

Edit `/work/training_config.yaml`:

```yaml
tm_type: vanilla          # "vanilla" or "coalesced"
clauses: 400              # total clauses (must be even)
classes: 10               # output classes
features: 512             # Boolean input features per sample
s: 5.0                    # specificity (float > 1.0)
T: 200                    # voting threshold
epochs: 50
max_included_literals: 32
seed: 42

train_data: /work/data/train.txt   # space-separated Boolean, last col = label
test_data:  /work/data/test.txt
output_dir: /work
```

**Data format:** space-separated values, all features must be Boolean (0 or 1), last column is the integer class label (0-indexed).

---

## Step 4 — Train

```bash
matador train --config /work/training_config.yaml
```

Outputs under `/work/TMIR/<model_name>/` — **namespaced per model**, so
training several models (different datasets, or just different
hyperparameters) into the same `/work` never overwrites an earlier one.
`model_name` defaults to `train_data`'s filename stem (`digits_train.txt` →
`digits` — matches `matador booleanize`'s own naming, so the
`ingest`/`booleanize`/`train` chain needs no extra configuration); set
`model_name:` explicitly in `training_config.yaml` to override it.

- `TM_TMIR_Clauses_<N>_...yaml` + `.npz` — trained model (TMIR format)
- `validation_config.yaml` — ready for `matador validate`
- `rom_inference.py` — standalone provenance script (numpy-only)

---

## Step 5 — Validate model accuracy (optional)

```bash
matador validate --config /work/TMIR/<model_name>/validation_config.yaml
```

---

## Step 6 — Generate RTL

Each backend requires its own config file. `matador list-backends` shows the
full set (see [Developer.md § Adding a new accelerator
backend](Developer.md#adding-a-new-accelerator-backend) to add another one).

```bash
# Copy the template for the backend you want:
cp examples/vanilla_tiled.yaml     /work/vanilla_tiled.yaml
cp examples/vanilla_hardwired.yaml /work/vanilla_hardwired.yaml
cp examples/vanilla_gp_tiled.yaml  /work/vanilla_gp_tiled.yaml

# Generate:
matador generate --backend vanilla_tiled     --config /work/vanilla_tiled.yaml
matador generate --backend vanilla_hardwired --config /work/vanilla_hardwired.yaml
matador generate --backend vanilla_gp_tiled  --config /work/vanilla_gp_tiled.yaml
```

Key config fields:

**`vanilla_tiled.yaml`**
```yaml
model_path:      /work/TMIR/<model_name>/<model>.yaml
output_dir:      /work
axis_data_width: 32
fifo_depth:      16
feat_slice:      8     # features per tile column
clause_slice:    8     # clauses evaluated in parallel
```

**`vanilla_hardwired.yaml`**
```yaml
model_path:      /work/TMIR/<model_name>/<model>.yaml
output_dir:      /work
axis_data_width: 32
fifo_depth:      16
pipeline_stages: 3     # adder-tree register stages (0 = combinational)
```

**`vanilla_gp_tiled.yaml`** — runtime-reprogrammable: synthesized once at a
capacity that fits `target_fpga`'s BRAM budget, then reprogrammed with any
model that fits inside it via a runtime `CMD_LOAD` — no resynthesis needed
to swap models (see Step 8).
```yaml
model_path:        /work/TMIR/<model_name>/<model>.yaml
output_dir:        /work
target_fpga:        xc7z020   # or xcku040; bram_bits_budget for anything else
feat_slice:         32
clause_slice:       32
max_features:       512
max_clauses_total:  256
max_classes:        32
```

RTL is written to `/work/<backend>/RTL/`.

---

## Step 7 — Simulate

```bash
matador simulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml
```

Runs all testbenches (unit + system) under iverilog.  For Verilator FST waveforms:

```bash
make -C /work/vanilla_tiled/RTL/sim/verilator run
```

---

## Step 8 — Prove multi-model reprogramming (vanilla_gp_tiled only, optional)

```bash
cp examples/reprogram_config.yaml /work/reprogram_config.yaml
```

Lists an ordered sequence of `{model, dataset}` steps — each a trained TMIR
model plus (optionally) a booleanized dataset to draw test vectors from.
This is exactly why `matador train`'s output is namespaced per model (Step
4) — training `digits` and `sports` into the same `/work` leaves both
models' files intact, ready to reference here by their own paths:

```yaml
steps:
  - model: /work/TMIR/digits/TM_TMIR_Clauses_400_s_value_5_T_value_200_epochs_50_max_literals_32.yaml
    dataset: /work/booleanised/digits_test.txt
    n_samples: 20
  - model: /work/TMIR/sports/TM_TMIR_Clauses_400_s_value_5_T_value_200_epochs_50_max_literals_32.yaml
    # no dataset -> uses this model's own embedded test vectors
```

```bash
matador reprogram-suite --backend vanilla_gp_tiled \
    --config /work/vanilla_gp_tiled.yaml \
    --reprogram-config /work/reprogram_config.yaml
```

Requires Step 6's `matador generate` to have already run. Builds a
testbench that reprograms the already-synthesized bundle across every step
in one continuous run, and prints the exact `iverilog` command to compile
and run it — a real, checkable demonstration that the hardware reprograms
correctly with your own models, not just a single one.

---

## Step 9 — Emulate (no simulator required)

```bash
matador emulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml --verify
```

Runs the cycle-accurate Python emulator on embedded test vectors and cross-checks against the reference inference engine.

---

## Step 10 — Provenance report

```bash
matador provenance \
  --model     /work/TMIR/<model_name>/<model>.npz \
  --test-data /work/data/test.txt
```

Produces a JSON report with model fingerprint, dataset SHA-256, accuracy, and confusion matrix.  Sharable without a Matador installation via the companion `rom_inference.py`.

---

## Workspace status

```bash
matador status    # current pipeline state at a glance
matador            # splash + status (interactive terminal only)
matador registry    # everything registered: backends + datasets
matador faena       # interactive wizard — walks you through whichever
                    # steps above your /work doesn't have yet
```

`matador status`/`matador` show one section per pipeline stage (raw data,
Boolean data, training config, trained models, RTL/backends), each listing
every item actually in your workspace — not just the newest — with the
specific next command shown directly under any item that isn't finished yet
(e.g. one `matador booleanize --dataset ...` line per raw dataset not yet
booleanized). This is exactly why `matador train`'s output is namespaced per
model (Step 4): several datasets/models can coexist in one `/work` and the
dashboard tracks each of them individually.

## Starting over

```bash
matador clean --dry-run   # preview what would be removed, deletes nothing
matador clean              # remove generated output (TMIR/, raw/, booleanised/,
                            # per-backend RTL/) — keeps your edited config YAMLs
matador clean --configs    # also remove the config YAMLs you edited
matador clean --all -y     # wipe /work entirely, no confirmation prompt
```

Always previews what it's about to remove and asks for confirmation unless
`-y`/`--yes` is given. The default scope only touches directories matador
itself writes — anything else you've placed in `/work` is left alone unless
you pass `--all`.
