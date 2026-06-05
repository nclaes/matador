# Usage

All commands run **inside the container** (`make shell WORK_DIR=/path/to/your/data`).

---

## Step 1 — Prepare training config

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

## Step 2 — Train

```bash
matador train --config /work/training_config.yaml
```

Outputs under `/work/TMIR/`:
- `TM_TMIR_Clauses_<N>_...yaml` + `.npz` — trained model (TMIR format)
- `validation_config.yaml` — ready for `matador validate`
- `rom_inference.py` — standalone provenance script (numpy-only)

---

## Step 3 — Validate model accuracy (optional)

```bash
matador validate --config /work/TMIR/validation_config.yaml
```

---

## Step 4 — Generate RTL

Each backend requires its own config file.

```bash
# Copy the template for the backend you want:
cp examples/vanilla_tiled.yaml     /work/vanilla_tiled.yaml
cp examples/vanilla_hardwired.yaml /work/vanilla_hardwired.yaml

# Generate:
matador generate --backend vanilla_tiled     --config /work/vanilla_tiled.yaml
matador generate --backend vanilla_hardwired --config /work/vanilla_hardwired.yaml
```

Key config fields:

**`vanilla_tiled.yaml`**
```yaml
model_path:      /work/TMIR/<model>.yaml
output_dir:      /work
axis_data_width: 32
fifo_depth:      16
feat_slice:      8     # features per tile column
clause_slice:    8     # clauses evaluated in parallel
```

**`vanilla_hardwired.yaml`**
```yaml
model_path:      /work/TMIR/<model>.yaml
output_dir:      /work
axis_data_width: 32
fifo_depth:      16
pipeline_stages: 3     # adder-tree register stages (0 = combinational)
```

RTL is written to `/work/<backend>/RTL/`.

---

## Step 5 — Simulate

```bash
matador simulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml
```

Runs all testbenches (unit + system) under iverilog.  For Verilator FST waveforms:

```bash
make -C /work/vanilla_tiled/RTL/sim/verilator run
```

---

## Step 6 — Emulate (no simulator required)

```bash
matador emulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml --verify
```

Runs the cycle-accurate Python emulator on embedded test vectors and cross-checks against the reference inference engine.

---

## Step 7 — Provenance report

```bash
matador provenance \
  --model     /work/TMIR/<model>.npz \
  --test-data /work/data/test.txt
```

Produces a JSON report with model fingerprint, dataset SHA-256, accuracy, and confusion matrix.  Sharable without a Matador installation via the companion `rom_inference.py`.

---

## Workspace status

```bash
matador status       # current pipeline state at a glance
matador              # splash + status (interactive terminal only)
```
