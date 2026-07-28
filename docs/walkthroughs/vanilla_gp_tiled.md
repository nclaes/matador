# Walkthrough — vanilla_gp_tiled: reprogramming across sports, MNIST, and a third model

Everything below was run for real inside the dev container (`make shell
WORK_DIR=...`) — the commands, output, and numbers are copied verbatim from
an actual run, not illustrative. If you follow along exactly you should see
the same shape of output (exact accuracy/timing will vary a little run to
run).

**What this demonstrates:**
1. Training two very different models — `sports` (19 classes, an
   **unverified** booleanisation recipe you have to edit yourself) and
   `mnist` (10 classes, a verified default) — into the same workspace.
2. Generating one `vanilla_gp_tiled` bitstream sized to fit both, and
   proving it actually reprograms between them with `matador
   reprogram-suite`.
3. Training a **third** model (`digits`) afterward and handing an RTL
   engineer test vectors for it via `matador export-model-json` —
   without regenerating the RTL, and without them needing matador
   installed at all.

If you just want the command reference, see [Usage.md](../Usage.md). This
page is the "why does it say that" narrative version, aimed at someone
doing this for the first time.

---

## Step 1 — Train `sports` (unverified recipe)

```bash
matador ingest --dataset sports --output-dir /work/raw
```

```
Ingesting 'sports' (Daily and Sports Activities (DSA))
  source: http_zip

WARNING: [sports] labels were not 0-indexed; shifted by -1 (raw min was 1)
WARNING: [sports] csv export skipped: estimated size ~436MB exceeds the 100MB threshold -- npz already has everything matador booleanize needs
  train samples: 7296  test samples: 1824
```

`sports` is one of the datasets `matador list-datasets` marks as having **no
verified booleanisation recipe** (see the table in the main
[README](../../README.md)) — the upstream data needs per-segment feature
engineering (min/max/mean/std/skew/kurtosis over a windowed signal) that
isn't automated, so `matador booleanize --dataset sports` (without
`--show-recipe`) refuses to run rather than silently guessing:

```bash
matador booleanize --dataset sports --show-recipe --raw-dir /work/raw --output-dir /work/booleanised
```

```yaml
# UNVERIFIED starting skeleton for 'sports' (Daily and Sports Activities (DSA)).
# No catalog-recorded default exists for this dataset -- the encoder below is a
# generic numeric guess (quantile-fit thermometer), not validated against any
# known original encoding. Read the notes below before trusting it as-is:
#   Upstream is 19 activities; ...
raw_npz: /work/raw/sports.npz
name: sports
output_dir: /work/booleanised
default_encoder:
  encoder: thermometer
  bits: 8
  quantile: true
```

**What to change:** the raw data here is 125×45 = 5625 floats per sample
(a whole windowed segment, flattened) — the default 8-bit thermometer
skeleton would encode that as **45,000 Boolean bits**, which trains and
simulates fine but is unnecessarily large for a first pass. Since this is
an unverified recipe anyway (i.e. you're expected to tune it), switch to a
1-bit `threshold` encoder — 5625 raw features become exactly 5625 Boolean
bits instead:

```yaml
# /work/sports_boolconfig.yaml
raw_npz: /work/raw/sports.npz
name: sports
output_dir: /work/booleanised
default_encoder:
  encoder: threshold
  bits: 1
```

```bash
matador booleanize --config /work/sports_boolconfig.yaml
```

```
Booleanizing /work/raw/sports.npz -> sports_{train,test}.txt

  train samples: 7296  test samples: 1824
  5625 raw features -> 5625 Boolean bits
```

Train it (a small clause count / one epoch — this is a demo, not a tuned
model):

```yaml
# /work/sports_train.yaml
tm_type: vanilla
clauses: 20
classes: 19
features: 5625
s: 5.0
T: 15
epochs: 1
max_included_literals: 32
seed: 42
train_data: /work/booleanised/sports_train.txt
test_data: /work/booleanised/sports_test.txt
model_name: sports
output_dir: /work
```

```bash
matador train --config /work/sports_train.yaml
```

```
Training for 1 epoch(s)…
INFO: Epoch 1/1 — accuracy: 55.70%  train: 3.53s  test: 3.27s
INFO: TMIR written to /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml  (+.npz)
```

55.70% on 19 classes with one epoch and 20 clauses is far from tuned (random
guessing would be ~5.3%), which is expected and fine — this walkthrough is
about the *pipeline*, not model quality. Increase `clauses`/`epochs` for a
real model.

---

## Step 2 — Train `mnist` (verified recipe)

`mnist` has a verified default recipe, so no `--show-recipe` detour is
needed:

```bash
matador ingest --dataset mnist --output-dir /work/raw
matador booleanize --dataset mnist --raw-dir /work/raw --output-dir /work/booleanised
```

```
  train samples: 60000  test samples: 10000
  784 raw features -> 784 Boolean bits
```

```yaml
# /work/mnist_train.yaml
tm_type: vanilla
clauses: 20
classes: 10
features: 784
s: 5.0
T: 15
epochs: 1
max_included_literals: 32
seed: 42
train_data: /work/booleanised/mnist_train.txt
test_data: /work/booleanised/mnist_test.txt
model_name: mnist
output_dir: /work
```

```bash
matador train --config /work/mnist_train.yaml
```

```
Training for 1 epoch(s)…
INFO: Epoch 1/1 — accuracy: 84.65%  train: 6.62s  test: 1.31s
INFO: TMIR written to /work/TMIR/mnist/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml  (+.npz)
```

`/work/TMIR/` now has two independent, non-colliding model directories —
`sports/` and `mnist/` — because `matador train` namespaces output by
`model_name` (defaulting to the training data's filename stem). This is
exactly why the next step can reference both by path without either
overwriting the other.

---

## Step 3 — Generate `vanilla_gp_tiled`, sized to fit both models

`vanilla_gp_tiled` is synthesized **once**, at a compile-time capacity
ceiling, then reprogrammed at runtime — so the config has to describe the
*largest* geometry you intend to ever load, not any one model's exact
shape. Between the two models trained above:

| | sports | mnist | this bundle needs |
|---|---|---|---|
| features | 5625 | 784 | ≥ 5625 |
| classes | 19 | 10 | ≥ 19 |
| clauses (20/class) | 380 | 200 | ≥ 380 |

```yaml
# /work/vanilla_gp_tiled.yaml
model_path: /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml
output_dir: /work
target_fpga: xcku040
max_features: 5632
max_clauses_total: 384
max_classes: 20
```

(`max_features`/`max_clauses_total` are rounded up to the nearest multiple
of `feat_slice`/`clause_slice` — 32 by default — hence 5632/384 rather than
the exact 5625/380.)

**What to change / a real failure mode worth seeing once:** the first
attempt here used `target_fpga: xc7z020` (the smaller of the two built-in
targets) and failed immediately, before writing anything:

```
Error: Invalid config: 1 validation error for GPTiledAcceleratorConfig
  Value error, tile_mem for this config needs 4,325,376 bits (2 x 5,632
  features_padded x 384 clauses_padded), which exceeds the 3,675,000-bit
  budget for target_fpga='xc7z020' ...
```

This is the capacity check working as intended — better to fail at
`matador generate` than discover it doesn't fit at synthesis time.
Switching to `target_fpga: xcku040` (a bigger device, 38,000,000 BRAM bits
vs. `xc7z020`'s 4,900,000) fixed it, which is the config above.

```bash
matador generate --backend vanilla_gp_tiled --config /work/vanilla_gp_tiled.yaml
```

```
  vanilla TM  |  5625 features  |  19 classes  |  380 clauses
  Generating [vanilla_gp_tiled]…

RTL written to: /work/vanilla_gp_tiled/RTL
```

`model_path` pointed at `sports` here, but that only determines what
`sim/model_stimulus.memh`/`test_vectors.txt` demonstrate *by default* (see
[GeneratedOutputs.md](../GeneratedOutputs.md) for the full file map) — the
capacity fields, not `model_path`, are what the synthesized core actually
enforces. Either model (or any model within capacity) can be loaded at
runtime regardless of which one `generate` happened to be pointed at.

---

## Step 4 — Prove it actually reprograms, with both real models

```yaml
# /work/reprogram_config.yaml
steps:
  - model: /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml
  - model: /work/TMIR/mnist/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml
```

Neither step lists a `dataset:`/`vectors:` — omitting both falls back to
each model's own embedded test vectors, whose expected classes come from
matador's independent mathematical reference model (not from anything that
builds the RTL stimulus itself — the strongest, least circular check
available).

```bash
matador reprogram-suite --backend vanilla_gp_tiled \
    --config /work/vanilla_gp_tiled.yaml \
    --reprogram-config /work/reprogram_config.yaml
```

```
Building reprogram suite for vanilla_gp_tiled at /work/vanilla_gp_tiled/RTL (2 step(s))

Wrote:
  /work/vanilla_gp_tiled/RTL/tb/tb_reprogram_suite.v
  /work/vanilla_gp_tiled/RTL/sim/reprogram_stimulus.memh
  /work/vanilla_gp_tiled/RTL/sim/reprogram_expected.memh
  /work/vanilla_gp_tiled/RTL/sim/reprogram_manifest.txt
```

`reprogram_manifest.txt` maps every beat to which {model, vector} it
belongs to — this is the ground truth for the console output below:

```
step 0: model='...sports...'  vectors from: this model's own embedded verification.test_vectors
  beat 0: LOAD ack
  beat 1: vector[0] -> expected_class 14
  ...
  beat 8: vector[7] -> expected_class 4
  ...

step 1: model='...mnist...'  vectors from: this model's own embedded verification.test_vectors
  beat 11: LOAD ack
  beat 12: vector[0] -> expected_class 7
  ...
```

Compile and run it (the exact command `reprogram-suite` prints):

```bash
cd /work/vanilla_gp_tiled/RTL
iverilog -g2001 -Wall -Wno-timescale -o sim/tb_reprogram_suite \
    src/axis_fifo.v src/clause_eval.v src/tile_mem.v src/score_acc_rt.v src/argmax_rt.v src/tm_accel_gp.v \
    tb/tb_reprogram_suite.v
(cd sim && vvp tb_reprogram_suite)
```

```
PASS beat[0]: data=a5000840 last=1     <- sports LOAD ack (status=0x00 OK, info=0x0840=2112 tiles)
PASS beat[1]: data=0000000e last=0     <- 0x0e = 14, matches manifest's expected_class 14
PASS beat[2]: data=0000000e last=0
...
PASS beat[8]: data=00000004 last=0     <- 0x04 = 4, matches manifest's expected_class 4
...
PASS beat[10]: data=0000000e last=1
PASS beat[11]: data=a50000af last=1    <- mnist LOAD ack (info=0x00af tiles)
PASS beat[12]: data=00000007 last=0    <- 0x07 = 7, matches manifest's expected_class 7
...
PASS beat[21]: data=00000009 last=1
tb_reprogram_suite: ALL TESTS PASSED (22 beats checked)
```

All 22 beats — one LOAD ack + 10 predictions for `sports`, then one LOAD ack
+ 10 predictions for `mnist` — pass, in one continuous simulation run, on
one synthesized core. This is what "runtime-reprogrammable" actually means
in practice: no resynthesis between the two `LOAD` packets, just a new AXI-
Stream burst.

Verilator confirms the same result independently:

```bash
make -C sim/verilator run ARGS="--stim ../reprogram_stimulus.memh --exp ../reprogram_expected.memh"
```

To view it: `bash sim/waves.sh tb_reprogram_suite` (see
[GeneratedOutputs.md](../GeneratedOutputs.md) and the generated
`RTL/README.md` § 2 for how to read the waveform against
`reprogram_manifest.txt`).

---

## Step 5 — Add a third model, without touching the RTL at all

Suppose a third model gets trained *after* the bitstream above was already
built — here, `digits` (a small, fast, verified dataset, picked for
variety):

```bash
matador ingest --dataset digits --output-dir /work/raw
matador booleanize --dataset digits --raw-dir /work/raw --output-dir /work/booleanised
```

```yaml
# /work/digits_train.yaml
tm_type: vanilla
clauses: 20
classes: 10
features: 512
s: 5.0
T: 15
epochs: 2
max_included_literals: 32
seed: 42
train_data: /work/booleanised/digits_train.txt
test_data: /work/booleanised/digits_test.txt
model_name: digits
output_dir: /work
```

```bash
matador train --config /work/digits_train.yaml
```

512 features / 10 classes / 200 clauses (20 × 10) fits comfortably inside
the bundle's existing 5632/384/20 capacity — no regeneration needed. Now
simulate handing this off to **an RTL engineer who does not have matador
installed** — only the already-built `RTL/` folder:

```bash
matador export-model-json --backend vanilla_gp_tiled \
    --model /work/TMIR/digits/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml \
    -o /work/vanilla_gp_tiled/RTL/sim/digits_model.json
```

```
Exported .../TM_TMIR_...digits....yaml -> /work/vanilla_gp_tiled/RTL/sim/digits_model.json

Hand this file (and sim/gen_vectors.py + sim/tm_emulator.py, if the
recipient doesn't already have them from an earlier bundle) to whoever
is building test vectors -- no matador install needed on their end:
  python3 gen_vectors.py combined digits_model.json --random -n 20 \
      -o my_stim.memh --expected my_exp.memh --testbench tb_my_model.v
```

To actually prove "no matador needed", this was verified in a directory
containing **only** `sim/gen_vectors.py`, `sim/tm_emulator.py`,
`sim/capacity.json`, `sim/digits_model.json`, and `src/` — no `matador`
package importable at all:

```bash
cd sim
python3 gen_vectors.py combined digits_model.json --random -n 15 \
    -o my_stim.memh --expected my_exp.memh --testbench tb_digits.v --seed 3
iverilog -g2001 -Wall -Wno-timescale -o tb_digits \
    ../src/axis_fifo.v ../src/clause_eval.v ../src/tile_mem.v \
    ../src/score_acc_rt.v ../src/argmax_rt.v ../src/tm_accel_gp.v \
    tb_digits.v
vvp tb_digits
```

```
PASS beat[0]: data=a5000070 last=1
PASS beat[1]: data=00000001 last=0
...
PASS beat[15]: data=00000004 last=1
tb_TM_TMIR_..._digits...: ALL TESTS PASSED (16 beats checked)
```

15 random vectors, generated and checked entirely from the standalone
bundle, no matador involved — this is the concrete answer to "does the RTL
engineer actually have everything they need."

---

## Summary — what changed at each stage

| Stage | Command | What it produced | What you'd change for your own data |
|---|---|---|---|
| 1 | `matador booleanize --show-recipe` then `--config` | `sports_{train,test}.txt`, 5625 bits, 19 classes | The encoder (`threshold` vs `thermometer`, bit width) — this dataset has no verified default, so you're expected to tune it |
| 2 | `matador booleanize --dataset mnist` | `mnist_{train,test}.txt`, 784 bits, 10 classes | Nothing — verified datasets just work |
| 3 | `matador generate --backend vanilla_gp_tiled` | `RTL/` sized for the *larger* of your models | `max_features`/`max_clauses_total`/`max_classes`/`target_fpga` — set these to your largest anticipated model, not any one model's exact shape |
| 4 | `matador reprogram-suite` | A testbench proving multiple real models reprogram correctly on one core | The `steps:` list — one entry per model you want to prove reprogramming across |
| 5 | `matador export-model-json` | A plain JSON file, usable with zero matador install | Nothing about the RTL — this only ever produces a JSON file, the bitstream is untouched |

See also: [Usage.md](../Usage.md) for the command reference, and
[GeneratedOutputs.md](../GeneratedOutputs.md) for what every file in
`RTL/` actually is.
