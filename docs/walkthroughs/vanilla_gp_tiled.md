# Walkthrough — vanilla_gp_tiled: reprogramming across sports, Statlog, and a third model

Everything below was run for real inside the dev container (`make shell
WORK_DIR=...`) — the commands, output, and numbers are copied verbatim from
an actual run, not illustrative. If you follow along exactly you should see
the same shape of output (exact accuracy/timing will vary a little run to
run).

**What this demonstrates:**
1. Training two very different models — `sports` (19 classes, an
   **unverified** booleanisation recipe you have to edit yourself) and
   `statlog` (4 classes, a verified default) — into the same workspace.
2. Generating one `vanilla_gp_tiled` bitstream sized to fit both, and
   proving it actually reprograms between them with `matador
   reprogram-suite` — checked under **both iverilog and Verilator**.
3. Training a **third** model (`human_activity`) afterward and handing an
   RTL engineer test vectors for it via `matador export-model-json` —
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
INFO: Epoch 1/1 — accuracy: 55.70%  train: 1.77s  test: 1.63s
INFO: TMIR written to /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml  (+.npz)
```

55.70% on 19 classes with one epoch and 20 clauses is far from tuned (random
guessing would be ~5.3%), which is expected and fine — this walkthrough is
about the *pipeline*, not model quality. Increase `clauses`/`epochs` for a
real model. (This result is exactly reproducible — a second, independent run
with the same seed landed on 55.70% again, byte-for-byte.)

---

## Step 2 — Train `statlog` (verified recipe)

`statlog` (Statlog Vehicle Silhouettes, 4 classes) has a verified default
recipe, so no `--show-recipe` detour is needed:

```bash
matador ingest --dataset statlog --output-dir /work/raw
matador booleanize --dataset statlog --raw-dir /work/raw --output-dir /work/booleanised
```

```
Booleanizing /work/raw/statlog.npz -> statlog_{train,test}.txt

  train samples: 677  test samples: 169
  18 raw features -> 288 Boolean bits
```

(18 raw features × a 16-bit quantile thermometer, per `data/Raw_Data_Bank.yaml`'s
recorded default for this dataset — `matador registry` shows the same
`{encoder: thermometer, bits: 16, quantile: true}` for anyone who wants to
double-check.)

```yaml
# /work/statlog_train.yaml
tm_type: vanilla
clauses: 20
classes: 4
features: 288
s: 5.0
T: 15
epochs: 2
max_included_literals: 32
seed: 42
train_data: /work/booleanised/statlog_train.txt
test_data: /work/booleanised/statlog_test.txt
model_name: statlog
output_dir: /work
```

```bash
matador train --config /work/statlog_train.yaml
```

```
INFO: Epoch 1/2 — accuracy: 50.30%  train: 0.06s  test: 0.01s
INFO: Epoch 2/2 — accuracy: 54.44%  train: 0.05s  test: 0.01s
INFO: TMIR written to /work/TMIR/statlog/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml  (+.npz)
```

`/work/TMIR/` now has two independent, non-colliding model directories —
`sports/` and `statlog/` — because `matador train` namespaces output by
`model_name` (defaulting to the training data's filename stem). This is
exactly why the next step can reference both by path without either
overwriting the other.

---

## Step 3 — Generate `vanilla_gp_tiled`, sized to fit both models

`vanilla_gp_tiled` is synthesized **once**, at a compile-time capacity
ceiling, then reprogrammed at runtime — so the config has to describe the
*largest* geometry you intend to ever load, not any one model's exact
shape. Between the two models trained above:

| | sports | statlog | this bundle needs |
|---|---|---|---|
| features | 5625 | 288 | ≥ 5625 |
| classes | 19 | 4 | ≥ 19 |
| clauses (20/class) | 380 | 80 | ≥ 380 |

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
vs. `xc7z020`'s 4,900,000) fixed it, which is the config above. Sports is
still what dominates this bundle's capacity — swapping the second model
from `mnist` to the much smaller `statlog` (288 vs. 784 features, 80 vs.
200 clauses) doesn't change the story at all; `xcku040` is required either
way, purely because of `sports`.

```bash
matador generate --backend vanilla_gp_tiled --config /work/vanilla_gp_tiled.yaml
```

```
Loading model: /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml
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

## Step 4 — Prove it actually reprograms, with both real models (iverilog *and* Verilator)

```yaml
# /work/reprogram_config.yaml
steps:
  - model: /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml
  - model: /work/TMIR/statlog/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml
```

Neither step lists a `dataset:`/`vectors:` — omitting both falls back to
each model's own embedded test vectors, whose expected classes come from
matador's independent mathematical reference model (not from anything that
builds the RTL stimulus itself — the strongest, least circular check
available). Those embedded vectors are sampled round-robin across the
ground-truth classes present in the test set (seeded off `config.seed`,
reproducible) rather than just the first N rows of the test file — several
datasets, `sports` included, store their booleanized test file as
contiguous per-class blocks (19 blocks of 96 rows here), so a naive "first
10 rows" sample would silently embed 10 vectors that are all the *same*
true class. The beat data below reflects that: `sports`' 10 embedded
vectors span true classes 0–9, not one repeated class.

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
  /work/vanilla_gp_tiled/RTL/sim/tb_reprogram_suite.gtkw
```

`reprogram_manifest.txt` maps every beat to which {model, vector} it
belongs to — this is the ground truth for the console output below:

```
step 0: model='TM_TMIR_..._sports...'  vectors from: this model's own embedded verification.test_vectors
  beat 0: LOAD ack
  beat 1: vector[0] -> expected_class 10
  beat 2: vector[1] -> expected_class 10
  beat 3: vector[2] -> expected_class 2
  beat 4: vector[3] -> expected_class 3
  beat 5: vector[4] -> expected_class 4
  beat 6: vector[5] -> expected_class 5
  beat 7: vector[6] -> expected_class 15
  beat 8: vector[7] -> expected_class 7
  beat 9: vector[8] -> expected_class 10
  beat 10: vector[9] -> expected_class 10

step 1: model='TM_TMIR_..._statlog...'  vectors from: this model's own embedded verification.test_vectors
  beat 11: LOAD ack
  beat 12: vector[0] -> expected_class 1
  beat 13: vector[1] -> expected_class 1
  beat 14: vector[2] -> expected_class 2
  beat 15: vector[3] -> expected_class 3
  beat 16: vector[4] -> expected_class 1
  beat 17: vector[5] -> expected_class 1
  beat 18: vector[6] -> expected_class 2
  beat 19: vector[7] -> expected_class 1
  beat 20: vector[8] -> expected_class 1
  beat 21: vector[9] -> expected_class 1
```

(`expected_class` here is the *model's own prediction* for each vector, not
the vector's ground-truth label — this is a hardware-vs-software
self-consistency check, not an accuracy check. The 10 sports vectors' true
labels are classes 0–9, one each; the model, at only 55.70% accuracy,
mispredicts several of them — e.g. true class 0 and true class 1 both get
predicted as class 10 — which is exactly the kind of thing you want a demo
to surface, not hide.)

**Compile and run under iverilog** (the exact command `reprogram-suite`
prints):

```bash
cd /work/vanilla_gp_tiled/RTL
iverilog -g2001 -Wall -Wno-timescale -o sim/tb_reprogram_suite \
    src/axis_fifo.v src/clause_eval.v src/tile_mem.v src/score_acc_rt.v src/argmax_rt.v src/tm_accel_gp.v \
    tb/tb_reprogram_suite.v
(cd sim && vvp tb_reprogram_suite)
```

```
PASS beat[0]: data=a5000840 last=1     <- sports LOAD ack (status=0x00 OK, info=0x0840=2112 tiles)
PASS beat[1]: data=0000000a last=0     <- 0x0a = 10, matches manifest's expected_class 10
PASS beat[2]: data=0000000a last=0
PASS beat[3]: data=00000002 last=0
PASS beat[4]: data=00000003 last=0
PASS beat[5]: data=00000004 last=0
PASS beat[6]: data=00000005 last=0
PASS beat[7]: data=0000000f last=0     <- 0x0f = 15, matches manifest's expected_class 15
PASS beat[8]: data=00000007 last=0
PASS beat[9]: data=0000000a last=0
PASS beat[10]: data=0000000a last=1
PASS beat[11]: data=a500001b last=1    <- statlog LOAD ack (info=0x001b=27 tiles)
PASS beat[12]: data=00000001 last=0
PASS beat[13]: data=00000001 last=0
PASS beat[14]: data=00000002 last=0
PASS beat[15]: data=00000003 last=0
PASS beat[16]: data=00000001 last=0
PASS beat[17]: data=00000001 last=0
PASS beat[18]: data=00000002 last=0
PASS beat[19]: data=00000001 last=0
PASS beat[20]: data=00000001 last=0
PASS beat[21]: data=00000001 last=1
tb_reprogram_suite: ALL TESTS PASSED (22 beats checked)
```

All 22 beats — one LOAD ack + 10 predictions for `sports`, then one LOAD ack
+ 10 predictions for `statlog` — pass, in one continuous simulation run, on
one synthesized core. This is what "runtime-reprogrammable" actually means
in practice: no resynthesis between the two `LOAD` packets, just a new AXI-
Stream burst.

**Verilator, run against the exact same stimulus/expected files**, to
directly answer "does the reprogram suite work under both simulators":

```bash
make -C sim/verilator run ARGS="--stim ../reprogram_stimulus.memh --exp ../reprogram_expected.memh"
```

```
PASS beat[0]: data=a5000840 last=1
PASS beat[1]: data=0000000a last=0
...
PASS beat[11]: data=a500001b last=1
PASS beat[12]: data=00000001 last=0
...
PASS beat[21]: data=00000001 last=1
tb_top: ALL TESTS PASSED (22 beats checked)
```

**Yes — confirmed with fresh output, byte-for-byte identical to iverilog's**
(same 22 `data=`/`last=` values, beat for beat). `reprogram-suite` builds
one `.memh` stimulus/expected pair up front; iverilog replays it through
`tb/tb_reprogram_suite.v`, Verilator replays the *same files* through
`sim/verilator`'s harness — there's no separate code path or separate
stimulus generation for either simulator to diverge on.

To view it: `bash sim/waves.sh tb_reprogram_suite` (see
[GeneratedOutputs.md](../GeneratedOutputs.md) and the generated
`RTL/README.md` § 2 for how to read the waveform against
`reprogram_manifest.txt`).

---

## Step 5 — Add a third model, without touching the RTL at all

Suppose a third model gets trained *after* the bitstream above was already
built — here, `human_activity` (UCI HAR, 6 classes, another
**unverified**-recipe dataset, picked for variety):

```bash
matador ingest --dataset human_activity --output-dir /work/raw
matador booleanize --dataset human_activity --show-recipe --raw-dir /work/raw --output-dir /work/booleanised
```

```yaml
# UNVERIFIED starting skeleton for 'human_activity' (Human Activity Recognition Using Smartphones (UCI HAR)).
# No catalog-recorded default exists for this dataset -- the encoder below is a
# generic numeric guess (quantile-fit thermometer), not validated against any
# known original encoding. Read the notes below before trusting it as-is:
#   Cleanest match in the whole set: 7352/2947 is the official UCI split,
#   reproduced exactly. ...
raw_npz: /work/raw/human_activity.npz
name: human_activity
output_dir: /work/booleanised
default_encoder:
  encoder: thermometer
  bits: 8
  quantile: true
```

Same reasoning as `sports` in Step 1 — 561 raw features at the default
8-bit thermometer would be 4488 Boolean bits. For this demo, switch to a
1-bit `threshold` encoder instead (561 raw features → 561 Boolean bits):

```yaml
# /work/human_activity_boolconfig.yaml
raw_npz: /work/raw/human_activity.npz
name: human_activity
output_dir: /work/booleanised
default_encoder:
  encoder: threshold
  bits: 1
```

```bash
matador booleanize --config /work/human_activity_boolconfig.yaml
```

```yaml
# /work/human_activity_train.yaml
tm_type: vanilla
clauses: 20
classes: 6
features: 561
s: 5.0
T: 15
epochs: 2
max_included_literals: 32
seed: 42
train_data: /work/booleanised/human_activity_train.txt
test_data: /work/booleanised/human_activity_test.txt
model_name: human_activity
output_dir: /work
```

```bash
matador train --config /work/human_activity_train.yaml
```

```
INFO: Epoch 2/2 — accuracy: 80.32%  train: 0.28s  test: 0.11s
INFO: TMIR written to /work/TMIR/human_activity/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml  (+.npz)
```

561 features / 6 classes / 120 clauses (20 × 6) fits comfortably inside
the bundle's existing 5632/384/20 capacity — no regeneration needed. Now
simulate handing this off to **an RTL engineer who does not have matador
installed** — only the already-built `RTL/` folder:

```bash
matador export-model-json --backend vanilla_gp_tiled \
    --model /work/TMIR/human_activity/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml \
    -o /work/vanilla_gp_tiled/RTL/sim/human_activity_model.json
```

```
Exported .../TM_TMIR_...human_activity....yaml -> /work/vanilla_gp_tiled/RTL/sim/human_activity_model.json

Hand this file (and sim/gen_vectors.py + sim/tm_emulator.py, if the
recipient doesn't already have them from an earlier bundle) to whoever
is building test vectors -- no matador install needed on their end:
  python3 gen_vectors.py combined human_activity_model.json --random -n 20 \
      -o my_stim.memh --expected my_exp.memh --testbench tb_my_model.v
```

To actually prove "no matador needed", this was verified in a directory
containing **only** `sim/gen_vectors.py`, `sim/tm_emulator.py`,
`sim/capacity.json`, `sim/human_activity_model.json`, and `src/` — no
`matador` package importable at all:

```bash
cd sim
python3 gen_vectors.py combined human_activity_model.json --random -n 15 \
    -o my_stim.memh --expected my_exp.memh --testbench tb_human_activity.v --seed 3
iverilog -g2001 -Wall -Wno-timescale -o tb_human_activity \
    ../src/axis_fifo.v ../src/clause_eval.v ../src/tile_mem.v \
    ../src/score_acc_rt.v ../src/argmax_rt.v ../src/tm_accel_gp.v \
    tb_human_activity.v
vvp tb_human_activity
```

```
PASS beat[0]: data=a5000048 last=1
PASS beat[1]: data=00000000 last=0
PASS beat[2]: data=00000000 last=0
...
PASS beat[14]: data=00000000 last=0
PASS beat[15]: data=00000000 last=1
tb_TM_TMIR_..._human_activity...: ALL TESTS PASSED (16 beats checked)
```

16 beats (1 LOAD ack + 15 random vectors), generated and checked entirely
from the standalone bundle, no matador involved — this is the concrete
answer to "does the RTL engineer actually have everything they need."

**One honest caveat worth flagging**, in the same spirit as the accuracy
caveats in [examples/paper_reproduction/](../../examples/paper_reproduction/):
every one of those 15 random vectors predicted class `0`
(`data=00000000` on every beat). The test still **legitimately passes** —
it's checking that the RTL's prediction matches the Python reference
model's prediction for the same random input, a self-consistency check,
not an accuracy check — but an all-one-class response on random inputs is
the same symptom seen with `gas_sensor`'s 1-bit threshold encoding
(19.02% real accuracy, see the paper-reproduction README), and
`human_activity` here uses the same coarse 1-bit encoding for the same
reason (keeping the demo's feature count small). Random inputs aren't
real accelerometer/gyroscope readings, so this isn't proof the trained
model is bad — but if you're using this walkthrough as a template for your
own model, don't mistake `reprogram-suite`/`gen_vectors.py --random`
passing for evidence of real-world accuracy. Test against real held-out
data (`matador validate`) for that.

---

## Summary — what changed at each stage

| Stage | Command | What it produced | What you'd change for your own data |
|---|---|---|---|
| 1 | `matador booleanize --show-recipe` then `--config` | `sports_{train,test}.txt`, 5625 bits, 19 classes | The encoder (`threshold` vs `thermometer`, bit width) — this dataset has no verified default, so you're expected to tune it |
| 2 | `matador booleanize --dataset statlog` | `statlog_{train,test}.txt`, 288 bits, 4 classes | Nothing — verified datasets just work |
| 3 | `matador generate --backend vanilla_gp_tiled` | `RTL/` sized for the *larger* of your models | `max_features`/`max_clauses_total`/`max_classes`/`target_fpga` — set these to your largest anticipated model, not any one model's exact shape |
| 4 | `matador reprogram-suite` | A testbench proving multiple real models reprogram correctly on one core, checked under both iverilog and Verilator | The `steps:` list — one entry per model you want to prove reprogramming across |
| 5 | `matador export-model-json` | A plain JSON file, usable with zero matador install | Nothing about the RTL — this only ever produces a JSON file, the bitstream is untouched |

See also: [Usage.md](../Usage.md) for the command reference, and
[GeneratedOutputs.md](../GeneratedOutputs.md) for what every file in
`RTL/` actually is.
