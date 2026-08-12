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

Every stage — booleanisation, training, RTL generation, and test-vector
generation — has real, user-facing configuration knobs, not just the
handful of fields shown in the main example. This walkthrough calls those
out explicitly as they come up (**"Configuration knobs"** boxes), explains
*why* each one exists and what trade-off it controls, and proves the less
obvious ones (how many models, how many test vectors) actually work with
real commands and real output, not just description.

If you just want the command reference, see [Usage.md](../Usage.md). This
page is the "why does it say that, and what else could I set here" narrative
version, aimed at someone doing this for the first time.

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

### Configuration knobs — booleanisation

Every field below belongs to `FeatureEncoderSpec` / `BooleanisationConfig`
(`matador/config/schema.py`) — either as `default_encoder:` (applied to
every raw column not otherwise listed) or per-column inside a `features:`
list (`column: <int or "lo-hi">` + the same encoder fields, for when
different columns of one dataset need different treatment).

| Field | Applies to | What it controls |
|---|---|---|
| `encoder` | all | `thermometer`, `threshold`, `onehot`, or `passthrough` — see below |
| `bits` | `thermometer` | How many Boolean bits represent one raw column |
| `quantile` | `thermometer` | Fit thresholds from the *training split's own* value distribution, instead of a fixed range |
| `range` | `thermometer` | Explicit `(min, max)` instead of `quantile` — use when you know the true bounds (e.g. a sensor's datasheet range) |
| `bins` | `thermometer` | Explicit threshold list, for full manual control over where each bit fires |
| `threshold` | `threshold` | The single cutoff value (raw value ≥ threshold → 1) |
| `categories` | `onehot` | Explicit category list (otherwise inferred from the training split) |
| `test_size` / `seed` / `stratify` | whole config | Only relevant when `raw_npz` isn't pre-split into train/test — controls the random split matador performs itself |

**The four encoders:**
- **`thermometer`** — `bits` monotonically-increasing 1s below a value, 0s above (or vice versa depending on convention); the standard choice for ordinal/continuous data, since it preserves ordering (a higher raw value differs from a lower one by strictly more set bits, which is friendly to a Tsetlin Machine's conjunctive-clause structure).
- **`threshold`** — a single bit, 1 raw column → 1 Boolean bit. The coarsest possible encoding; used throughout this walkthrough's unverified-recipe datasets specifically to keep the demo's feature/literal count small.
- **`onehot`** — one bit per category, exactly one set; for genuinely categorical (non-ordinal) columns, where a thermometer's implied ordering would be meaningless.
- **`passthrough`** — the raw column is already 0/1; no encoding applied.

**Design rationale — why bit width matters more here than in most ML
pipelines:** a Tsetlin Machine's clauses operate on individual Boolean
*literals* (`n_literals = 2 × n_features` — one positive and one negated
literal per Boolean feature, the standard TM convention). Every bit you add
to the encoding is a real literal a real clause has to evaluate, and — once
you reach Step 3 — a real slice of `tile_mem` BRAM that has to physically
exist on the FPGA (`tile_mem` scales linearly with `features_padded`). So
the encoder isn't just a data-representation choice, it's a direct hardware
sizing choice: more bits generally means more discriminative power *and* a
proportionally bigger, slower-to-reprogram core. This is exactly why this
walkthrough downgrades `sports` from the default 8-bit thermometer skeleton
(45,000 bits) to a 1-bit threshold (5625 bits) — not because 8-bit is wrong,
but because this demo doesn't need that much resolution to prove the
pipeline works, and a smaller core reprograms and simulates faster.
`quantile: true` is the default recommendation for unknown-range data
(genomic/sensor data rarely comes with documented bounds) since it adapts
to whatever the training split actually contains, rather than silently
clipping outliers against a wrong guessed range.

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

### Configuration knobs — training

Every field below is `TrainingConfig` (`matador/config/schema.py`):

| Field | What it controls | Design rationale |
|---|---|---|
| `tm_type` | `vanilla` or `coalesced` TM variant | Coalesced shares clause structure across classes (fewer total parameters); vanilla gives each class its own independent clause bank. This walkthrough uses `vanilla` throughout, matching `vanilla_gp_tiled`'s architecture. |
| `clauses` | Clauses **per class**, must be even | `n_clauses_total = classes × clauses` — this is what feeds `max_clauses_total` in Step 3. Must be even because a TM splits each class's clause bank exactly in half: one half votes *for* the class (positive polarity), one half votes *against* it (negative polarity) — an odd count can't split evenly. |
| `classes` | Number of output classes | Must match the booleanized data's label range exactly. |
| `features` | Number of Boolean input bits | Must match the booleanized data's column count exactly (validated against `train_data`/`test_data` at load time). |
| `s` | Specificity | Controls how aggressively clauses specialize on the training data — higher `s` produces more specific (more literals included, tighter-fitting) clauses; lower `s` produces coarser, more general ones. This is the main clause-complexity/generalization knob. |
| `T` | Voting threshold | Clamps the summed clause vote before it drives the learning-feedback probability — a standard Tsetlin Machine mechanism, not matador-specific. Larger `T` requires a larger margin of consistent votes before feedback saturates. |
| `epochs` | Training passes over the data | More epochs generally improve accuracy up to a point, at linear cost in training time. |
| `max_included_literals` | Cap on literals per clause | Validated against `features × 2` at config-load time — a config with `max_included_literals` above that limit is rejected outright (a real bug this walkthrough's own `examples/paper_reproduction/mammographic_training_config.yaml` hit and had to be fixed for). Caps clause complexity (regularization, similar in spirit to `s`) **and** directly bounds the RTL's per-clause literal-evaluation width, so it's a hardware area knob too, not just a training one. |
| `seed` | RNG seed | Threaded into both the TM's own weight initialization/tie-breaking *and* (since the fix described in Step 4) which test rows get embedded as this model's self-verification vectors — so the same seed reproduces the whole pipeline's output byte-for-byte, as demonstrated above. |
| `model_name` | Output namespace | `matador train` writes to `<output_dir>/TMIR/<model_name>/`, defaulting to `train_data`'s filename stem (e.g. `sports_train.txt` → `sports`) if not given. This is exactly why `sports` and `statlog` (Step 2) can be trained into the same `/work` directory without colliding. |

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
double-check. This is the same `bits`/`quantile` knobs from Step 1's
configuration table, just tuned differently: `statlog`'s 18 raw features are
genuine continuous measurements with a verified, working 16-bit resolution,
versus `sports`'s deliberately-coarsened 1-bit demo encoding.)

A verified recipe is not a locked one — it's simply matador's own
best-known `BooleanisationConfig` for that dataset, stored in
`data/Raw_Data_Bank.yaml`. You can always override it with your own
`--config` file (the same `default_encoder`/`features` shape shown for
`sports` above) even for a "yes" dataset; `matador booleanize --dataset X`
without `--config` is the convenience path, not the only path.

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

### Configuration knobs — RTL generation

Every field below is `GPTiledAcceleratorConfig`
(`matador/backends/gp_tiled/config.py`):

| Field | Default | What it controls |
|---|---|---|
| `target_fpga` | — (required) | Device key into a small built-in BRAM-bit table (`xc7z020`, `xcku040` today — see `fpga_budget.FPGA_BRAM_BITS`). Any string is accepted; unknown devices require `bram_bits_budget` explicitly. |
| `bram_bits_budget` | `None` | Explicit override of the target device's total on-chip Block RAM, in bits — use this for a device not in the built-in table. |
| `feat_slice` | 32 | Features evaluated **per clock cycle**. |
| `clause_slice` | 32 | Clauses evaluated **per clock cycle**. |
| `max_features` | — (required) | Compile-time ceiling on `n_features` for *any* model this bitstream will ever load. |
| `max_clauses_total` | — (required) | Compile-time ceiling on `n_clauses_total` (`classes × clauses`) for any loaded model. |
| `max_classes` | 32 | Compile-time ceiling on `n_classes`. Negligible BRAM cost (doesn't factor into the device-fit check below), but still hard-capped at 255 by the protocol (see below). |
| `axis_data_width` | 32 (fixed) | AXI-Stream `TDATA` width. Only 32 is currently supported — the vendored packet encoder hardcodes this. |
| `fifo_depth` | 16 | Input FIFO depth in beats; must be a power of 2. |

**Design rationale — why capacity is checked against real device BRAM, not
arbitrary limits:** `tile_mem` — the only capacity-scaling resource that
matters here — is sized as `2 × features_padded × clauses_padded` bits
(the same "2×" positive/negated-literal convention from Step 1's booleanisation
rationale, now showing up again as a hardware storage cost). `matador
generate` checks that figure against **75% of the target device's total
Block RAM** (`fpga_budget.TILE_MEM_BRAM_FRACTION`), not 100% — the
remaining quarter is deliberately reserved for the input FIFO and any other
on-chip logic sharing the same BRAM pool, so a config that "just barely
fits" on paper doesn't actually fail at synthesis time once real place-and-route
logic is added. This is exactly the check that caught `xc7z020` being too
small above, before any RTL was even written.

**Design rationale — `feat_slice`/`clause_slice` is a latency/area
trade-off, independent of the BRAM story:** these control how many
features/clauses the core evaluates per clock cycle, i.e. how many rounds
of computation one inference takes. Smaller slices mean more clock cycles
per inference (higher latency) but less parallel comparison logic (smaller
LUT/DSP footprint); larger slices are the reverse. Unlike `tile_mem`, this
doesn't change *storage*, so it's tunable somewhat independently of the
BRAM-fit decision above — a design targeting minimum latency and a design
targeting minimum logic area would pick different `feat_slice`/`clause_slice`
values for the identical `max_features`/`max_clauses_total`.

**Hard protocol limits worth knowing about**, all enforced at config-validation
time (`matador/backends/gp_tiled/tmir_bridge.py`): the `CMD_LOAD` packet
header encodes `n_feat_slices` and `n_clause_slices` as 8-bit fields (max
255 each — so `max_features ≤ 255 × feat_slice`, `max_clauses_total ≤ 255 ×
clause_slice`), `n_classes` as an 8-bit field (max 255), and
`n_clauses_total` itself as a 16-bit field (max 65535). None of these are
reachable by accident at the scale of this walkthrough's models, but they're
the reason `max_classes` has a hard ceiling regardless of how much BRAM
headroom you have.

---

## Step 4 — Prove it actually reprograms, with real models (iverilog *and* Verilator)

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

Each output beat's `data` word is either a LOAD ack (see the box after
[Step 3](#configuration-knobs--rtl-generation) — top byte `0xA5`, status
byte, then a 16-bit tile count) or a prediction (a small class index) — the
two are unambiguous because ack words are always ≥ `0xA5000000`
(2.77 billion), far larger than any real class index.

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

### Configuration knobs — reprogram-suite (how many models, how many vectors)

Every field below is `ReprogramStepConfig`/`ReprogramSuiteConfig`
(`matador/backends/gp_tiled/reprogram_config.py`):

| Field | Scope | What it controls |
|---|---|---|
| `steps` | whole config | An ordered list of `{model, ...}` entries — **one entry per model**, unlimited (only validated as non-empty). Each model just has to individually fit the bitstream's compile-time capacity from Step 3. |
| `model` | per step | Path to that step's TMIR `.yaml`/`.yml`/`.npz`. |
| `vectors` | per step | A raw bit-vector file, one test vector per line — full manual control over exactly which inputs get exercised. |
| `dataset` | per step | A booleanized `*_test.txt`-shaped file (label column dropped) — point at real held-out data instead of hand-written vectors. |
| `n_samples` | per step, with `dataset` | Subsample this many rows (seeded, reproducible). Omit it to use **every** row in the file. |
| `seed` | per step | Subsampling seed for `n_samples`. |
| `name` | per step | Display name in `reprogram_manifest.txt`; defaults to the model file's stem. |

If neither `vectors` nor `dataset` is given for a step (as above), it falls
back to that model's own embedded `verification.test_vectors` — fixed at
exactly 10 vectors, chosen at *training* time (round-robin across classes,
as described above), not adjustable per reprogram-suite run. For anything
beyond that lightweight built-in self-check — more vectors, specific inputs
you care about, or real held-out accuracy-adjacent coverage — use `dataset:`
/`n_samples:` or `vectors:` instead.

**Proof this scales past two models and past the fixed 10-vector default,**
run for real against the exact same synthesized bitstream from Step 3 (no
regeneration needed — this is the same "add models without touching the
RTL" property demonstrated in Step 5, just via `reprogram-suite` instead of
`export-model-json`):

```yaml
# /work/reprogram_config_3step.yaml
steps:
  - model: /work/TMIR/sports/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_1_max_literals_32.yaml
    # no dataset/vectors -> embedded default, 10 vectors
  - model: /work/TMIR/statlog/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml
    dataset: /work/booleanised/statlog_test.txt
    n_samples: 5
    seed: 7
  - model: /work/TMIR/human_activity/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml
    # no dataset/vectors -> embedded default, 10 vectors
```

```bash
matador reprogram-suite --backend vanilla_gp_tiled \
    --config /work/vanilla_gp_tiled.yaml \
    --reprogram-config /work/reprogram_config_3step.yaml
```

```
Building reprogram suite for vanilla_gp_tiled at /work/vanilla_gp_tiled/RTL (3 step(s))
...
step 0: model='TM_TMIR_..._sports...'      vectors from: this model's own embedded verification.test_vectors
  beat 0: LOAD ack ... beat 10: vector[9] -> expected_class 10

step 1: model='TM_TMIR_..._statlog...'     vectors from dataset: /work/booleanised/statlog_test.txt
  beat 11: LOAD ack
  beat 12: vector[0] -> expected_class 2
  beat 13: vector[1] -> expected_class 1
  beat 14: vector[2] -> expected_class 3
  beat 15: vector[3] -> expected_class 3
  beat 16: vector[4] -> expected_class 1     <- only 5 vectors, exactly n_samples: 5

step 2: model='TM_TMIR_..._human_activity...' vectors from: this model's own embedded verification.test_vectors
  beat 17: LOAD ack ... beat 27: vector[9] -> expected_class 3
```

```bash
iverilog -g2001 -Wall -Wno-timescale -o sim/tb_reprogram_suite \
    src/axis_fifo.v src/clause_eval.v src/tile_mem.v src/score_acc_rt.v src/argmax_rt.v src/tm_accel_gp.v \
    tb/tb_reprogram_suite.v
(cd sim && vvp tb_reprogram_suite)
```

```
PASS beat[0]:  data=a5000840 last=1     <- sports LOAD ack
PASS beat[10]: data=0000000a last=1     <- sports' last (10th) prediction
PASS beat[11]: data=a500001b last=1     <- statlog LOAD ack
PASS beat[12]: data=00000002 last=0
PASS beat[13]: data=00000001 last=0
PASS beat[14]: data=00000003 last=0
PASS beat[15]: data=00000003 last=0
PASS beat[16]: data=00000001 last=1     <- statlog's LAST beat is vector[4], only 5 vectors this time
PASS beat[17]: data=a5000048 last=1     <- human_activity LOAD ack (info=0x0048=72 tiles)
PASS beat[27]: data=00000003 last=1     <- human_activity's 10th and final prediction
tb_reprogram_suite: ALL TESTS PASSED (28 beats checked)
```

28 beats total: `(1 + 10) + (1 + 5) + (1 + 10)` — three models, one of them
with an explicitly-controlled 5-vector sample instead of the default 10, all
reprogrammed and checked in one continuous run on the one bitstream built
back in Step 3.

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

### Configuration knobs — standalone test-vector generation (no matador at all)

`gen_vectors.py` (`matador/backends/gp_tiled/vendor/gen_vectors.py`) is
copied verbatim into every generated bundle's `RTL/sim/`, stdlib-only, with
zero matador imports — everything below works on a machine with nothing but
Python 3 and the exported `.json` model file(s).

**One model at a time — `combined` subcommand:**

| Flag | What it controls |
|---|---|
| `model` (positional) | The exported model JSON (`matador export-model-json`'s output). |
| `vectors` (positional, optional) | A text file of real feature vectors — omit and use `--random` instead. |
| `--random` | Generate `-n` random feature vectors instead of reading a file. |
| `-n` | How many random vectors (default 10) — this is the standalone equivalent of a reprogram step's vector count. |
| `--seed` | Random seed for `--random` (default 0) — reproducible, same idea as `ReprogramStepConfig.seed`. |
| `-o` | Output `.memh` stimulus path. |
| `--expected` | Also write expected ack+prediction beats, for a self-checking testbench. |
| `--testbench` | Also emit a ready-to-compile Verilog testbench (requires `--expected`). |

**Multiple models, fully standalone — `sequence` subcommand:** the direct,
matador-free equivalent of `reprogram-suite`'s `steps:` list from Step 4 —
same idea (unlimited models, per-step vector control), expressed as a plain
JSON manifest instead of a matador config file:

```json
[
  {"model": "statlog_model.json", "random": true, "n": 4, "seed": 11},
  {"model": "human_activity_model.json", "random": true, "n": 3, "seed": 22}
]
```

```bash
python3 gen_vectors.py sequence manifest.json \
    -o seq_stim.memh --expected seq_exp.memh --testbench tb_seq.v
```

Run for real (same standalone, no-matador directory as above, plus a second
exported model):

```
  step 0: model='TM_TMIR_..._statlog...' (4 classes, 20 clauses/class) -- 4 vector(s), predictions=[1, 2, 1, 1]
  step 1: model='TM_TMIR_..._human_activity...' (6 classes, 20 clauses/class) -- 3 vector(s), predictions=[0, 0, 0]
wrote 6436 words across 2 reprogram step(s) -> seq_stim.memh
wrote 9 expected beats -> seq_exp.memh
```

```bash
iverilog -g2001 -Wall -Wno-timescale -o tb_manifest \
    ../src/axis_fifo.v ../src/clause_eval.v ../src/tile_mem.v \
    ../src/score_acc_rt.v ../src/argmax_rt.v ../src/tm_accel_gp.v \
    tb_seq.v
vvp tb_manifest
```

```
PASS beat[0]: data=a500001b last=1     <- statlog LOAD ack
PASS beat[1]: data=00000001 last=0
PASS beat[2]: data=00000002 last=0
PASS beat[3]: data=00000001 last=0
PASS beat[4]: data=00000001 last=1     <- only 4 vectors, exactly n: 4
PASS beat[5]: data=a5000048 last=1     <- human_activity LOAD ack
PASS beat[6]: data=00000000 last=0
PASS beat[7]: data=00000000 last=0
PASS beat[8]: data=00000000 last=1     <- only 3 vectors, exactly n: 3
tb_manifest: ALL TESTS PASSED (9 beats checked)
```

9 beats = `(1+4) + (1+3)` — two models, independently-sized random vector
batches, checked with `gen_vectors.py` alone. Every "how many models / how
many vectors" knob described for `reprogram-suite` in Step 4 has a
standalone equivalent here — the two tools share the same underlying
packet encoder (`tm_emulator.py`'s `encode_load_packet`/`encode_infer_packet`),
just invoked from matador's config-driven CLI on one side and a plain JSON
manifest on the other.

---

## Summary — what changed at each stage

| Stage | Command | What it produced | What you'd change for your own data |
|---|---|---|---|
| 1 | `matador booleanize --show-recipe` then `--config` | `sports_{train,test}.txt`, 5625 bits, 19 classes | The encoder (`threshold` vs `thermometer`, bit width, `quantile`/`range`/`bins`) — this dataset has no verified default, so you're expected to tune it |
| 1 | `matador train` | `sports`'s TMIR | `clauses`/`s`/`T`/`epochs`/`max_included_literals`/`seed` — the model-quality and RTL-area knobs |
| 2 | `matador booleanize --dataset statlog` | `statlog_{train,test}.txt`, 288 bits, 4 classes | Nothing required — verified datasets just work, but `--config` can still override the default recipe |
| 3 | `matador generate --backend vanilla_gp_tiled` | `RTL/` sized for the *larger* of your models | `max_features`/`max_clauses_total`/`max_classes`/`target_fpga`/`feat_slice`/`clause_slice` — set the capacity fields to your largest anticipated model, tune the slice widths for your latency/area target |
| 4 | `matador reprogram-suite` | A testbench proving multiple real models reprogram correctly on one core, checked under both iverilog and Verilator | The `steps:` list (any number of models) and, per step, `vectors:`/`dataset:`/`n_samples:`/`seed:` (any number/choice of test vectors, or the 10-vector embedded default) |
| 5 | `matador export-model-json` + `gen_vectors.py` | A plain JSON file, usable with zero matador install | Nothing about the RTL — this only ever produces a JSON file, the bitstream is untouched. `gen_vectors.py combined -n`/`--seed` or `sequence`'s manifest give the same models/vectors control fully standalone |

See also: [Usage.md](../Usage.md) for the command reference, and
[GeneratedOutputs.md](../GeneratedOutputs.md) for what every file in
`RTL/` actually is.
