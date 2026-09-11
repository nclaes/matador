# Walkthrough — redress: train, generate, simulate, emulate, and confirm 32-wide batching

Everything below was run for real inside the dev container (`make shell
WORK_DIR=...`) — the commands, output, and numbers are copied verbatim from
an actual run. For the full command reference see [Usage.md](../Usage.md);
this page is the narrated, single-model version.

`redress` is a hardware translation of the REDRESS 16-bit Include-Encoding
compression scheme (source thesis Chapter 4.3, Algorithms 11/12) — MUSTANG's
Single-Core configuration (Chapter 5.3). Like
[vanilla_gp_tiled](vanilla_gp_tiled.md), it's runtime-reprogrammable: one
synthesis, sized by a capacity ceiling, can be reprogrammed at runtime with
any model that fits — no resynthesis. Unlike gp_tiled's dense tile memory,
the loaded representation here is **sparse**: one 16-bit instruction word per
*included* literal, nothing stored for excludes or empty clauses. Algorithm
12's other defining feature carries straight through to hardware: **W
datapoints are classified in parallel per pass through the instruction
stream** — `batch_width` (16/32/64) parallel AND-accumulate lanes, one
Feature FIFO word per feature holding all W datapoints' bit for that feature
simultaneously. This walkthrough's last step demonstrates that concretely,
end to end, on a real held-out test set.

---

## Step 1 — Have a trained model

```bash
matador ingest --dataset mammographic --output-dir /work/raw
matador booleanize --dataset mammographic --raw-dir /work/raw --output-dir /work/booleanised
```

```
train samples: 768  test samples: 193
5 raw features -> 40 Boolean bits
```

```yaml
# /work/mammographic_train.yaml
tm_type: vanilla
clauses: 20
classes: 2
features: 40
s: 5.0
T: 15
epochs: 10
max_included_literals: 32
seed: 42
train_data: /work/booleanised/mammographic_train.txt
test_data: /work/booleanised/mammographic_test.txt
model_name: mammographic
output_dir: /work
```

```bash
matador train --config /work/mammographic_train.yaml
```

```
INFO: Epoch 10/10 — accuracy: 80.31%  train: 0.04s  test: 0.00s

Exporting TMIR…
Done. TMIR written to:
  /work/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.yaml
  /work/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.npz
```

---

## Step 2 — Generate

```yaml
# /work/redress.yaml
model_path: /work/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.yaml
output_dir: /work
axis_data_width: 32
fifo_depth: 16
batch_width: 32
target_fpga: xc7z020
max_includes: 1024
max_features: 64
max_classes: 4
```

**What to change:** `max_includes` is the one genuinely data-dependent knob
— unlike gp_tiled's dense `max_clauses_total` (derivable from geometry
alone), REDRESS's compressed size depends on how *sparse* the trained model
actually is. `matador generate` tells you exactly what you need if you guess
too small:

```bash
matador generate --backend redress --config /work/redress.yaml
```

```
Error: RTL generation failed: model encodes to 1,094 MUSTANG instruction
words, exceeding this config's max_includes=1,024. Raise max_includes (and
re-check the BRAM budget), or use a sparser/smaller model.
```

Raise it and re-run:

```bash
sed -i 's/max_includes: 1024/max_includes: 2048/' /work/redress.yaml
matador generate --backend redress --config /work/redress.yaml
```

```
  vanilla TM  |  40 features  |  2 classes  |  40 clauses
  Generating [redress]…

RTL written to: /work/redress/RTL
```

```
RTL/
├── src/   redress_accel.v  redress_bram.v  score_acc_rt.v  argmax_rt.v
├── tb/    tb_redress_model.v
├── sim/   run_iverilog.sh  waves.sh  verilator/  *.memh  test_vectors.txt
└── README.md
```

`redress_accel.v` is the top module — its own header comment documents the
full AXI-Stream protocol (Header → Instruction-stream → Feature-stream →
predictions) and the 16-bit MUSTANG instruction word layout. `batch_width`
above is *also* baked into this file: it sets `BATCH_WIDTH`, which in turn
sizes the Feature FIFO's word width and how many parallel `score_acc_rt` +
`argmax_rt` lane instances get generated (see Step 5).

---

## Step 3 — Simulate

```bash
matador simulate --backend redress --config /work/redress.yaml
```

```
Running RTL simulation…

  tb_redress_model           PASS   /work/redress/RTL/tb/tb_redress_model.v:121: $finish called at 55505000 (1ps)
  tb_system [verilator]      PASS   tb_top: ALL TESTS PASSED (32 beats checked)

2 passed, 0 failed
```

Two independent simulators, same correctness contract: `tb_redress_model`
under iverilog and the Verilator harness both load this model via one
Instruction-stream packet, then replay its embedded verification vectors —
matador caps embedded `verification.test_vectors` at 10 regardless of the
real test-set size, which is why this first pass only checks 32 beats (one
padded batch of `batch_width`). Step 5 below replays the *full* 193-vector
test set instead.

---

## Step 4 — Emulate (no simulator needed)

```bash
matador emulate --backend redress --config /work/redress.yaml --verify
```

```
Running emulator on 10 test vector(s)…

  [  0] PASS  predicted=0  expected=0  scores=[0, 0]
  [  1] PASS  predicted=0  expected=0  scores=[8, -7]
  ...
  [  9] PASS  predicted=0  expected=0  scores=[0, 0]

10 passed, 0 failed

Cross-checking emulator against reference inference…
  Emulator matches reference on all 10 vector(s).

Emulation passed.
```

`--verify` cross-checks the Python cycle-accurate model (`RedressEmulator`,
built directly on `codec.mustang_decode_and_infer()`) against matador's
independent mathematical reference — a second, non-RTL confirmation that the
predictions are actually correct, not just self-consistent.

---

## Step 5 — Confirm 32-wide parallel batching, on the full test set

Everything above only exercised one padded batch (the embedded-vector cap).
To actually prove Algorithm 12's "W datapoints processed in parallel per
pass through the encoded stream" on real hardware, `matador reprogram-suite`
can pull test vectors from a real dataset file instead of the embedded cap —
here, the *entire* 193-row booleanized test set from Step 1, no subsampling:

```yaml
# /work/reprogram_config.yaml
steps:
  - model: /work/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.yaml
    dataset: /work/booleanised/mammographic_test.txt
    name: mammographic_full_test_set
```

```bash
matador reprogram-suite --backend redress \
    --config /work/redress.yaml \
    --reprogram-config /work/reprogram_config.yaml
```

```
Building reprogram suite for redress at /work/redress/RTL (1 step(s))

Wrote:
  /work/redress/RTL/tb/tb_reprogram_suite.v
  /work/redress/RTL/sim/reprogram_stimulus.memh
  /work/redress/RTL/sim/reprogram_expected.memh
  /work/redress/RTL/sim/reprogram_manifest.txt
  /work/redress/RTL/sim/tb_reprogram_suite.gtkw
```

`reprogram_manifest.txt` reports exactly what got built — **193 real
vectors, chunked into 7 batches of `batch_width`=32**, all against one
instruction load (the "load once, infer many" pattern this backend is built
around):

```
step 0: model='mammographic_full_test_set'  tmir=/work/TMIR/mammographic/...
  vectors from dataset: /work/booleanised/mammographic_test.txt
  n_instr=1094  n_vectors=193  n_batches=7  first stimulus beat=0
```

Run it — under **both** simulators:

```bash
bash /work/redress/RTL/sim/run_iverilog.sh
```

```
tb_reprogram_suite: ALL 224 TESTS PASSED
/work/redress/RTL/sim/../tb/tb_reprogram_suite.v:119: $finish called at 322565000 (1ps)
```

```bash
make -C /work/redress/RTL/sim/verilator run \
    ARGS="--stim ../reprogram_stimulus.memh --exp ../reprogram_expected.memh"
```

```
PASS beat[223]: data=00000000 last=1
tb_top: ALL TESTS PASSED (224 beats checked)
```

224 = 7 batches × `batch_width` 32 (the last batch is padded — 193 = 6×32 +
1, so batch 7 carries 1 real datapoint plus 31 zero-padded lanes; the
generator still checks every lane's prediction, since it's fully
deterministic). Every one of the 224 output beats — both `tdata` (predicted
class) *and* `tlast` (must fire on exactly the last of each 32) — matches
`codec.mustang_decode_and_infer(..., W=32)` bit-for-bit, on **both**
independent simulators.

**Where the 32-way parallelism actually lives in the RTL** (`src/redress_accel.v`):

- The Feature FIFO is `batch_width` bits wide — one word per feature, bit
  `j` of that word is datapoint `j`'s value for that feature. Loading one
  feature loads all 32 datapoints' bit for it in a single beat.
- `clause_mask` (`BATCH_WIDTH` bits) AND-accumulates all 32 lanes *in the
  same cycle* — one shared instruction stream drives the same literal
  selection into all 32 lanes simultaneously; there's no per-lane
  instruction re-fetch.
- `generate for (gi = 0; gi < BATCH_WIDTH; gi = gi + 1)` instantiates 32
  parallel `score_acc_rt` (vote accumulator) and `argmax_rt` (winner select)
  pairs, one per lane — each lane's own lane bit of `clause_mask` gates its
  own accumulator, all fed by the same broadcast `valid`/`cls`/`polarity`
  control signals.

That's the hardware expression of Algorithm 12's batching: one decode of the
compressed stream, W independent classifications computed in lockstep.

---

## Summary

| Stage | Command | What to change for your own model |
|---|---|---|
| Generate | `matador generate --backend redress` | `max_includes` (data-dependent — raise it if generate() tells you to), `max_features`/`max_classes` (geometry ceiling), `batch_width` (16/32/64 — parallel lane count) |
| Simulate | `matador simulate --backend redress` | Nothing — runs both simulators against the embedded (capped) test vectors unconditionally |
| Emulate | `matador emulate --backend redress --verify` | Nothing — same |
| Reprogram-suite | `matador reprogram-suite --backend redress` | `reprogram_config.yaml`'s `steps:` — point `dataset:`/`vectors:` at a full test set, or list several models to reprogram across in one run |

See also: [Usage.md](../Usage.md) for the command reference,
[GeneratedOutputs.md](../GeneratedOutputs.md) for the full generated-file
map, and [vanilla_gp_tiled.md](vanilla_gp_tiled.md) for the sibling
runtime-reprogrammable backend (dense tile memory instead of a sparse
instruction stream).
