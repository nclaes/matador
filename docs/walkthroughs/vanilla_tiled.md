# Walkthrough — vanilla_tiled: train, generate, simulate, emulate

Everything below was run for real inside the dev container (`make shell
WORK_DIR=...`) — the commands, output, and numbers are copied verbatim from
an actual run. For the full command reference see [Usage.md](../Usage.md);
this page is the narrated, single-model version.

`vanilla_tiled` sits between the other two backends: like
[vanilla_hardwired](vanilla_hardwired.md), one RTL build is one model (no
runtime reprogramming); unlike it, the trained weights live in an on-chip
ROM read by a sequential FSM, not wired-in combinational logic — trading
some latency for a smaller, more area-tunable design via the tile-size
knobs below.

---

## Step 1 — Have a trained model

Same `digits` model as the [vanilla_hardwired
walkthrough](vanilla_hardwired.md), so the two are directly comparable:

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

```
INFO: Epoch 2/2 — accuracy: 78.83%  train: 0.11s  test: 0.04s
INFO: TMIR written to /work/TMIR/digits/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml  (+.npz)
```

---

## Step 2 — Generate

```yaml
# /work/vanilla_tiled.yaml
model_path: /work/TMIR/digits/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml
output_dir: /work
axis_data_width: 32
fifo_depth: 16
feat_slice: 8
clause_slice: 8
```

**What to change:** `feat_slice`/`clause_slice` control how many features
and clauses are evaluated per clock cycle — the FSM sweeps
`ceil(n_features/feat_slice) × ceil(n_clauses_total/clause_slice)` tiles per
inference, one per cycle. For this model that's `ceil(512/8) × ceil(200/8)
= 64 × 25 = 1600` compute cycles. Smaller slices mean less combinational
logic per cycle (smaller area) but more cycles per inference (higher
latency); larger slices are the opposite tradeoff. `8`/`8` here are small,
conservative defaults for a demo — a real design would size these against
actual LUT budget and latency requirements.

```bash
matador generate --backend vanilla_tiled --config /work/vanilla_tiled.yaml
```

```
  vanilla TM  |  512 features  |  10 classes  |  200 clauses
  Generating [vanilla_tiled]…

RTL written to: /work/vanilla_tiled/RTL
```

```
RTL/
├── src/  axis_fifo.v  clause_eval.v  score_acc.v  argmax.v  tm_accelerator.v
├── tb/   tb_axis_fifo.v  tb_clause_eval.v  tb_score_acc.v  tb_argmax.v  tb_system.v
├── sim/  run_iverilog.sh  lint_verilator.sh  run_xsim.sh  waves.sh  verilator/
└── README.md
```

`tm_accelerator.v`'s tile ROM (an `initial` block of packed hex constants)
is where the trained model's weights live — unlike `hcb_blocks.v` in the
hardwired backend, this is genuinely a ROM lookup table, not structural
logic, which is what makes `feat_slice`/`clause_slice` a meaningful area
knob independent of the model itself.

---

## Step 3 — Simulate

```bash
matador simulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml
```

```
Running RTL simulation…

  tb_argmax                  PASS   .../tb_argmax.v:109: $finish called at 20000 (1ps)
  tb_axis_fifo                PASS   .../tb_axis_fifo.v:1717: $finish called at 3405000 (1ps)
  tb_clause_eval              PASS   .../tb_clause_eval.v:113: $finish called at 70000 (1ps)
  tb_score_acc                PASS   .../tb_score_acc.v:284: $finish called at 2355000 (1ps)
  tb_system                   PASS   .../tb_system.v:462: $finish called at 181785000 (1ps)
  tb_system [verilator]       PASS   tb_system: ALL 10 TESTS PASSED

6 passed, 0 failed
```

Four per-module unit testbenches (FIFO, clause evaluator, score
accumulator, argmax) plus the full-system test under both iverilog and
Verilator — more granular than `vanilla_hardwired`'s two testbenches,
reflecting the extra FSM/tiling machinery this backend has that the other
doesn't.

---

## Step 4 — Emulate (no simulator needed)

```bash
matador emulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml --verify
```

```
Loading model: /work/TMIR/digits/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml
Running emulator on 10 test vector(s)…

  [  0] PASS  predicted=9  expected=9  scores=[-4, -1, -1, 0, -2, 1, -4, 0, 0, 2]
  ...
  [  9] PASS  predicted=6  expected=6  scores=[-1, -1, -4, -6, 0, -4, 1, -3, -2, -3]

10 passed, 0 failed

Cross-checking emulator against reference inference…
  Emulator matches reference on all 10 vector(s).

Emulation passed.
```

Identical predicted classes and scores to the `vanilla_hardwired`
walkthrough — same model, same math, different RTL architecture. Try
regenerating with a much smaller `feat_slice`/`clause_slice` (e.g. `4`/`4`)
and re-running Steps 3-4: the predictions won't change (the tiling knobs
don't affect *what* the model computes, only how many cycles it takes),
which is a good way to build confidence that they're purely an area/latency
tradeoff.

---

## Summary

| Stage | Command | What to change for your own model |
|---|---|---|
| Generate | `matador generate --backend vanilla_tiled` | `feat_slice`/`clause_slice` (area vs. latency) — everything about the model itself is baked in automatically from `model_path` |
| Simulate | `matador simulate --backend vanilla_tiled` | Nothing — runs all five testbenches against the embedded test vectors unconditionally |
| Emulate | `matador emulate --backend vanilla_tiled --verify` | Nothing — same |

Like `vanilla_hardwired`, a different model means regenerating — there's no
runtime reprogramming path here (that's specifically what
[vanilla_gp_tiled](vanilla_gp_tiled.md) trades area/complexity for).

See also: [Usage.md](../Usage.md) for the command reference, and
[GeneratedOutputs.md](../GeneratedOutputs.md) for the full generated-file map.
