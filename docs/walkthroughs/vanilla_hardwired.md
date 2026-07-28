# Walkthrough — vanilla_hardwired: train, generate, simulate, emulate

Everything below was run for real inside the dev container (`make shell
WORK_DIR=...`) — the commands, output, and numbers are copied verbatim from
an actual run. For the full command reference see [Usage.md](../Usage.md);
this page is the narrated, single-model version.

`vanilla_hardwired` is the opposite design point from
[vanilla_gp_tiled](vanilla_gp_tiled.md): the trained model's weights are
wired directly into combinational logic (HCB blocks + an adder tree) at
generation time. **One RTL build is one model** — there's no runtime
reprogramming, no capacity ceiling to plan for, and no `steps:`/reprogram
config to write. If you need a different model, you regenerate.

---

## Step 1 — Have a trained model

This walkthrough reuses a `digits` model trained the same way as in the
[vanilla_gp_tiled walkthrough](vanilla_gp_tiled.md) — see that page's Step 5
if you don't already have one:

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
# /work/vanilla_hardwired.yaml
model_path: /work/TMIR/digits/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_2_max_literals_32.yaml
output_dir: /work
axis_data_width: 32
fifo_depth: 16
pipeline_stages: 2
```

**What to change:** unlike `vanilla_gp_tiled`'s `max_features`/
`max_clauses_total`/`target_fpga` (a capacity ceiling for *any* model
you'll ever load), `vanilla_hardwired`'s only real design knob is
`pipeline_stages` — how many register stages the adder tree that sums
clause votes is split into. `0` is fully combinational (shortest latency,
deepest combinational path); higher values add pipeline registers (longer
latency, shorter critical path, likely a higher achievable clock frequency
after synthesis). `2` here is an arbitrary middle value for the demo —
tune it against your actual timing closure results.

```bash
matador generate --backend vanilla_hardwired --config /work/vanilla_hardwired.yaml
```

```
  vanilla TM  |  512 features  |  10 classes  |  200 clauses
  Generating [vanilla_hardwired]…

RTL written to: /work/vanilla_hardwired/RTL
```

```
RTL/
├── src/  axis_fifo.v  hcb_blocks.v  hw_score.v  hw_tm_accelerator.v
├── tb/   tb_hcb_blocks.v  tb_hw_system.v
├── sim/  run_iverilog.sh  lint_verilator.sh  run_xsim.sh  waves.sh  verilator/
└── README.md
```

`hcb_blocks.v` is where the trained model's weights actually live —
`digits`'s 200 clauses (20/class × 10 classes) are wired-in Hardware Clause
Blocks, not a ROM lookup like `vanilla_tiled`. Open it if you want to see
the trained model's Include bits as literal RTL structure, not data.

---

## Step 3 — Simulate

```bash
matador simulate --backend vanilla_hardwired --config /work/vanilla_hardwired.yaml
```

```
Running RTL simulation…

  tb_hcb_blocks              PASS   /work/vanilla_hardwired/RTL/tb/tb_hcb_blocks.v:165: $finish called at 686000 (1ps)
  tb_hw_system               PASS   /work/vanilla_hardwired/RTL/tb/tb_hw_system.v:211: $finish called at 1786000 (1ps)
  tb_system [verilator]      PASS   ALL 10 TESTS PASSED

3 passed, 0 failed
```

Two testbenches under iverilog (`tb_hcb_blocks` — unit test for the wired-in
clause logic; `tb_hw_system` — full end-to-end AXI-Stream test against the
model's embedded test vectors) plus the same system test independently
re-run under Verilator. All three agree.

---

## Step 4 — Emulate (no simulator needed)

```bash
matador emulate --backend vanilla_hardwired --config /work/vanilla_hardwired.yaml --verify
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

`--verify` cross-checks the Python cycle-accurate model against matador's
independent mathematical reference — a second, non-RTL confirmation that
the predictions above are actually correct, not just self-consistent.

Notice the predicted classes and scores are **identical** to the
`vanilla_tiled` walkthrough's Step 4 output for the same model — same TMIR,
same math, two completely different RTL architectures (wired combinational
logic here vs. a sequential FSM + tile ROM there). That agreement is the
point: `AcceleratorSpec.simulate(x) == RTL_simulation(x)` regardless of
which backend built the hardware.

---

## Summary

| Stage | Command | What to change for your own model |
|---|---|---|
| Generate | `matador generate --backend vanilla_hardwired` | `pipeline_stages` (timing/latency tradeoff) — everything else about the model is baked in automatically from `model_path` |
| Simulate | `matador simulate --backend vanilla_hardwired` | Nothing — runs both simulators against the embedded test vectors unconditionally |
| Emulate | `matador emulate --backend vanilla_hardwired --verify` | Nothing — same |

There is no equivalent of `vanilla_gp_tiled`'s Step 3/5
(`export-model-json`/new-model-without-resynthesizing): a *different*
model here means running `matador generate` again, which is the whole
point of this backend's simplicity — no capacity planning, no runtime
protocol, just "one model, one build."

See also: [Usage.md](../Usage.md) for the command reference, and
[GeneratedOutputs.md](../GeneratedOutputs.md) for the full generated-file map.
