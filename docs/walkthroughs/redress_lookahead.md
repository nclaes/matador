# Walkthrough — redress_lookahead on a real model: mammographic, end to end

Everything below was run for real on macOS (Python 3.10, Icarus Verilog 13.0, Verilator 5.048),
outside the dev container, with paths under `build/la_stage6/` (gitignored). It follows
[redress.md](redress.md) for training, then adds the look-ahead backend and fault-injection
measurements. Outputs are verbatim, trimmed only where marked `...`.

## Step 1 — Train the same model as the redress walkthrough

```
$ matador ingest --dataset mammographic --output-dir build/la_stage6/raw
$ matador booleanize --dataset mammographic --raw-dir build/la_stage6/raw --output-dir build/la_stage6/booleanised
Booleanizing build/la_stage6/raw/mammographic.npz -> mammographic_{train,test}.txt
  train samples: 768  test samples: 193
  5 raw features -> 40 Boolean bits
```

The training config is `redress.md`'s, with paths changed:

```yaml
# build/la_stage6/mammographic_train.yaml
tm_type: vanilla
clauses: 20
classes: 2
features: 40
s: 5.0
T: 15
epochs: 10
max_included_literals: 32
seed: 42
train_data: build/la_stage6/booleanised/mammographic_train.txt
test_data: build/la_stage6/booleanised/mammographic_test.txt
model_name: mammographic
output_dir: build/la_stage6
```

> **Environment gotcha (outside the container).** `matador train` fails with
> `TypeError: TMBaseModel.__init__() missing 1 required positional argument: 'K'`. The `matador`
> console script imports the site-packages `tmu` (0.8.3), not the repo's vendored `tmu/`.
> Running the same CLI through Python *from the repo root* picks up the vendored copy:

```
$ python -c "import sys; from matador.cli import main; sys.argv=['matador','train','--config','build/la_stage6/mammographic_train.yaml']; main()"
INFO: Epoch 10/10 — accuracy: 80.31%  train: 0.05s  test: 0.00s
INFO: TMIR written to build/la_stage6/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.yaml  (+.npz)
```

That's the same 80.31% as `redress.md`. Training is seeded, and two runs gave the same model.

**The shape of this model matters for everything below.** It has 2 classes, 20 clauses per class
(40 non-empty clauses), and **1,094 stored words**. That's two classes of 526 and 568 words, about
27 literals per clause, and only **one** one-literal clause.

## Step 2 — Generate, simulate and emulate both backends

Configs (`build/la_stage6/redress.yaml` and `build/la_stage6/redress_lookahead.yaml`), with the same
capacity in both:

```yaml
model_path: build/la_stage6/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.yaml
output_dir: build/la_stage6
target_fpga: xc7z020
max_includes: 2048
max_features: 64
max_classes: 4
batch_width: 32
# redress_lookahead.yaml only:
e_lookahead_depth: 1
cc_lookahead: true
```

```
=== redress
  Generating [redress]…
RTL written to: build/la_stage6/redress/RTL
  tb_redress_model           PASS   .../build/la_stage6/redress/RTL/tb/tb_redress_model.v:123: $finish called at 55505000 (1ps)
  tb_system [verilator]      PASS   tb_top: ALL TESTS PASSED (32 beats checked)
2 passed, 0 failed
10 passed, 0 failed
  Emulator matches reference on all 10 vector(s).
Emulation passed.
=== redress_lookahead
  Generating [redress_lookahead]…
RTL written to: build/la_stage6/redress_lookahead/RTL
  tb_redress_model           PASS   .../build/la_stage6/redress_lookahead/RTL/tb/tb_redress_model.v:123: $finish called at 55555000 (1ps)
  tb_system [verilator]      PASS   tb_top: ALL TESTS PASSED (32 beats checked)
2 passed, 0 failed
10 passed, 0 failed
  Emulator matches reference on all 10 vector(s).
Emulation passed.
```

The commands were `matador generate|simulate|emulate --verify --backend <b> --config build/la_stage6/<b>.yaml`.
The look-ahead bundle finishes **50 ns (5 cycles) later**. That is exactly its one extra stored word
(4 decode cycles + 1 load beat), not a per-word cost.

## Step 3 — Exhaustive single-bit campaign, full test set

The campaign flips each of the 16 × 1,094 bits in turn and runs all 193 held-out vectors through
`decode_faithful`. P is measured against the fault-free prediction, not the label.

```
$ python -m matador.backends.redress_lookahead.campaign \
    --model build/la_stage6/TMIR/mammographic/TM_TMIR_Clauses_20_s_value_5_T_value_15_epochs_10_max_literals_32.yaml \
    --test-data build/la_stage6/booleanised/mammographic_test.txt --variant v0 v3 v4 v3d2 v4d2
model: 2 classes, 40 features, 1094 as-built words; 193 test vectors; batch_width=32, max_classes=16
variant   words   extra     N x P    gain          P       CC        E   offset        L
----------------------------------------------------------------------------------------
v0         1094    0.0%    288.03      1x       0.15   144.47   140.10     2.22     1.09
v3         1094    0.0%    148.15      2x       0.15   144.47     0.23     2.22     1.09
v4         1095    0.1%      3.37     85x       0.15     0.43     0.23     1.79     0.78
v3d2       1094    0.0%    148.31      2x       0.15   144.47     0.38     2.22     1.09
v4d2       1095    0.1%      3.53     82x       0.15     0.43     0.38     1.79     0.78
```
This takes about 10 minutes for all five variants. Use `--vectors N` to subsample.

**Reading it:**
- **As built, CC matters as much as E here (144 against 140).** These clauses are long (~27 literals).
  A CC flip in the middle splits one rarely-firing clause into a prefix, a one-literal clause and a
  suffix. Those short clauses fire much more often, so the class sums change a lot.
- **v3 (E look-ahead) removes E sensitivity almost completely (140 → 0.23, 607×) for free**, but the
  total only halves, because CC is untouched.
- **v4 also removes CC sensitivity (144 → 0.43, 336×).** The total falls 85×, from 288 to 3.37,
  at a cost of **one** word (+0.1%), since this model has a single one-literal clause.
- **Two-word look-ahead (`d2`) is slightly *worse* for single flips** (0.38 against 0.23 on E). Its
  extra condition can also reject a real boundary near a flip, and its benefit is adjacent-flip
  tolerance, which this single-flip campaign doesn't measure.

**Against the strawman's Table 13 (mammographic):**

| | v0 N × P | v3 | v4 | v4 extra words |
|---|---|---|---|---|
| Strawman Table 13 | 51.8 | 0.96 (54×) | 0.30 (173×) | 10.4% |
| This model | 288.03 | 148.15 (1.9×) | 3.37 (85×) | 0.1% |

The models differ, and so does the *distribution* of sensitivity. The strawman's numbers imply E dominated
its model (v3 alone gave 54×) and that it had many one-literal clauses (10.4% overhead). This model
has long clauses, so CC is half of the as-built sensitivity and v3 alone can't help much. The
qualitative result holds: **v4 cuts instruction-memory N × P by about two orders of magnitude for a
negligible word cost.** The size of v3's share is model-dependent.

## Step 4 — Corrupted-model replay through both RTLs

A random sample of single-bit-corrupted Instruction Memories, each loaded into the real RTL (iverilog)
and checked against `decode_faithful` in the matching mode. There are 64 held-out vectors per replay
(2 batches).

Uniform over all 16 bits:
```
$ python docs/lookahead_exercise/scripts/replay_sample.py --model <model>.yaml \
    --test-data build/la_stage6/booleanised/mammographic_test.txt --n 50 --vectors 64 --out build/la_stage6/replays
64 vectors per replay, 50 random replay-safe single-bit flips per RTL (fields: P CC E offset L)
design                   replays  RTL==model  visible   visible flips by field
redress (as built)            50      50/50         3   CC 1, E 2
redress_lookahead (v4)        50      50/50         0   -
```

Targeted at E and CC, where faults are visible:
```
$ python docs/lookahead_exercise/scripts/replay_sample.py ... --fields E CC --seed 2
64 vectors per replay, 50 random replay-safe single-bit flips per RTL (fields: E CC)
design                   replays  RTL==model  visible   visible flips by field
redress (as built)            50      50/50        37   CC 23, E 14
redress_lookahead (v4)        50      50/50         1   E 1
```

- **All 200 replays matched `decode_faithful` on their RTL.** So the campaign numbers in Step 3, computed
  in software in minutes, describe the hardware.
- **Uniform random flips are mostly invisible** (3 of 50 as built). Most stored bits are offsets and
  literals deep inside long clauses that rarely fire. A replay sample that only draws uniformly would
  check the RTL on almost no *interesting* faults. The targeted sample is the one that tests the look-ahead:
  37 of 50 E/CC flips change predictions as built, and 1 of 50 with v4 (an E flip; under v4 only
  flips on a class's first, second or last words can still move a class boundary).

## Summary

| Step | Command | Result |
|---|---|---|
| Train | `matador train` (vendored `tmu`) | 80.31%, 1,094 words, 1 one-literal clause |
| Generate / simulate / emulate | `matador … --backend redress_lookahead` | iverilog + Verilator PASS; emulator == reference; +5 cycles total (1 word) |
| Campaign | `python -m matador.backends.redress_lookahead.campaign` | N × P 288 → 3.37 (85×) with v4, for +0.1% words |
| Replay | `docs/lookahead_exercise/scripts/replay_sample.py` | 200 / 200 RTL == `decode_faithful` |
