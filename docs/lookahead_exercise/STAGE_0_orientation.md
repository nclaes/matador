# Stage 0 — Orientation, baseline, and a warm-up bug

## 1. What changed and why

- **Fixed a real bug in `redress`'s argmax hookup.** When `n_classes == max_classes`
  and that number is a power of two, the accelerator only ever predicted class 0.
  This also affects anyone already using the `redress` backend, including the shared
  `REDRESS_TM_TinyML` bundle, which has the same line.
- **Added a parametrized RTL test**, `test_rtl_simulation_n_classes_vs_max_classes`, over
  (n_classes, max_classes) = (2,16), (3,4), (2,2), (4,4), (16,16). It was written
  first and failed on exactly the three power-of-two cases.
- **Tidied `tests/backends/test_redress_rtl.py` until ruff passes**: import order plus three
  long-line wraps. These are formatting changes only, with no change in behaviour.
- **Added `Thesis.pdf` to `.gitignore`.** It's a local reference copy.
- **Checked in `docs/lookahead_exercise/EXERCISE.pdf`** and this README.
- **Baseline recorded** (Section 3). No other `redress` behaviour changed.

**Where the code disagrees with EXERCISE.pdf (Rule 6: the code wins)**

| EXERCISE.pdf says | What this repo actually does |
|---|---|
| Work on branch `feature/redress-lookahead` | The user chose to work directly on `production_ready`. Tags `la-stage-N` still mark each stage. Nothing is pushed. |
| Baseline: 53 passed, 1 skipped | **54 passed, 0 skipped.** Verilator 5.048 is installed here, so the Verilator test runs instead of skipping. |
| The config docs say RTL goes to `<output_dir>/RTL/` | `matador generate` writes to **`<output_dir>/redress/RTL/`** (see Section 3). |
| Line numbers (`≈130`, `≈261`, ...) | Close. After the fix, the argmax hookup is at `redress_accel.v:270`. |

**Things that were already broken and are left alone** (out of scope, noted here for the record):
- The full `pytest -q` run has **7 failures before and after this stage**: 2 in
  `tests/test_tmu_reference_consistency.py` and 5 in `tests/test_verification.py::TestEmulateCLI`.
  None of them involve `redress`. I checked by stashing this stage's changes and rerunning.
- Strict Verilator lint (`-Wall` with WIDTH warnings on) shows 2 `WIDTHTRUNC` warnings in
  `redress_accel.v`: the `feat_raddr` assignment from the 12-bit offset, and `pred_class[out_idx + 1'b1]`.
  Both are identical on `HEAD`. The new lines are lint-clean.

## 2. Files touched

- `matador/backends/redress/src/redress_accel.v`: widened the argmax `n_classes` hookup (the bug fix)
- `tests/backends/test_redress_rtl.py`: new parametrized test, plus ruff-only formatting
- `.gitignore`: ignore `Thesis.pdf`
- `docs/lookahead_exercise/EXERCISE.pdf`: the exercise brief
- `docs/lookahead_exercise/STAGE_0_orientation.md`: this file

## 3. Run it yourself

Run everything from the repo root with the project venv active
(`pyenv activate virtual_env_3_10`, or put its `bin/` first on `PATH`).

**Tools seen in this environment:** Python 3.10.0, Icarus Verilog 13.0, Verilator 5.048,
GTKWave 3.4.0, Docker 28.1.1. `ruff` came from `pip install -e ".[dev]"`.

**a) Baseline before the fix** (`git stash` the two source files, or check out `la-stage-0~1`):
```
$ pytest -q tests/backends/test_redress_codec.py tests/backends/test_redress_rtl.py
......................................................                   [100%]
54 passed in 14.80s
```

**b) The new test, before the fix.** It fails on exactly the power-of-two cases:
```
$ pytest -q tests/backends/test_redress_rtl.py -k n_classes
E         FAIL out[1]: expected data=3 last=0  got data=0 last=0
E         FAIL out[11]: expected data=2 last=0  got data=0 last=0
...
FAILED tests/backends/test_redress_rtl.py::test_rtl_simulation_n_classes_vs_max_classes[2-2]
FAILED tests/backends/test_redress_rtl.py::test_rtl_simulation_n_classes_vs_max_classes[4-4]
FAILED tests/backends/test_redress_rtl.py::test_rtl_simulation_n_classes_vs_max_classes[16-16]
3 failed, 2 passed, 26 deselected in 0.62s
```
Every wrong answer is `got data=0`: the hardware is stuck on class 0.

**c) After the fix:**
```
$ pytest -q tests/backends/test_redress_rtl.py -k n_classes
5 passed, 26 deselected in 0.62s
$ pytest -q tests/backends/test_redress_codec.py tests/backends/test_redress_rtl.py
59 passed in 9.60s
$ ruff check tests/backends/test_redress_rtl.py
All checks passed!
```

**d) Hands-on: build a 4-of-4-class bundle and look at the waves.** `build/` is gitignored.
```bash
mkdir -p build/la_stage0
python - <<'EOF'
import sys; sys.path.insert(0, "tests/backends")
from test_redress_rtl import _make_tmir
t = _make_tmir(n_features=15, n_classes=4, cpc=8, density=0.08, n_samples=32, seed=14)
t.to_yaml("build/la_stage0/model.yaml")
EOF
cat > build/la_stage0/redress.yaml <<'EOF'
model_path: build/la_stage0/model.yaml
output_dir: build/la_stage0
target_fpga: xc7z020
max_includes: 4096
max_features: 256
max_classes: 4
batch_width: 32
EOF
matador generate --backend redress --config build/la_stage0/redress.yaml
bash build/la_stage0/redress/RTL/sim/run_iverilog.sh
bash build/la_stage0/redress/RTL/sim/waves.sh tb_redress_model     # opens GTKWave
```
Real output:
```
RTL written to: build/la_stage0/redress/RTL
...
tb_redress_model: ALL 32 TESTS PASSED
```
The first lines of `sim/model_stimulus.memh` (bit 32 = TLAST, bits 31:0 = data):
```
0C100004E   header: NEW_STREAM=1, INSTR=1, n_classes=4, n_instructions-1=0x4E (79 words)
000006000   word 0: P=0 CC=1 E=1 offset=0  L=0
000006015   word 1: P=0 CC=1 E=1 offset=10 L=1
000006016   word 2: P=0 CC=1 E=1 offset=11 L=0
```
To see the bug in waves, rerun (d) on the pre-fix RTL and look at `pred_class` (always 0)
next to `argmax_n_classes` / the `n_classes` input on `lanes[0].amax`, which is 0.

## 4. Look here

- `matador/backends/redress/src/redress_accel.v:130-138`: the nine FSM states.
  Inference uses four per instruction: `INSTR → DECODE → FEAT → ACCUM`.
- `redress_accel.v:200-213`: decode wires. Note that `class_boundary` and `clause_boundary` are
  *edge* tests (`!=` against a latched `E_prev`/`CC_prev`), not level tests.
- `redress_accel.v:243-270`: the fix. `argmax_rt`'s port is `[CLASS_WIDTH:0]` (one bit
  wider than a class index), and the old code filled it with `{1'b0, hdr_n_classes[CLASS_WIDTH-1:0]}`.
- `matador/backends/redress/src/redress_bram.v:42-47`: `rdata <= mem[raddr]` is a registered read.
  That's why every read address is issued one state before the data is used.
- `matador/backends/redress/codec.py:539-560`: `E_bit` starts at 0 and toggles *before* each class.
  `CC` resets to 0 per class and toggles on each non-empty clause's first literal.
- `matador/backends/redress/rtl.py:542`: expected outputs come from `codec.mustang_decode_and_infer()`.
  `rtl.py:683` is where the testbench prints `FAIL out[...]`.

## 5. Concept notes

**The decode path, end to end**

```
TMIR includes ─► codec.mustang_encode()          codec.py:505
                 one 16-bit word per included literal [P|CC|E|offset(12)|L]
             ─► RedressBackend._build_stimulus() rtl.py:490
                 header + instr words + feature batches ─► sim/model_stimulus.memh
                 expected preds (mustang_decode_and_infer) ─► sim/model_expected.memh
             ─► tb_redress_model.v drives s_axis from the .memh
             ─► redress_accel FSM                 redress_accel.v:281-447
                 S_LOAD_INSTR: words ─► instr_mem (redress_bram)
                 S_LOAD_FEAT : packed batch words ─► feature_fifo; TLAST starts inference
                 per word: INSTR ─► DECODE ─► FEAT ─► ACCUM
             ─► score_acc_rt (one per lane)       score_acc_rt.v:53-67
                 on a flush: scores[class] += ±1 if that lane's clause_mask bit is 1
             ─► argmax_rt (one per lane)          argmax_rt.v:48
                 highest score among k < n_classes; ties go to the lowest index
             ─► S_OUTPUT: batch_width beats of pred_class on m_axis
```

**Cycle table for one instruction.** This is a real trace from `build/la_stage0`'s VCD,
covering the first two words of the stream. Values are what each register holds *during* that cycle.

| cyc | state | ptr | instr_re / raddr | instr_rdata | instr_latched | feat_re / raddr | feat_rdata | clause_mask | E_prev / CC_prev |
|---|---|---|---|---|---|---|---|---|---|
| 0 | INSTR (3)  | 0 | **1 / 0** | x | x | 0 / x | x | ffffffff | x / x |
| 1 | DECODE (4) | 0 | 0 / 0 | **0x6000** | x | 0 / x | x | ffffffff | x / x |
| 2 | FEAT (5)   | 0 | 0 / 0 | 0x6000 | **0x6000** | **1 / 0** | x | ffffffff | x / x |
| 3 | ACCUM (6)  | 0 | 0 / 0 | 0x6000 | 0x6000 | 0 / 0 | **6ecd5da6** | ffffffff | x / x |
| 4 | INSTR (3)  | 1 | **1 / 1** | 0x6000 | 0x6000 | 0 / 0 | 6ecd5da6 | **6ecd5da6** | **1 / 1** |
| 5 | DECODE (4) | 1 | 0 / 1 | **0x6015** | 0x6000 | 0 / 0 | 6ecd5da6 | 6ecd5da6 | 1 / 1 |
| 6 | FEAT (5)   | 1 | 0 / 1 | 0x6015 | **0x6015** | **1 / 0xa** | 6ecd5da6 | 6ecd5da6 | 1 / 1 |
| 7 | ACCUM (6)  | 1 | 0 / 1 | 0x6015 | 0x6015 | 0 / 0xa | **58196f65** | 6ecd5da6 | 1 / 1 |
| 8 | INSTR (3)  | 2 | 1 / 2 | 0x6015 | 0x6015 | 0 / 0xa | 58196f65 | **26c41082** | 1 / 1 |

How to read it:
- **Read latency.** The read is issued in cycle 0, and the data appears in cycle 1 (`DECODE`).
  The feature side does the same: issued in 2, data in 3 (`ACCUM`).
- **Word 0, cycle 3.** `have_clause` is still 0, so there's no flush. `clause_mask` becomes
  `ffffffff & literal` = `6ecd5da6`, which is why the reset value is all ones.
- **Word 1, cycle 7.** Here `L=1`, so the selected literal is `~58196f65 = a7e6909a`.
  ANDed with `6ecd5da6`, that gives `26c41082`, the value you see in cycle 8.
- **Latches.** `E_prev`/`CC_prev` are latched from word 0 in its `ACCUM` cycle. There is no reset value.

**The bug in one table.** `CLASS_WIDTH = clog2(MAX_CLASSES)`.

| n_classes | MAX_CLASSES | CLASS_WIDTH | old: `{0, n[CW-1:0]}` | new: `n[CW:0]` |
|---|---|---|---|---|
| 2  | 16 | 4 | 2  | 2  |
| 3  | 4  | 2 | 3  | 3  |
| 2  | 2  | 1 | **0** | 2 |
| 4  | 4  | 2 | **0** | 4 |
| 16 | 16 | 4 | **0** | 16 |

The fix zero-extends `hdr_n_classes` before slicing `[CLASS_WIDTH:0]`. That way the slice
stays in range even when `CLASS_WIDTH+1 > 8` (`MAX_CLASSES = 256`, checked with Verilator lint).

## 6. Check yourself

1. In which state is `instr_rdata` valid for word i?
2. On a flush, why is `clause_mask` set to the selected literal and not to all ones?
3. Which E value does class 0 carry, and which line decides it?
4. Why does the bug only appear when n_classes == max_classes and that number is a power of two?
5. Which register could you flip to mimic an E flip on one word, for one batch only?

## 7. If something goes wrong

- **`pytest` can't import `matador`.** You're in the wrong Python. Use `virtual_env_3_10`,
  or run `pip install -e ".[dev]"` in the one you're using.
- **The iverilog tests are skipped.** `iverilog` isn't on `PATH`. Check with `which iverilog`.
- **`run_iverilog.sh: No such file`.** The bundle is under `<output_dir>/redress/RTL/`, not `<output_dir>/RTL/`.
- **A test fails with `got data=0` on every wrong line.** That's the argmax class-count signature. Check
  `lanes[0].amax.n_classes` in GTKWave against the header's `n_classes`.
- **A test fails with "fixture must expect some non-zero classes".** The random model predicts class 0
  for everything, because no clause fires and all scores tie. Lower `density` or change `seed`.
- **Many `WIDTHTRUNC` warnings from Verilator.** The generated Makefile passes `--Wno-WIDTH`. The two
  remaining strict-lint warnings already existed (Section 1).

## 8. Answers

1. In `S_INFER_DECODE`. The read is issued at the end of `S_INFER_ACCUM` (or at `S_LOAD_FEAT`'s TLAST
   for word 0), so `raddr` is stable through `S_INFER_INSTR` and `rdata` updates on the edge into `DECODE`.
   In the trace, `0x6000` first appears in cycle 1, state 4. `DECODE` copies it into `instr_latched`.
2. On that same edge, `score_acc_rt` consumes the *old* `clause_mask` for the clause being flushed.
   The new clause's first literal is already in hand, so it starts the new mask directly.
   Setting all ones would need an extra AND cycle. All ones is only the reset state, used for word 0.
3. E = 1. `codec.py:542` (`E_bit ^= 1`) toggles `E_bit` from its initial 0 *before* encoding class 0.
   In the RTL, word 0's E is simply latched into `E_prev` (`redress_accel.v:394`) with no check, because
   `have_class` is 0.
4. `argmax_rt`'s `n_classes` is `CLASS_WIDTH+1` bits, but only the low `CLASS_WIDTH` bits of the header
   value were passed in. A value only loses information there when it needs bit `CLASS_WIDTH`, that is
   n_classes ≥ 2^CLASS_WIDTH. Since n_classes ≤ MAX_CLASSES ≤ 2^CLASS_WIDTH, that only happens when
   n_classes = MAX_CLASSES = 2^CLASS_WIDTH. Then the low bits are all 0 and argmax only considers class 0.
5. `instr_latched[13]`, between `S_INFER_DECODE` (where it's written) and `S_INFER_ACCUM` (where
   `class_boundary` reads `E_cur`). Instruction memory is untouched, so the next batch re-reads the
   correct word. That's what makes it a one-batch, one-word effect.
