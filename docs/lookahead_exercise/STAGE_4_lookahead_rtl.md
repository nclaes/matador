# Stage 4 — The look-ahead RTL

## 1. What changed and why

- **New `matador/backends/redress_lookahead/src/redress_la_accel.v`.** It's a copy of `redress_accel.v`
  (Stage 0 fix included) with two new parameters, `E_LA_DEPTH` (0/1) and `CC_LA` (0/1). The functional
  change is **17 code lines**.
- **What the RTL does:**
  - **Idle-slot read:** `S_INFER_DECODE` also reads word `ptr+1`.
  - **Look-ahead signals:** `have_next`, `E_next`, `CC_next`, `e_confirm`, `cc_confirm`.
  - **Gated decisions:** `class_boundary` and `clause_boundary` require confirmation.
  - **Batch start:** with depth 1, `E_prev ← E_INIT = 1` and `have_class ← 1`.
  - **End of stream:** with no next word, a change is rejected.
- **With `E_LA_DEPTH = CC_LA = 0`** no extra read is issued, and the module decodes like `redress_accel`.
- **Decision on the look-ahead word: reuse the BRAM output register (`instr_rdata`), no `instr_next`
  register.** The read of word i+1 lands on `instr_rdata` exactly in `S_INFER_ACCUM`, the cycle where the
  decision is made. An explicit register could only capture it at the *end* of ACCUM, one cycle too late,
  so it would cost a cycle or a restructured fetch. In waves and for fault injection, the named targets
  are the wires `E_next`/`CC_next`, read from `instr_rdata`.
- **New `config.py` and `rtl.py`** (`RedressLookaheadConfig`, `RedressLookaheadBackend`). The Stage 4 tests
  need a backend to generate bundles, so these came forward from Stage 5. Registration, the emulator,
  the CLI and docs remain Stage 5.
- **The generated `.gtkw`** gains a "Look-ahead decision" group: `E_cur`, `E_prev`, `E_next`, `have_next`,
  `class_boundary`, and the CC equivalents.
- **New `docs/lookahead_exercise/scripts/vcd_table.py`** prints a VCD as a per-cycle table: the text
  version of GTKWave used in Section 5.
- **13 RTL tests**, all under iverilog:
  - 4 clean parameter combinations;
  - 40 corrupted replays (v4 and v3);
  - interior E flips that change nothing;
  - edge-of-stream flips;
  - an equal-cycles check;
  - the gtkw group;
  - the config rejecting depth 2.

**Mutation testing: every rule has a test that fails without it.** I broke the RTL five ways and reran
the suite. The first run let M3 and M4 through: random replays almost never flip the first or last word.
I then added `test_stream_edge_flips`.

| Mutation | Tests failing |
|---|---|
| M1: no E look-ahead (`e_confirm = 1`) | 5 |
| M2: the latch takes a rejected E | 4 |
| M3: no end-of-stream gate (`have_next = 1`) | 1 (caught by the edge test, missed before it existed) |
| M4: no `E_INIT` reset (`have_class` starts at 0) | 1 (caught by the edge test, missed before it existed) |
| M5: no CC look-ahead (`cc_confirm = 1`) | 2 |
| *(restored)* | 0 (13 passed) |

**Where the code disagrees with EXERCISE.pdf**

| EXERCISE.pdf | This repo |
|---|---|
| Stage 5 creates `config.py` / `rtl.py` | Created here, because the Stage 4 tests need them. Stage 5 completes them. |
| "Decide with the user": reuse the BRAM output register or add `instr_next` | You asked me to complete all stages unattended, so I decided: reuse it (reason above). It's easy to revisit. |
| Hint: issue `instr_re/instr_raddr = ptr + 1` in DECODE | Done, plus a `LA_READ` guard so the 0/0 configuration issues no extra reads at all. |

## 2. Files touched

- `matador/backends/redress_lookahead/src/redress_la_accel.v`: new RTL
- `matador/backends/redress_lookahead/config.py`: `RedressLookaheadConfig` (depth 0/1, `cc_lookahead`)
- `matador/backends/redress_lookahead/rtl.py`: `RedressLookaheadBackend`, built on the Stage 1 hooks, plus the gtkw group
- `tests/backends/test_redress_lookahead_rtl.py`: 13 tests
- `docs/lookahead_exercise/scripts/vcd_table.py`: VCD to cycle table
- `docs/lookahead_exercise/STAGE_4_lookahead_rtl.md`: this file

## 3. Run it yourself

```
$ pytest -q tests/backends/test_redress_lookahead_rtl.py
13 passed in 6.96s
$ diff matador/backends/redress/src/redress_accel.v matador/backends/redress_lookahead/src/redress_la_accel.v
```

**Strict lint**, for all four parameter combinations:
```
verilator --lint-only -Wall -Wno-TIMESCALEMOD -Wno-UNUSEDSIGNAL -Wno-BLKSEQ --top-module redress_la_accel \
  -GE_LA_DEPTH=1 -GCC_LA=1 -GMAX_CLASSES=16 -GCLASS_WIDTH=4 \
  matador/backends/redress/src/redress_bram.v matador/backends/redress/src/score_acc_rt.v \
  matador/backends/common/src/argmax_rt.v matador/backends/redress_lookahead/src/redress_la_accel.v
```
All four combinations give only the two `WIDTHTRUNC` warnings inherited from `redress_accel.v`
(lines 393 and 479 here), recorded in Stage 0.

**Waves for one interior E flip.** Word 45 is in the middle of class 1 (`0x4010 → 0x6010`). The same
flipped stream is loaded into both RTLs (the bundles are in `build/la_stage4/`):
```
bash build/la_stage4/lookahead/RTL/sim/run_iverilog.sh     # tb_redress_model: ALL 32 TESTS PASSED
bash build/la_stage4/lookahead/RTL/sim/waves.sh            # GTKWave: open the "Look-ahead decision" group
python docs/lookahead_exercise/scripts/vcd_table.py build/la_stage4/lookahead/RTL/sim/tb_redress_model.vcd \
  --signals state,ptr,instr_re,instr_latched,instr_rdata,have_next,E_prev,E_next,class_boundary,current_class \
  --ptr 44 46 --first-batch
```
To regenerate `build/la_stage4`, run the snippet in Section 7. Both RTLs finish at the same time:
```
as_built   $finish called at 6355000 (1ps)
lookahead  $finish called at 6355000 (1ps)
```

## 4. Look here

- `redress_la_accel.v:236-243`: the whole decision. `have_next` gates the stale read at the end of the
  stream; `e_confirm`/`cc_confirm` are 1 when that look-ahead is disabled.
- `redress_la_accel.v:396`: the idle-slot read. Compare it with `S_INFER_ACCUM`'s own `instr_raddr <= ptr + 1`.
  Both read the same address, at different times, for different reasons.
- `redress_la_accel.v:370`: `have_class <= (E_LA_DEPTH != 0)`, with `E_prev <= E_INIT` on the next line.
- The cycle table below, against the waveform at word 45: `class_boundary` in **state 6** is the only
  value that matters.

## 5. Concept notes

**When are word i and word i+1 valid?** This was drawn before writing the RTL, and the trace confirms it.
```
state         INSTR(3)      DECODE(4)          FEAT(5)            ACCUM(6)
instr_re      1 (from ACCUM) 0                 1 (look-ahead)     0
instr_rdata   stale         word i             word i             word i+1   <- decision made here
instr_latched word i-1      word i-1 -> i      word i             word i
feat_rdata    -             -                  -                  feature[offset(i)]
```

**The real trace** (`vcd_table.py`, look-ahead RTL, flipped word 45 = `ptr 0x2d`):
```
cyc | state | ptr    | instr_re | instr_latched | instr_rdata | have_next | E_prev | E_next | class_boundary | current_class
316 | 6     | 0x002c | 0        | 0x000d        | 0x6010      | 1         | 0      | 1      | 0              | 1
317 | 3     | 0x002d | 1        | 0x000d        | 0x6010      | 1         | 0      | 1      | 0              | 1
318 | 4     | 0x002d | 0        | 0x000d        | 0x6010      | 1         | 0      | 1      | 0              | 1
319 | 5     | 0x002d | 1        | 0x6010        | 0x6010      | 1         | 0      | 1      | 1              | 1
320 | 6     | 0x002d | 0        | 0x6010        | 0x4018      | 1         | 0      | 0      | 0              | 1
324 | 6     | 0x002e | 0        | 0x4018        | 0x401b      | 1         | 0      | 0      | 0              | 1
```
- **Cycle 320, word 45 in ACCUM.** `E_cur = 1 ≠ E_prev = 0`, but `E_next` (word 46, `0x4018`) is 0,
  so `class_boundary = 0`. The change is rejected, `E_prev` stays 0 and `current_class` stays 1.
- **Cycle 319 shows `class_boundary = 1` in FEAT. That's harmless and worth understanding.** The wire is
  combinational and always computed, but in FEAT `instr_rdata` still holds word i itself, so "next"
  trivially confirms. Only `flush_pulse = (state == S_INFER_ACCUM) && ...` acts on it.
  In GTKWave, read the decision only where `state = 6`.

**The as-built RTL on the same flipped stream** (ACCUM rows only):
```
cyc | state | ptr    | instr_latched | E_prev | class_boundary | current_class
316 | 6     | 0x002c | 0x000d        | 0      | 0              | 1
320 | 6     | 0x002d | 0x6010        | 0      | 1              | 1     <- spurious boundary into word 45
324 | 6     | 0x002e | 0x4018        | 1      | 1              | 2     <- and out of it: class 3 from here on
```
That is two boundaries from one bit, so every later class sits two indices too high: the cascade.

## 6. Check yourself

1. In which state does `instr_rdata` hold word i+1?
2. Why gate the look-ahead on `ptr != hdr_n_instr`? What would the read return otherwise?
3. What goes wrong if `E_prev` takes the value of a rejected E?
4. Why can't a two-word look-ahead reuse the same idle slot? What would you build instead?
5. Why did the random replays miss mutations M3 and M4, and what does that say about random fault sampling?

## 7. If something goes wrong

- **Replays fail only on the first or last word.** Check `E_INIT`/`have_class` at batch start, and the
  `have_next` gate (`test_stream_edge_flips`).
- **The clean simulation fails with depth 1.** The stream wasn't hardened (a class under 4 words, or an
  unduplicated one-literal clause with `CC_LA = 1`). Bundles from `RedressLookaheadBackend` always are;
  hand-built replays must use `codec_la.encode`.
- **`class_boundary` looks wrong in waves.** Check that you're looking at state 6. In other states it's
  computed from the wrong `instr_rdata`.
- **To regenerate `build/la_stage4`**, run this from the repo root:
  ```python
  from pathlib import Path
  from matador.backends.redress.rtl import RedressBackend
  from matador.backends.redress_lookahead import campaign
  from matador.backends.redress_lookahead.config import RedressLookaheadConfig
  from matador.backends.redress_lookahead.replay import build_replay_bundle, flip
  from matador.backends.redress_lookahead.rtl import RedressLookaheadBackend
  out = Path("build/la_stage4"); out.mkdir(parents=True, exist_ok=True)
  tmir = campaign.make_synthetic_tmir(n_classes=4, cpc=12, density=0.08, n_samples=32, seed=2)
  tmir.to_yaml(out / "model.yaml")
  cfg = RedressLookaheadConfig.model_validate(dict(model_path=str(out/"model.yaml"),
        output_dir=str(out/"lookahead"), target_fpga="xc7z020", max_includes=4096,
        max_features=256, max_classes=4, batch_width=32))
  flipped = flip(RedressLookaheadBackend().encode_stream(tmir, cfg), 45, 13)
  build_replay_bundle(tmir, flipped, cfg, mode=cfg.decoder_mode, backend_cls=RedressLookaheadBackend)
  build_replay_bundle(tmir, flipped, cfg.model_copy(update={"output_dir": out/"as_built"}),
                      mode={}, backend_cls=RedressBackend)
  ```

## 8. Answers

1. `S_INFER_ACCUM`. The read is issued in `S_INFER_DECODE` (registered, so visible during FEAT), and
   the BRAM updates `rdata` on the edge into ACCUM.
2. At the last word no look-ahead read is issued, so `instr_rdata` still holds the *current* word, and
   "next == current" always confirms. That's mutation M3. If a read *were* issued past the end, it
   would return stale memory, or wrap to address 0 if the stream fills the memory.
3. The next correct word then differs from the latch and gets confirmed by the word after it, which is a
   spurious boundary. The cascade is back (mutation M2: 4 tests fail).
4. At decision time you need words i+1 *and* i+2. Only one read fits in the idle slot before ACCUM, and
   the BRAM has one read port. You'd build a prefetch window: run the decoder D words behind memory,
   filling D words at batch start (stretch goal).
5. Uniform random picks over about 400 words land on word 0 or the last word with probability ~0.5%
   each. Rules that only act at the edges need *targeted* faults. Exhaustive campaigns don't have this
   gap, but random replays do.
