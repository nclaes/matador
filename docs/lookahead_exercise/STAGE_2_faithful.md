# Stage 2 — A fault-faithful software model and your first fault injections

## 1. What changed and why

- **New package `matador/backends/redress_lookahead/`.** Nothing is registered yet; registration is Stage 5.
- **`faithful.py`: `decode_faithful()`.** It copies `redress_accel.v` register by register, and each
  step names the RTL line it mirrors. Specifically:
  - one score bank per class *index*;
  - a `CLASS_WIDTH`-bit class counter that wraps;
  - votes to classes ≥ `MAX_CLASSES` are dropped;
  - argmax only looks at k < n_classes, with ties going to the lowest index;
  - the offset is truncated to `FEAT_ADDR_W` bits.
- **Convention for unloaded features:** Feature FIFO entries ≥ n_features (X in simulation) read as **0**.
  Replays skip flips that would read them (`replay_safe()`).
- **`replay.py`: corrupted-model replay.** A throwaway `RedressBackend` subclass (using the Stage 1 hooks)
  loads a flipped stream into the RTL and expects `decode_faithful`'s predictions.
- **`campaign.py`: exhaustive single-bit campaign.** It reports N × P in total and by field
  (P, CC, E, offset, L). Its CLI takes `--synthetic` or `--model`.
- **Tests, 9 in total:**
  - Clean-stream equality with the codec *and* `reference.predict` (5 geometries).
  - A flip-by-flip codec-vs-faithful audit.
  - 22 corrupted replays through the as-built RTL.
  - A **negative control**: the same harness with the codec's expectations must FAIL. A replay check
    that can't fail proves nothing.
  - A campaign check.
- **Bug found and fixed in my own helper:** `run_iverilog` called `vvp` on a relative path from inside
  `sim/`, which made every run "fail" when the bundle path was relative. The negative control exposed it.

**Where the code disagrees with EXERCISE.pdf**

| EXERCISE.pdf | This repo |
|---|---|
| Strawman: codec and RTL-faithful model "disagreed on 56 of 1,134 single flips of P, CC or E" | On this repo's synthetic model the disagreement is much larger: **383 of 1,170** (`max_classes=4`). Nearly every interior E flip pushes the class index past the end, and the codec then predicts a class that doesn't exist (Section 5). The models differ, so the counts differ. The *causes* are the three the brief lists. |
| E accounts for 96–99.6% of N × P (strawman, real models) | **68%** on this ~390-word synthetic model. Per bit, E is still about 48× more sensitive than an offset bit. A cascade shifts the classes *after* the flip, so its damage grows with model length, and real models are longer. Stage 6 measures a real one. |
| `decode_faithful` returns predictions | It returns `(pred_class per lane, scores[W, MAX_CLASSES])`. The emulator (Stage 5) needs the scores. |

## 2. Files touched

- `matador/backends/redress_lookahead/__init__.py`: new package
- `matador/backends/redress_lookahead/faithful.py`: `decode_faithful`, `Geometry`, `predict_batches`
- `matador/backends/redress_lookahead/replay.py`: `flip`, `replay_safe`, `build_replay_bundle`, `run_iverilog`
- `matador/backends/redress_lookahead/campaign.py`: campaign engine and CLI, plus the synthetic model
- `tests/backends/test_redress_lookahead_faithful.py`: 9 tests
- `docs/lookahead_exercise/STAGE_2_faithful.md`: this file

## 3. Run it yourself

```
$ pytest -q tests/backends/test_redress_lookahead_faithful.py
9 passed in 13.28s
$ pytest -q -s tests/backends/test_redress_lookahead_faithful.py -k disagreements
[codec vs faithful] {'flips': 3015, 'agree': 2631, 'past_end': 383, 'short': 0}
```
That second run checks every replay-safe single flip of all 16 bits.
Whenever the decoder visits exactly classes 0..n−1, the two models **always** agree (2,631 flips).
All 383 disagreements push the class index past the end.

**The campaign** (as built, synthetic model):
```
$ python -m matador.backends.redress_lookahead.campaign --synthetic --variant v0
model: 4 classes, 15 features, 390 as-built words; 64 test vectors; batch_width=32, max_classes=16
variant   words   extra     N x P    gain          P       CC        E   offset        L
----------------------------------------------------------------------------------------
v0          390    0.0%    290.06      1x       6.70    24.11   197.67    49.50    12.08
```
It takes about 6 s for 6,240 flips × 64 vectors.

**The negative control, step by step.** This is the most instructive run of the stage.
The script is `matador/backends/redress_lookahead/replay.py` plus about 20 lines; the test
`test_corrupted_replay_as_built_rtl_matches_faithful` contains the same check.
```
E flip at word 0: faithful != codec on 31 of 64 vectors
expected = codec   : FAIL | mismatching beats: 31
expected = faithful: PASS | ['tb_redress_model: ALL 64 TESTS PASSED']
```
The same corrupted Instruction Memory and the same RTL: the codec's expectations fail on 31 beats,
and `decode_faithful`'s pass all 64.

## 4. Look here

- `faithful.py:116-135` next to `redress_accel.v:371-411`. Read them line for line.
  Each Python statement carries its RTL line number as a comment.
- `faithful.py:99` (`flush`): `if current_class < max_classes`. This is the "vote dropped" case.
  A Verilog write to `scores[cls]` past the end of the array has no target.
- `faithful.py:127`: `% class_mod`. This is the counter wrap the codec doesn't have.
- `faithful.py:155-157`: argmax. Class 0 is always the starting candidate; other classes only count if k < n_eff.
- `replay.py:40-59`: the replay harness is only 20 lines, because the Stage 1 hooks do the work.
- The campaign table: E is 1 bit out of 16 but accounts for 68% of N × P.

## 5. Concept notes

**The vocabulary, in practice**

| Term | Here |
|---|---|
| Fault model | One stored bit flipped. It persists until reload, so it's present for every test vector. |
| Campaign | Flip each of the 16 × 390 bits once and run the 64 vectors through `decode_faithful`. |
| P (per bit) | The fraction of the 64 predictions that change, compared with the *fault-free run* and not the true label. |
| N × P | The sum of P over all 6,240 bits: 290 "effective sensitive bits" as built. |
| Fault-faithful | Gives the RTL's answer on corrupted streams. Proved by replay, not by inspection. |

**Why the codec and the RTL disagree once bits flip.** The causes below were all measured with
`P/CC/E` flips on the same synthetic model. The first disagreement in each geometry is a single E flip on word 0.

| Cause | Geometry | Disagreeing flips | Example: E flip on word 0 |
|---|---|---|---|
| Counter **wrap** + per-index accumulation | `max_classes=4` (CLASS_WIDTH=2) | 383 / 1170 | The codec predicts classes {0,1,2,**3,4**}; the RTL predicts {0,1,2}. Class index 4 wraps to 0, so its votes *add to* class 0's. |
| **argmax limit** (k < n_classes) | `max_classes=16` | 356 / 1170 | The codec predicts class **4** (doesn't exist); the RTL stays in {0..3}. |
| **Dropped votes** (cls ≥ MAX_CLASSES) | 3 classes, `max_classes=3` | 281 / 855 | The codec predicts class **3**; the RTL has no `scores[3]`, so those votes vanish. |

Worked example of the cascade behind all three:
```
clean   E: 1 1 1 1 | 0 0 0 | 1 1 1 1 | 0 0 0 0      classes 0 | 1 | 2 | 3
word 0 flipped
        E: 0 1 1 1 | 0 0 0 | 1 1 1 1 | 0 0 0 0
latch:     0→1 at word 1 is a boundary, so every later word sits one class higher
decoded:   0 1 1 1 | 2 2 2 | 3 3 3 3 | 4 4 4 4
```
Class "4" is the original class 3. The codec scores it as a fifth segment and can predict it.
The RTL wraps it to 0 (`max_classes=4`), ignores it in argmax (`max_classes=16`), or drops its votes
(`max_classes=3`). An *interior* E flip produces **two** boundaries (into the flipped word, and out of
it), so later classes shift by two.

**Codec ≠ RTL, but only in one situation.** `test_single_flips_codec_vs_faithful_disagreements_are_explained`
proves the converse: if the decoder visits exactly classes 0..n−1, the two always agree. Per-segment and
per-index accumulation only differ when an index is revisited, which needs a wrap.

## 6. Check yourself

1. Why compare against the fault-free prediction rather than the true label?
2. Why can memory flips be modelled as present for the whole test set, while flip-flop flips need a random time?
3. Before running, predict which field dominates N × P. Were you right, and why?
4. Name two reasons the stock codec disagrees with the RTL once bits are flipped.
5. Why does the replay test also run the codec's expectations through the RTL?

## 7. If something goes wrong

- **The replay fails with no `FAIL out[` lines.** The simulation didn't run. Check the path handling
  (this stage's own bug), or that `iverilog`/`vvp` are on `PATH`.
- **The replay disagrees on an offset flip.** The flip probably reads an unloaded Feature FIFO entry (X).
  Check `replay_safe()` with the right `feat_addr_w` and `n_features`.
- **`IndexError` in `codec.mustang_decode_and_infer`.** The codec indexes `X_pack[offset]` without
  truncation. Offset flips beyond n_features crash it, which is another way it differs from the RTL.
- **The campaign is slow.** It does 16 × n_words × n_batches decodes. Use `--vectors` to subsample,
  or a smaller model.

## 8. Answers

1. P measures how much the *fault* changes the output. Comparing with labels would mix the model's own
   errors into the measurement: a wrong-but-unchanged prediction isn't fault sensitivity.
2. A memory bit stays flipped until the model is reloaded, so every inference sees it and a static
   corrupted stream is exact. A flip-flop is rewritten all the time, so whether a flip matters depends on
   whether it propagates before the next write. That's a property of *when* it happens.
3. E. It's one bit per word, but an interior E flip adds two spurious class boundaries and shifts every
   later class. Measured: 198 of 290 (68%) here. Offset is second in *total* (49.5) only because it is
   12 bits wide; per bit it is about 4.1, against 198 for E.
4. Any two of: per-segment scoring vs per-index accumulation (they differ after a wrap); no counter wrap
   in the codec; the RTL argmax ignores classes ≥ n_classes (and offers unvisited ones at score 0);
   the RTL drops votes to classes ≥ MAX_CLASSES; the offset is truncated to FEAT_ADDR_W bits in the RTL.
5. To prove the harness can fail. If both "expected = codec" and "expected = faithful" passed, the check
   would be measuring nothing. That's exactly how the `vvp` path bug in `run_iverilog` was caught.
