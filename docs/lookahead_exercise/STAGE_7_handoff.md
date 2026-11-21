# Stage 7 — Hand-off for flip-flop injection (EMBER), and a mini saboteur

## 1. What changed and why

- **A register table for the team's EMBER campaign** (Section 5): every register whose behaviour in
  `redress_la_accel.v` differs from `redress_accel.v`, with its width, how long it is live per
  instruction, and a suggested state class and Tier 1 protection. **The look-ahead adds no flip-flops.**
  It changes *when* and *how* existing registers are written and read.
- **New `docs/lookahead_exercise/scripts/saboteur.py`, a mini saboteur for learning.** It rewrites the
  model testbench into `tb_saboteur.v`, which inverts one bit of a named register through a hierarchical
  reference at a random inference cycle (`+FLIP_CYCLE=`). It then logs whether any output changed.
  It ran 200 trials × 6 registers × 2 designs.
- **New `tests/backends/test_redress_lookahead_saboteur.py`**, the deterministic version (3 tests):
  - a run with no flip passes;
  - a current-word E flip (`instr_latched[13]` in ACCUM) breaks only the as-built design;
  - an **E latch flip (`E_prev`) breaks both designs**.
- **Main finding: the look-ahead protects the *stored* E bits, not the E *latch*.** A flipped `E_prev`
  makes the next correct word look like a change, and the word after it confirms the change. That causes
  a spurious boundary and a one-class cascade in both designs. `E_prev`, `CC_prev` and `current_class`
  are the Tier 1 targets.

**Where the code disagrees with EXERCISE.pdf**

| EXERCISE.pdf | This repo |
|---|---|
| Saboteur example target `dut.instr_latched[13]` | Used. The look-ahead word lives in the BRAM output register, so that target is `dut.instr_mem.rdata[13]` (`instr_rdata` is a wire in the top module). |
| "Run about 200 trials on each design" | 200 trials per register per design, on one 32-vector batch of the 390-word synthetic model. |

## 2. Files touched

- `docs/lookahead_exercise/scripts/saboteur.py`: the mini saboteur
- `tests/backends/test_redress_lookahead_saboteur.py`: 3 deterministic saboteur tests
- `docs/lookahead_exercise/STAGE_7_handoff.md`: this file

## 3. Run it yourself

```
$ pytest -q tests/backends/test_redress_lookahead_saboteur.py
3 passed in 0.76s
$ python docs/lookahead_exercise/scripts/saboteur.py --trials 200 --out build/la_stage7      # about 3 min
| register | what it is | redress | redress_lookahead |
|---|---|---|---|
| `E_prev` | E latch (class boundary reference) | 200/200 | 198/200 |
| `CC_prev` | CC latch (clause boundary reference) | 153/200 | 138/200 |
| `have_class` | E latch valid flag | 1/200 | 0/200 |
| `instr_latched[13]` | current word's E (DECODE..ACCUM) | 109/200 | 4/200 |
| `instr_rdata[13]` | BRAM output E (look-ahead word in ACCUM) | 55/200 | 2/200 |
| `current_class[0]` | class counter LSB | 200/200 | 200/200 |
```
Each trial is one `vvp` run. Every trial checks that the `SABOTEUR: inverted ...` line actually
appeared, so a flip that never happened can't be counted as "no change".

**One trial by hand** (after the script has generated `build/la_stage7`):
```
$ cd build/la_stage7/redress_lookahead/RTL/sim
$ iverilog -g2001 -Wno-timescale -DSAB_TARGET=dut.E_prev -o sab ../src/*.v tb_saboteur.v
$ vvp -n sab                      # no +FLIP_CYCLE: no flip
tb_redress_model: ALL 32 TESTS PASSED
$ vvp -n sab +FLIP_CYCLE=703      # 703 = 4*175 + 3: word 175's ACCUM cycle
SABOTEUR: inverted `SAB_TARGET at inference cycle 703 (state 6)
...
tb_redress_model: FAILED 20/32
```
(The log prints the macro's name, not its value, because iverilog doesn't expand macros inside
string literals. `(state 6)` confirms the flip landed in ACCUM.)

## 4. Look here

- **The saboteur's results for `E_prev` on both designs** (198/200 and 200/200). This is the point of the stage.
- `instr_latched[13]`: 109 → 4. That's the stored-E protection, seen from the flip-flop side.
- `instr_rdata[13]`: 55/200 as built against 2/200 with look-ahead, even though look-ahead makes this
  register live for *more* cycles (Section 5). Liveness and sensitivity are not the same thing.
- `docs/lookahead_exercise/scripts/saboteur.py` `SABOTEUR_BLOCK`: a hierarchical procedural
  assignment (`` `SAB_TARGET = ~`SAB_TARGET; ``) on the *falling* edge. Why the falling edge?
  (Section 6, question 3.)

## 5. Concept notes

**Registers that behave differently in `redress_la_accel.v`.** "Live" means the cycles, out of the four
per instruction (INSTR, DECODE, FEAT, ACCUM), in which a flip can still reach a decision.

| Register | Width | What changed | Live per instruction | State class | Tier 1 suggestion | Saboteur (as built → look-ahead) |
|---|---|---|---|---|---|---|
| `E_prev` | 1 | Reset to `E_INIT` at batch start; updated **only** on a confirmed class boundary | 4 / 4 (read every ACCUM, held across words) | control | TMR | 200 → 198 / 200 |
| `CC_prev` | 1 | Updated only on a confirmed clause boundary (`CC_LA = 1`) | 4 / 4 | control | TMR | 153 → 138 / 200 |
| `have_class` | 1 | Set at batch start (`E_LA_DEPTH = 1`) and constant through inference | 4 / 4, but constant 1 | control | Remove it: tie to 1 when `E_LA_DEPTH = 1` | 1 → 0 / 200 |
| `instr_rdata` (`instr_mem.rdata`) | 16 | Loaded twice per instruction; `[14:13]` of word i+1 read in ACCUM | DECODE (all bits) + ACCUM (`[14:13]`): 2 / 4 (as built: 1 / 4) | `[15:13]` control, `[12:0]` datapath | Parity on `[15:13]` | 55 → 2 / 200 (bit 13) |
| `instr_re`, `instr_raddr` | 1, `INSTR_ADDR_W` | Also issued in DECODE (the look-ahead read) | INSTR + FEAT: 2 / 4 (as built: 1 / 4) | control | Parity on `instr_raddr` | not measured |
| `ptr`, `hdr_n_instr` | `INSTR_ADDR_W` | Now also form `have_next` (read in DECODE and ACCUM) | 4 / 4 (already loop-critical as built) | control | TMR (already warranted as built) | not measured |

There are no new flip-flops. `have_next`, `E_next`, `CC_next`, `e_confirm` and `cc_confirm` are wires.

**Why the latch isn't protected.** One cycle with `E_prev` flipped, in the middle of class 1 (E = 0):
```
latch:  0 -> 1 (flipped)
word j:   E = 0 != latch 1, next word E = 0 == 0   -> CONFIRMED change: boundary, latch <- 0
=> one spurious boundary; every later class sits one index too high (as built: the same, without needing confirmation)
```
The look-ahead asks "is the *word* consistent with its neighbour?". A wrong latch passes that test,
because every word around it is consistent.

**Transient vs persistent.** A flipped *memory* bit is there for every batch until reload. That's why
Stage 2–6 replay a static corrupted stream, and why one flip affects all 64 or 193 vectors. A flipped
*flip-flop* is rewritten: `instr_latched` every instruction, `E_prev` at the next accepted boundary,
everything at the next batch start. Whether it matters depends on *when* it lands, so the saboteur
samples a random cycle, and EMBER-style campaigns sample (register, cycle) pairs.

**What this saboteur is not.** It's not a substitute for EMBER:
- it covers 6 hand-picked registers, not all state;
- "changed" means some output beat differed. It doesn't detect reconvergence, latent corruption,
  or a fault that changes timing without changing the outputs;
- it uses uniform cycle sampling over one batch of one model.

## 6. Check yourself

1. Why does flipping `E_prev` for one cycle behave differently from flipping E in memory?
2. `instr_rdata[13]` is live for *more* cycles with look-ahead (2 of 4, against 1 of 4), but it's far
   less sensitive (2/200 against 55/200). How can both be true?
3. Why does the saboteur flip on the falling clock edge?
4. Why is `have_class` a candidate for *removal*, not protection?

## 7. If something goes wrong

- **`flip at cycle N never happened`.** The cycle is beyond the batch's inference cycles
  (`4 × n_words + 1` per batch), or the testbench finished first. Check `n_words` for that design:
  hardened streams are longer.
- **`iverilog: Unable to bind wire/reg/memory 'dut.instr_rdata'`.** `instr_rdata` is a wire, and you can
  only assign to a reg. Use `dut.instr_mem.rdata[...]`.
- **Every trial says "changed".** Check the no-flip run (`test_no_flip_passes`) first. A broken
  testbench rewrite fails *every* run, with or without a flip.

## 8. Answers

1. A memory E flip is one wrong *word*. Its neighbours are right, so the look-ahead sees an unconfirmed
   change and rejects it. A latch flip is a wrong *reference*. The next correct word differs from it,
   and the word after agrees with that word, so the change is confirmed and a spurious boundary is
   inserted. It's also transient: the next accepted boundary rewrites the latch, but by then every
   later class has already shifted.
2. Liveness is "can a flip reach a decision". Sensitivity is "does the decision then change". In ACCUM,
   a flipped look-ahead E (`E_next`) only matters at a real class boundary (where it can reject the real
   change) or right after a stored fault. Inside a class, `E_cur == E_prev`, so `E_next` is never
   consulted. As built, the DECODE-cycle flip goes straight into `instr_latched`, which is a full
   stored-E fault that the as-built decoder can't reject.
3. The DUT samples and updates on the rising edge. Flipping halfway between edges means the flipped
   value is what the next rising edge reads, and the flip isn't overwritten in the same delta cycle by
   the DUT's own nonblocking assignment to that register.
4. With `E_LA_DEPTH = 1` it's 1 from batch start onwards and is never cleared during inference, so it
   carries no information. Tying it to a constant removes a flip-flop and its failure mode at the same
   time, which is cheaper than triplicating it.

## Open items for the team

- **Multi-bit upsets on adjacent E bits.** With one word per memory row, E bits of *consecutive words*
  sit in adjacent cells. A one-word look-ahead can't survive two adjacent flips at any class length
  (Stage 3 search: "None"). The options are an SRAM address permutation (interleaving) so consecutive
  words aren't physically adjacent, or a two-word look-ahead (minimum class length 7, needs a prefetch window).
- **Minimum class length wording in the paper.** It should be **4** words for one-word look-ahead,
  not 2. The exhaustive search in `tests/backends/test_redress_lookahead_codec.py` confirms 4 (and 6/7 for depth 2).
- **The paper's Figure 3 example.** Check it against the confirmed 4-word minimum and the 3-word-class
  counterexample (`test_three_word_class_fails_with_one_word_lookahead`). I haven't seen the figure.
- **The class-count-table idea (v6).** Store each class's end address three times. Model it and score it
  with `campaign.py` before any RTL.
- **The latches are now the weak point.** `E_prev`, `CC_prev` and `current_class` dominate flip-flop
  sensitivity in both designs (Section 5). Tier 1 TMR on these three is 1 + 1 + `CLASS_WIDTH` bits.
- **CC look-ahead covers clause interiors only.** A flip on a clause's 1st, 2nd or last word still moves
  that boundary (Stage 3). On very sparse models, v4 can be no better than v3 on CC.
- **Unrelated, found along the way:**
  - 7 test failures that predate this work: 5 stale `TestEmulateCLI` tests missing `--backend`, and
    2 TMU weighted/coalesced reference mismatches.
  - The `matador` console script imports site-packages `tmu` 0.8.3, which breaks `matador train`
    (Stage 6).
  - `matador simulate` with a relative `output_dir` was fixed in Stage 5.
