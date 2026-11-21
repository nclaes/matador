# Stage 6 — A real model and real numbers

## 1. What changed and why

- **Trained the `redress.md` mammographic model** (80.31%, seeded, reproducible) and ran both backends
  end to end: generate, iverilog + Verilator simulate, and `emulate --verify`. All of them pass.
- **Ran the exhaustive campaign on the real model with the full 193-vector held-out set**, for v0, v3, v4
  and the software-only depth-2 variants.
- **Replayed 200 corrupted streams through the two RTLs** (50 uniform + 50 targeted E/CC, per RTL).
  All 200 match `decode_faithful`.
- **Wrote `docs/walkthroughs/redress_lookahead.md`** with verbatim outputs. That walkthrough is the main
  document for this stage; this README is the learning layer on top of it.
- **`campaign.py` gained `--test-data`**, which loads a booleanised `*_test.txt` with the label column
  dropped. The TMIR only embeds 10 test vectors, too few for P.
- **New `docs/lookahead_exercise/scripts/replay_sample.py`**: a reusable random corrupted-replay sampler,
  with `--fields` for targeted sampling.

**Where the code disagrees with EXERCISE.pdf**

| EXERCISE.pdf | Measured |
|---|---|
| "Mammographic has many one-literal clauses … 10.4% in Table 12" | **This** trained model has **one** one-literal clause, so v4 costs **+0.1%** (1 word). Its clauses are long (~27 literals). |
| E dominates instruction-memory sensitivity (96–99.6%) | Not for this model: **CC 144.47, E 140.10** of 288.03. v3 alone gives only 1.9×; v4 gives **85×**. |
| Table 13 mammographic: v0 51.8, v3 0.96, v4 0.30 | This model: v0 288.03, v3 148.15, v4 3.37. The models differ. The size of the v4 effect (about two orders of magnitude) carries over; v3's share does not. |
| Follow `redress.md` (dev container) | Run outside the container. `matador train` needs the vendored `tmu`, so I ran it through `python -c` from the repo root (walkthrough Step 1). |

## 2. Files touched

- `docs/walkthroughs/redress_lookahead.md`: new, with verbatim outputs
- `matador/backends/redress_lookahead/campaign.py`: `--test-data`, `load_test_data()`
- `docs/lookahead_exercise/scripts/replay_sample.py`: new
- `docs/lookahead_exercise/STAGE_6_real_model.md`: this file

## 3. Run it yourself

Follow `docs/walkthroughs/redress_lookahead.md` Steps 1–4. Everything lands in `build/la_stage6/`.
The two headline commands:
```
python -m matador.backends.redress_lookahead.campaign --model <TMIR>.yaml \
    --test-data build/la_stage6/booleanised/mammographic_test.txt --variant v0 v3 v4      # ~6 min
python docs/lookahead_exercise/scripts/replay_sample.py --model <TMIR>.yaml \
    --test-data build/la_stage6/booleanised/mammographic_test.txt --fields E CC --seed 2  # ~30 s
```
Expected output (verbatim, from this machine):
```
variant   words   extra     N x P    gain          P       CC        E   offset        L
v0         1094    0.0%    288.03      1x       0.15   144.47   140.10     2.22     1.09
v3         1094    0.0%    148.15      2x       0.15   144.47     0.23     2.22     1.09
v4         1095    0.1%      3.37     85x       0.15     0.43     0.23     1.79     0.78
...
redress (as built)            50      50/50        37   CC 23, E 14
redress_lookahead (v4)        50      50/50         1   E 1
```

## 4. Look here

- The campaign table against the strawman's Table 13 row (walkthrough, Step 3). Same size of effect for
  v4; a very different split between E and CC.
- The **CC** column: 144.47 → 144.47 → 0.43. v3 doesn't touch it, and v4 removes it.
- The replay "visible" column, uniform (3/50) against targeted (37/50). This is the Stage 4 lesson
  again: random sampling mostly tests faults that don't matter.
- The generated `build/la_stage6/redress_lookahead/RTL/README.md`: "stored words = 1,095 (plain
  MUSTANG: 1,094; +0.1% from hardening)".

## 5. Concept notes

**Why CC is so damaging in *this* model.** With ~27 literals per clause, almost no clause fires on a
given input. A CC flip on an interior word of such a clause triggers two clause boundaries:
```
clause:   l1 l2 ... l13 [l14*] l15 ... l27           (* = CC flipped)
as built: [l1..l13] [l14] [l15..l27]                   -> three clauses
```
`[l14]` is a one-literal clause that fires whenever that one literal is true, and the shorter fragments fire far more
often than the original. Several new votes per input change the class sums. In a model of short clauses,
a split changes much less. So the E/CC split is a property of the trained model, not of the format.

**Synthetic results vs real ones.**

| | Synthetic (Stage 3) | Mammographic (this stage) |
|---|---|---|
| Words, classes | 390, 4 | 1,094, 2 |
| v0 N × P (E / CC) | 290 (198 / 24) | 288 (140 / 144) |
| v3 gain (total) | 3× | 1.9× |
| v4 gain (total) | 3× | **85×** |
| v4 word cost | +9.0% | +0.1% |

The synthetic model was dominated by E and had many short clauses (high v4 cost, little CC to remove).
The real model has long clauses (low v4 cost, a lot of CC to remove). **Measure on the model you'll
deploy.**

## 6. Check yourself

1. Mammographic has many one-literal clauses, per the brief. What overhead did v4 cost here, and does
   it match the 10.4% in Table 12?
2. Why did v3 alone help so little on this model, when it removed 607× of the E sensitivity?
3. What does "200 / 200 replays matched" let you conclude about Step 3's table, and what doesn't it?
4. Why is the look-ahead bundle 5 cycles slower in total, when Stage 4 says the look-ahead costs no cycles?

## 7. If something goes wrong

- **`TypeError: TMBaseModel.__init__() missing 1 required positional argument: 'K'`.** That's the
  site-packages `tmu`. Run training via `python -c "...main()"` from the repo root, or inside the dev
  container (`make shell`).
- **`model encodes to N … exceeding max_includes`.** The walkthrough's `max_includes: 1024` is too small
  for this model (1,094 words). Use 2048.
- **The campaign is slow.** It does 16 × 1,094 flips × 7 batches per variant. Use `--vectors 64` for a
  quick look; the full set is for the record.

## 8. Answers

1. +0.1% (one word). The trained model has a single one-literal clause, so the brief's premise doesn't
   hold for it, and Table 12's 10.4% belongs to a sparser model.
2. Because E was only half of the as-built N × P. CC flips (144 of 288) are untouched by v3, so the
   total can't drop below ~148. The gain from a fix is limited by the share of the problem it addresses.
3. That `decode_faithful` agrees with the RTL on every corrupted stream tried, including 37 that changed
   predictions, so the software campaign stands for the hardware on this model. It doesn't prove agreement
   on *all* 17,504 flips, and it says nothing about flip-flop faults (Stage 7).
4. The hardened stream has one more word, and every word costs 4 decode cycles plus 1 load beat:
   5 cycles in total. Per word, the look-ahead adds nothing (Stage 4 compared identical streams).
