# Stage 3 — Look-ahead in software and the hardened encoder

## 1. What changed and why

- **`decode_faithful` now implements the look-ahead.** An E change is accepted only if the next
  `e_depth` words exist and all carry the new E. On a rejection the latch is **not** updated.
  When `e_depth > 0`, the E latch starts at `E_INIT = 1` and counts as already set, so word 0 is
  checked too. `cc_lookahead=True` applies the same one-word rule to CC. Depth 2 is implemented
  in software only, as a stretch comparison.
- **New `codec_la.py`.** `encode(tmir, *, e_depth, cc_lookahead)` calls `codec.mustang_encode`
  and then hardens the stream:
  - with CC look-ahead, it duplicates the literal of every one-literal clause;
  - it pads any class shorter than `MIN_CLASS_WORDS[e_depth]` by repeating its last word.
- **`decode_faithful(..., class_trace=[])`** records the class each word lands in, for the
  minimum-length search.
- **The campaign has variants `v3`, `v4` and the software-only `v3d2`/`v4d2`.**
- **9 new test functions (31 test cases):**
  - clean equality for 5 variants × 4 densities;
  - include-array round trip;
  - backward compatibility over 40 random models;
  - an exhaustive **minimum-class-length search**;
  - campaign checks;
  - a CC-protection check.

**Where the code disagrees with EXERCISE.pdf, or adds to it**

| EXERCISE.pdf | Measured here |
|---|---|
| §2.5: minimum class length 4 (one-word look-ahead), 6 (two-word), 7 for adjacent flips | **Confirmed by exhaustive search:** 4 / 6 / 7. No length works as built, and no length works with one-word look-ahead under adjacent flips. |
| v4 adds CC protection | **Partly.** CC look-ahead only protects a clause's 3rd through second-to-last words. Flips on the 1st, 2nd or last word still move a clause boundary. On a very sparse model (about 2 literals per clause), v4's CC total went *up* (8.97 → 9.94) because the duplicated words add flippable bits. On the 390-word synthetic model it went down (24.11 → 17.38). I didn't assert "v4 < v3" in the tests, because it isn't true in general. |
| Total N × P gains of 19–266× (v3) and 48–1,116× (v4) on real models | E alone drops **420×** (197.67 → 0.47). The *total* only drops **3×**, because on this synthetic model the other fields (offset 49.5, CC 24.1) were already a third of the total. Stage 6 measures a real model. |

## 2. Files touched

- `matador/backends/redress_lookahead/faithful.py`: look-ahead, `e_init`, `class_trace`
- `matador/backends/redress_lookahead/codec_la.py`: new: `MIN_CLASS_WORDS`, `split_stream`, `harden`, `encode`
- `matador/backends/redress_lookahead/campaign.py`: variants v3, v4, v3d2, v4d2
- `tests/backends/test_redress_lookahead_codec.py`: new tests
- `docs/lookahead_exercise/STAGE_3_lookahead_software.md`: this file

## 3. Run it yourself

```
$ pytest -q -s tests/backends/test_redress_lookahead_codec.py -k minimum
[min class length]  e_depth | single flip | two adjacent flips
[min class length]        0 |        None |               None
[min class length]        1 |           4 |               None
[min class length]        2 |           6 |                  7
$ pytest -q tests/backends/test_redress_lookahead_codec.py
31 passed in 4.82s
$ pytest -q tests/backends/test_redress_lookahead_codec.py tests/backends/test_redress_lookahead_faithful.py \
            tests/backends/test_redress_codec.py tests/backends/test_redress_rtl.py
101 passed in 37.95s
```

The campaign on the synthetic model (about 30 s):
```
$ python -m matador.backends.redress_lookahead.campaign --synthetic --variant v0 v3 v4 v3d2 v4d2
model: 4 classes, 15 features, 390 as-built words; 64 test vectors; batch_width=32, max_classes=16
variant   words   extra     N x P    gain          P       CC        E   offset        L
----------------------------------------------------------------------------------------
v0          390    0.0%    290.06      1x       6.70    24.11   197.67    49.50    12.08
v3          390    0.0%     92.86      3x       6.70    24.11     0.47    49.50    12.08
v4          425    9.0%     85.92      3x       6.70    17.38     0.44    49.67    11.73
v3d2        390    0.0%     93.05      3x       6.70    24.11     0.66    49.50    12.08
v4d2        425    9.0%     86.11      3x       6.70    17.38     0.62    49.67    11.73
```
How to read it:
- **v3** stores the same 390 words, so it costs nothing, and only E changed.
- **v4** stores 9% more words, from duplicated one-literal clauses.
- **Depth 2** is no better for *single* flips here. Its benefit is surviving *adjacent* flips (the table above).

## 4. Look here

- `faithful.py:135-141`: the rejection branch. The one line that decides whether the cascade comes back
  is the latch update at `faithful.py:155`, `if not have_class or class_boundary`. Because
  `class_boundary` is already False for a rejected change, the latch keeps its old value.
- `faithful.py:108-112`: `have_class = e_depth > 0` and `E_prev = E_INIT`. Word 0 is now checked like any other word.
- `codec_la.py:31`: `MIN_CLASS_WORDS = {0: 1, 1: 4, 2: 6}`. The test at
  `tests/backends/test_redress_lookahead_codec.py:143` derives these and fails if they ever disagree.
- `codec_la.py:59` and `:63`: the two hardening steps, in two lines.
- `tests/backends/test_redress_lookahead_codec.py:111` (`_cascades`): what "cascade" means, as code.
  If a word whose class is two or more away from every flipped word's class changes class index,
  the flip has cascaded.

## 5. Concept notes

**The decode rule (depth 1)**
```
E change at word i  AND  word i+1 exists  AND  E(i+1) == E(i)   ->  accept: boundary, latch <- E(i)
E change at word i  otherwise                                      ->  reject: no boundary, latch unchanged
```

**Why a rejected change must leave the latch alone.** One interior flip, depth 1:
```
words:      ... A  A  A  B* A  A ...        (B* = the flipped word)
word B*:    differs from latch A, next word is A != B  -> reject, latch stays A
next A:     equals latch A                             -> no boundary.  Nothing happened.
If the latch HAD taken B: next A would differ from B, and with A A following it would be
accepted -> a spurious boundary. That's the cascade coming back.
```

**Why 4 words and not 2.** A 3-word class `[a b c]` between neighbours of the other E value
(`test_three_word_class_fails_with_one_word_lookahead`):
```
prev class (E=0) | a b c (E=1) | next class (E=0)
flip b to 0:
  a: change 0->1, next b is now 0       -> reject
  b: 0 == latch 0                       -> nothing
  c: change 0->1, next word is E=0      -> reject
  next class: 0 == latch 0              -> nothing
=> both real boundaries are lost: the class is swallowed and every later class shifts.
With 4 words [a b c d], flipping b: c sees next d == 1 -> accept. The boundary moves by two words. That's local.
```

**Why one-literal clauses must be duplicated for the CC rule.** A one-literal clause changes CC on
exactly one word, and the next word (the next clause) changes CC back. That is indistinguishable
from a single flipped CC bit, so the rule rejects it and merges the clause into its neighbour.
Storing `x` twice gives `x AND x = x` (free on clean data) and gives the change a confirming next word.

**Duplicates aren't free under faults.** Every added word is 16 more bits to flip. An offset flip
in the duplicate adds a *new* literal to the clause, and a P flip on a clause's first word still
changes its polarity. Hence v4's offset total rises slightly (49.50 → 49.67).

**The CC analogue of the minimum length.** Only a clause's 3rd through second-to-last words are
protected (`test_cc_lookahead_protects_interior_clause_words`). Flipping the 2nd word makes the
look-ahead at the 1st word reject the real boundary, so the boundary moves by two words.
That is the same mechanism as the 4-word class minimum.

## 6. Check yourself

1. Why must a rejected E change leave the latch alone?
2. Why is the minimum 4 words and not 2? Describe the flip that breaks a 3-word class.
3. Why does the CC rule misread an unduplicated one-literal clause?
4. A duplicated literal is free on clean data. Why isn't it entirely free under faults?
5. Why didn't depth 2 improve single-flip N × P here, and what *does* it buy?

## 7. If something goes wrong

- **A hardened stream gives different clean predictions.** Check `split_stream` on that stream. It
  assumes a *clean* stream (E and CC edges are real boundaries). Never harden an already-corrupted stream.
- **`ValueError: no minimum class length known for e_depth=3`.** `MIN_CLASS_WORDS` only has the
  depths that have been verified. Extend the search test first, then the table.
- **The minimum-length test fails after an edit.** The decode rule changed. Print `_trace(...)` for
  the failing length tuple and compare it with the clean class sequence.
- **`MustangEmptyClassError`.** Hardening doesn't change this. An empty class still has no word to
  carry its E (stretch goal: an always-false clause).

## 8. Answers

1. If the latch took the flipped value, the next correct word would differ from it. With a confirming
   word after it, that difference is accepted as a second, spurious boundary, and the cascade is back.
2. In a 3-word class, flipping the middle word makes *both* real boundaries fail their look-ahead:
   the first word's next word is now wrong, and the last word's next word belongs to the next class.
   The class is swallowed and every later class shifts. With 4 words, a flipped 2nd word still leaves
   words 3 and 4 to confirm each other, so the boundary only moves by two words.
3. Its CC change lasts exactly one word, which is exactly what a flipped CC bit looks like. The next
   word carries a different CC (the next clause's), so the change is rejected and the clause merges
   into its neighbour.
4. The extra word adds 16 more bits that can flip. An offset flip in the duplicate adds a new,
   different literal to the clause, and an L flip negates it. The clause is then no longer `x AND x`.
5. Single-flip residuals at depth 1 are already only boundary movements (0.47 here). Depth 2 can't
   remove those; its job is surviving *two adjacent* flips (minimum class length 7). Depth 1 can't
   survive those at any class length.
