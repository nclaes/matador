"""Hardened MUSTANG encoder for the look-ahead decoder.

encode() starts from codec.mustang_encode() -- the stored format does not
change -- and only adds words, using two freedoms the format already has:

  * Duplicating a literal inside a clause is free on clean data, because
    x AND x = x.
  * Any word of a class can be repeated without changing its class.

It applies them for two reasons:

  (a) cc_lookahead: a one-literal clause changes CC for exactly one word, which
      looks the same as a flipped CC bit, so the CC rule would reject it.
      Storing that literal twice gives the change a confirming next word.
  (b) e_depth > 0: a class shorter than MIN_CLASS_WORDS[e_depth] can still
      cascade under a single E flip (see tests/backends/
      test_redress_lookahead_codec.py, which finds these minimums by exhaustive
      search). Short classes are padded by repeating their last word.

The hardened stream still decodes correctly with the as-built decoder, so a
hardened model is backward compatible with the redress backend.
"""

from __future__ import annotations

from matador.backends.redress.codec import mustang_encode

# Words per class that each E look-ahead depth pads to, chosen from the
# exhaustive search in tests/backends/test_redress_lookahead_codec.py:
#   depth 1: 4 -- the minimum for which no SINGLE E flip cascades. (No length
#            survives two adjacent flips at depth 1.)
#   depth 2: 7 -- the minimum for which no pair of ADJACENT E flips cascades,
#            the reason to choose depth 2 (single flips need only 6).
# Depth 0 has no such length (every interior E flip cascades): no padding.
MIN_CLASS_WORDS = {0: 1, 1: 4, 2: 7}


def split_stream(words: list[int]) -> list[list[list[int]]]:
    """Split a CLEAN stream into classes -> clauses -> words, using the same
    edge rule as the decoder (a new class/clause starts where E/CC differs
    from the previous word)."""
    classes: list[list[list[int]]] = []
    prev = None
    for w in words:
        E, CC = (w >> 13) & 1, (w >> 14) & 1
        if prev is None or (prev >> 13) & 1 != E:
            classes.append([[w]])
        elif (prev >> 14) & 1 != CC:
            classes[-1].append([w])
        else:
            classes[-1][-1].append(w)
        prev = w
    return classes


def harden(words: list[int], *, e_depth: int, cc_lookahead: bool) -> list[int]:
    """Apply (a) and (b) from the module docstring to a clean stream."""
    if e_depth not in MIN_CLASS_WORDS:
        raise ValueError(f"no minimum class length known for e_depth={e_depth}")
    out: list[int] = []
    for clauses in split_stream(words):
        if cc_lookahead:
            clauses = [c * 2 if len(c) == 1 else c for c in clauses]
        flat = [w for c in clauses for w in c]
        shortfall = MIN_CLASS_WORDS[e_depth] - len(flat)
        if shortfall > 0:
            flat += [flat[-1]] * shortfall
        out.extend(flat)
    return out


def encode(tmir, *, e_depth: int, cc_lookahead: bool) -> list[int]:
    """mustang_encode(tmir), hardened for the given look-ahead mode.
    Raises codec.MustangEmptyClassError like mustang_encode()."""
    return harden(mustang_encode(tmir), e_depth=e_depth, cc_lookahead=cc_lookahead)
