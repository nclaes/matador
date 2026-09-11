"""REDRESS codec — the 16-bit Include-Encoding compression scheme and its
batched decode/inference routine.

Source: T. Rahman, "A Design Methodology for Edge Inference Systems using
Tsetlin Machines" (PhD thesis, Newcastle University, 2025), Chapter 4.3
(Algorithm 11: 16-bit Include-Encoding; Algorithm 12: batched decode/
inference), the basis for the MUSTANG hardware accelerator (Chapter 5.3)
this backend implements.

WHAT THIS FILE PROVIDES
  Classic REDRESS (software, Algorithm 11/12 exactly as published --
  what an MCU would run; also the reference this backend's own RTL is
  checked against):
    encode()             TMIR -> (E, inc_per_class): Algorithm 11, transcribed
                          against matador's own literal/polarity conventions
                          (see "Reconciling conventions" below).
    pack_features()       Boolean feature matrix -> batched W-bit packed words
                          (the X_pack[] structure Algorithm 12 consumes).
    decode_and_infer()   Algorithm 12 verbatim: one batch (<=W datapoints) of
                          packed features -> (best_class, best_score).
    infer()              Convenience wrapper: TMIR + a full X matrix (any
                          n_samples) -> (predictions, scores) in exactly the
                          shape matador.inference.reference.predict() returns,
                          so the two can be diffed directly.

  MUSTANG (hardware, Chapter 5.3's actual on-wire instruction format --
  what this backend's RTL implements and what generate() streams into the
  accelerator; see "The MUSTANG instruction format" below):
    mustang_encode()          TMIR -> single instruction stream (list[int]),
                               no side-channel Inc_per_class array -- class
                               boundaries are carried in-band via the E bit.
    mustang_decode_and_infer() One batch -> (best_class, best_score), reading
                               n_classes from the caller (matching the real
                               hardware's Instruction Header field) instead
                               of a per-class word count.
    mustang_infer()            Convenience wrapper, same shape as infer().

RECONCILING CONVENTIONS (thesis vs. matador)
  The thesis's Algorithm 11 pseudocode assumes two things that do NOT match
  matador's own TMIR/reference-inference conventions, and both had to be
  adapted -- not copied verbatim -- for encode() to actually agree with
  matador.inference.reference.predict():

  1. Literal layout. The thesis's own encoder scans literals in an
     INTERLEAVED order per its Algorithm 11 comment ("fixed literal order
     [B0, B0-bar, B1, B1-bar, ...]"), deriving offset/polarity as
     i=floor(k/2), b0=k mod 2. matador's TMIR stores literals CONCATENATED
     instead: index f<n_features is the positive literal of feature f,
     index n_features+f is its negation (see tm_ir.py, reference.py). This
     file iterates matador's actual (class, clause, feature) array directly
     rather than a linear interleaved k, so "offset" and "b0" below mean
     the same thing the thesis intends (feature index, literal polarity)
     even though the source layout differs. Decode is unaffected either
     way -- it never assumes an iteration order, only that every included
     literal for a clause appears somewhere before that clause's boundary
     is flagged.

  2. Clause polarity convention. The thesis's Algorithm 11 toggles BOTH the
     polarity bit P and the clause-boundary bit C together on every new
     clause (lines 9-11: "if first: P^=1; C^=1"). Given they start equal
     (0,0) and always receive the identical toggle at the identical
     trigger, P and C are then IDENTICAL for the rest of the stream. That
     is not a bug in the thesis -- it is only correct because the thesis's
     own TM library uses ALTERNATING clause polarity (clause 0 positive,
     clause 1 negative, clause 2 positive, ... -- see the pyTsetlinMachine
     convention), so "flip every clause" and "this clause's polarity" are
     the same signal there. matador's convention (tm_accel_gp.v,
     tm_emulator.py, hardwired/rtl.py, inference/reference.py -- all of
     them, consistently) is GROUPED, not alternating: the first half of
     each class's clauses are positive, the second half negative
     (`positive iff (clause_index % clauses_per_class) < clauses_per_class
     // 2`). Blindly transcribing "toggle P every clause" would encode the
     wrong sign for every model using matador's actual convention. Here, C
     still toggles on every new non-empty clause (that part of Algorithm 11
     is convention-independent -- it is purely a "did the clause change"
     marker the decoder needs because empty clauses are skipped and can't
     otherwise be counted). P is instead set directly from the clause
     index: 0 for the first half of a class's clauses, 1 for the second
     half. Decode (Algorithm 12) needs no changes at all for this -- it
     only ever reads back whatever P was latched, it never assumes how P
     was derived.

  Every other part of both algorithms (the 16-bit word layout, the
  batched W-wide AND-accumulate decode loop, the argmax) is a direct,
  unmodified transcription.

16-bit WORD LAYOUT (Algorithm 11, Figure 4.8(c))
  bit15   P   polarity of this include's clause (0 = positive, 1 = negative)
  bit14   C   clause-boundary marker (flips on every new non-empty clause;
              decode detects a boundary by comparing against the previous
              latched C, not by its absolute value)
  bits1-13    offset = feature index (13 bits, features up to 8192)
  bit0        b0 = literal polarity (0 = positive literal, 1 = negated)

No third-party dependencies beyond numpy (already a matador dependency).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from matador.ir.tm_ir import TMIR

# ---------------------------------------------------------------------------
# Word format
# ---------------------------------------------------------------------------
OFFSET_BITS = 13
MAX_FEATURES = 1 << OFFSET_BITS  # 8192, per Algorithm 11's 13-bit offset field


def _pack_word(P: int, C: int, offset: int, b0: int) -> int:
    if not (0 <= offset < MAX_FEATURES):
        raise ValueError(
            f"feature offset {offset} does not fit the 13-bit offset field "
            f"(max {MAX_FEATURES - 1}); REDRESS's 16-bit word cannot address "
            f"more than {MAX_FEATURES} features"
        )
    return (P << 15) | (C << 14) | (offset << 1) | b0


def _unpack_word(word: int) -> tuple[int, int, int, int]:
    P = (word >> 15) & 1
    C = (word >> 14) & 1
    offset = (word >> 1) & ((1 << OFFSET_BITS) - 1)
    b0 = word & 1
    return P, C, offset, b0


# ---------------------------------------------------------------------------
# Algorithm 11 -- 16-bit Include-Encoding
# ---------------------------------------------------------------------------

def encode(tmir: TMIR) -> tuple[list[int], list[int]]:
    """Compress a TMIR model's Include state into the REDRESS 16-bit stream.

    Args:
        tmir: A loaded, validated TMIR with clause_organization=="per_class"
              (REDRESS's grouped-polarity scheme -- see module docstring --
              is only meaningful for per-class organisation; coalesced
              models have no notion of a class-relative clause half).

    Returns:
        E:             list of 16-bit words (Python ints), class-major,
                       one word per included literal. Empty clauses emit
                       nothing, matching matador's own "empty clause never
                       fires" inference convention.
        inc_per_class: list of length n_classes; inc_per_class[c] is the
                       number of words in E belonging to class c, i.e. the
                       running total needed to delimit class c's segment
                       during decode (Algorithm 12's Inc_per_class input).
    """
    tmir.to_includes()
    arch = tmir.architecture
    if arch.clause_organization != "per_class":
        raise NotImplementedError(
            "REDRESS's Include-Encoding assumes per-class clause "
            f"organisation (grouped polarity halves); got "
            f"{arch.clause_organization!r}."
        )

    n_classes = arch.n_classes
    cpc = arch.n_clauses_per_class
    n_features = arch.n_features
    half = cpc // 2

    inc = np.asarray(tmir.representation.includes, dtype=bool)
    if inc.shape != (n_classes, cpc, 2 * n_features):
        raise ValueError(
            f"includes shape {inc.shape} != expected "
            f"{(n_classes, cpc, 2 * n_features)}"
        )

    E: list[int] = []
    inc_per_class: list[int] = []

    for c in range(n_classes):
        C = 0
        start = len(E)
        for n in range(cpc):
            P = 0 if n < half else 1  # matador's grouped polarity, not the
                                       # thesis's per-clause toggle -- see
                                       # module docstring point 2.
            clause = inc[c, n]
            first = True
            for f in range(n_features):
                if clause[f]:  # positive literal of feature f
                    if first:
                        C ^= 1
                        first = False
                    E.append(_pack_word(P, C, f, 0))
                if clause[n_features + f]:  # negated literal of feature f
                    if first:
                        C ^= 1
                        first = False
                    E.append(_pack_word(P, C, f, 1))
        inc_per_class.append(len(E) - start)

    return E, inc_per_class


# ---------------------------------------------------------------------------
# Batched feature packing (Figure 4.9(c))
# ---------------------------------------------------------------------------

def pack_features(X: np.ndarray, W: int) -> list[int]:
    """Bit-pack a (<=W, n_features) Boolean matrix into W-bit feature words.

    X_pack[i] holds feature i's value across all W datapoints in the batch
    (datapoint j's bit lives at bit position j). Rows beyond len(X) are
    treated as 0 (harmless padding -- their results are simply not read
    back by the caller).

    Args:
        X: uint8/bool array, shape (n_datapoints, n_features), n_datapoints <= W.
        W: batch width in bits (16, 32, or 64 per the thesis's MUSTANG
           configurations).

    Returns:
        list of n_features W-bit Python ints.
    """
    X = np.asarray(X, dtype=np.uint8)
    n_datapoints, n_features = X.shape
    if n_datapoints > W:
        raise ValueError(f"{n_datapoints} datapoints exceeds batch width W={W}")

    X_pack = [0] * n_features
    for j in range(n_datapoints):
        row = X[j]
        for i in range(n_features):
            if row[i]:
                X_pack[i] |= 1 << j
    return X_pack


# ---------------------------------------------------------------------------
# Algorithm 12 -- batched decode/inference
# ---------------------------------------------------------------------------

def decode_and_infer(
    E: Sequence[int],
    inc_per_class: Sequence[int],
    X_pack: Sequence[int],
    n_classes: int,
    W: int,
) -> tuple[list[int], list[int]]:
    """Decode E and classify one batch of up to W datapoints, in one pass.

    Direct transcription of Algorithm 12 -- see the thesis for the full
    line-by-line variable-role commentary this mirrors.

    Args:
        E:             encoded includes from encode() (or an RTL-loaded copy
                       reconstructed bit-for-bit the same way).
        inc_per_class: from encode(); delimits each class's segment in E.
        X_pack:        from pack_features(): W-bit word per feature.
        n_classes:     number of classes (== len(inc_per_class)).
        W:             batch width in bits (must match pack_features's W).

    Returns:
        best_class[0..W-1], best_score[0..W-1]: predicted class and its
        score for each of the W datapoint lanes (lanes beyond the real
        batch size, if padded, are computed but meaningless -- discard
        them).
    """
    FULL = (1 << W) - 1
    NEG_INF = -(1 << 62)

    best_score = [NEG_INF] * W
    best_class = [-1] * W
    ptr = 0

    for c in range(n_classes):
        y = [0] * W
        clause_mask = FULL
        have_clause = False
        P_prev = 0
        C_prev = 0

        for _ in range(inc_per_class[c]):
            word = E[ptr]
            ptr += 1
            P, C, offset, b0 = _unpack_word(word)

            if have_clause and C != C_prev:
                # Clause boundary: flush the just-closed clause into y[].
                sign = -1 if P_prev == 1 else 1
                for j in range(W):
                    if (clause_mask >> j) & 1:
                        y[j] += sign
                clause_mask = FULL
                have_clause = False

            if not have_clause:
                P_prev, C_prev = P, C
                have_clause = True

            lit = X_pack[offset]
            if b0 == 1:
                lit = (~lit) & FULL
            clause_mask &= lit

        if have_clause:
            # Flush the final clause of this class's segment.
            sign = -1 if P_prev == 1 else 1
            for j in range(W):
                if (clause_mask >> j) & 1:
                    y[j] += sign

        for j in range(W):
            if y[j] > best_score[j]:
                best_score[j] = y[j]
                best_class[j] = c

    return best_class, best_score


# ---------------------------------------------------------------------------
# Convenience: full-dataset inference in reference.predict()'s shape
# ---------------------------------------------------------------------------

def infer(
    tmir: TMIR, X: np.ndarray, W: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """Encode tmir once, then classify every row of X, batched W at a time.

    Returns (predictions, scores) in exactly the shape
    matador.inference.reference.predict() returns, so the two can be
    compared directly -- this is the correctness check this codec is
    verified against.
    """
    X = np.asarray(X, dtype=np.uint8)
    n_samples, n_features = X.shape
    n_classes = tmir.architecture.n_classes

    E, inc_per_class = encode(tmir)

    predictions = np.empty(n_samples, dtype=np.int64)
    scores = np.empty((n_samples, n_classes), dtype=np.int64)

    for start in range(0, n_samples, W):
        batch = X[start : start + W]
        n = batch.shape[0]
        X_pack = pack_features(batch, W)
        best_class, best_score = decode_and_infer(E, inc_per_class, X_pack, n_classes, W)
        for j in range(n):
            predictions[start + j] = best_class[j]
        # best_score above is only the WINNING class's score per datapoint
        # (Algorithm 12 never materialises the full per-class score vector
        # for a batch -- only running best_score/best_class). To return a
        # full (n_samples, n_classes) score matrix comparable with
        # reference.predict(), re-run decode class-by-class is wasteful;
        # instead recompute full per-class sums directly here for the
        # comparison harness's sake (RTL/hardware never needs this -- only
        # argmax's winner does).
        scores[start : start + n] = _full_class_scores(
            E, inc_per_class, X_pack, n_classes, W
        )[:n]

    return predictions, scores


def _full_class_scores(
    E: Sequence[int], inc_per_class: Sequence[int], X_pack: Sequence[int],
    n_classes: int, W: int,
) -> np.ndarray:
    """Same decode loop as decode_and_infer(), but keeping every class's
    score (not just the running argmax) -- needed only to compare against
    reference.predict()'s full score vector; the real accelerator only
    ever needs the argmax, which decode_and_infer() already computes."""
    FULL = (1 << W) - 1
    all_scores = np.zeros((W, n_classes), dtype=np.int64)
    ptr = 0

    for c in range(n_classes):
        y = [0] * W
        clause_mask = FULL
        have_clause = False
        P_prev = 0
        C_prev = 0

        for _ in range(inc_per_class[c]):
            word = E[ptr]
            ptr += 1
            P, C, offset, b0 = _unpack_word(word)

            if have_clause and C != C_prev:
                sign = -1 if P_prev == 1 else 1
                for j in range(W):
                    if (clause_mask >> j) & 1:
                        y[j] += sign
                clause_mask = FULL
                have_clause = False

            if not have_clause:
                P_prev, C_prev = P, C
                have_clause = True

            lit = X_pack[offset]
            if b0 == 1:
                lit = (~lit) & FULL
            clause_mask &= lit

        if have_clause:
            sign = -1 if P_prev == 1 else 1
            for j in range(W):
                if (clause_mask >> j) & 1:
                    y[j] += sign

        all_scores[:, c] = y

    return all_scores


# ---------------------------------------------------------------------------
# MUSTANG -- REDRESS translated to hardware (thesis Chapter 5.3, Figures
# 5.12/5.13)
# ---------------------------------------------------------------------------
#
# THE MUSTANG INSTRUCTION FORMAT (Figure 5.12.4, "Include Instruction
# Encoding")
#   MUSTANG folds a new "Class Change" bit (E) into the instruction word,
#   replacing classic REDRESS's separate Inc_per_class side-channel array
#   with an in-band marker -- one fewer packet type the processor has to
#   send the accelerator (see the thesis's "Programming for Real-time
#   Adaptability" discussion). E behaves exactly the way CC (renamed from
#   classic REDRESS's C) already does, one level up: CC flips on the first
#   include of a new CLAUSE, E flips on the first include of a new CLASS,
#   and both are detected the same way during decode -- by comparing the
#   incoming bit against a latched previous value, never by their absolute
#   value.
#
#   16-bit layout, with a caveat: Figure 5.12 labels the word "16'b" but its
#   own five segment widths (+/-:1, CC:1, E:1, Offset:13, L:1) sum to 17.
#   The offset field is 12 bits here, not the 13 the figure draws --
#   confirmed against the thesis's own prose, which frames the tradeoff as
#   "allocating an additional bit for Class Change appears to reduce the
#   range available for the offset" (i.e. the offset shrinks to make room,
#   the total word does not grow past the stated 16-bit MCU-agnostic
#   baseline):
#     bit15   +/-  clause polarity (0 = positive, 1 = negative)
#     bit14   CC   clause-change marker (flips on every new non-empty clause)
#     bit13   E    class-change marker (flips on every new non-empty class)
#     bits1-12     offset = feature index (12 bits, features up to 4096)
#     bit0         L = literal polarity (0 = positive literal, 1 = negated)
#
# WHAT'S DELIBERATELY UNCHANGED FROM classic REDRESS
#   The AND-accumulate decode loop, the clause-flush-on-boundary logic, and
#   the grouped (first-half-positive/second-half-negative) polarity
#   convention are identical to encode()/decode_and_infer() above -- MUSTANG
#   only changes how class segmentation is signalled, nothing about how a
#   clause or a class score is computed.
#
# A REAL LIMITATION THIS FORMAT INHERITS: EMPTY CLASSES
#   Because E only physically exists on transmitted words, and a fully
#   excluded class emits no words at all (same "empty clause emits nothing"
#   rule as encode()), a class with EVERY clause empty leaves no word behind
#   to carry its E-toggle. The next non-empty class's first word then shows
#   the SAME E value the empty class's predecessor ended on -- decode would
#   read that as "still the same class", silently merging the empty class
#   into whichever class follows it and permanently losing track of class
#   boundaries for the rest of the stream. Classic REDRESS has no such gap
#   (Inc_per_class explicitly lists 0 for an empty class, so decode still
#   visits it correctly). mustang_encode() detects this and raises
#   MustangEmptyClassError rather than silently producing a stream that
#   would decode incorrectly -- this is a real property of the format as
#   described, not an implementation shortcut.


MUSTANG_OFFSET_BITS = 12
MUSTANG_MAX_FEATURES = 1 << MUSTANG_OFFSET_BITS  # 4096


class MustangEmptyClassError(ValueError):
    """Raised by mustang_encode() when a model has a class with zero
    included literals across every clause -- see the module-level "A REAL
    LIMITATION..." note above for why the MUSTANG format can't represent
    this (classic REDRESS's encode()/decode_and_infer(), which carry class
    boundaries via a separate Inc_per_class array rather than an in-band
    bit, have no such restriction)."""


def _pack_mustang_word(P: int, CC: int, E: int, offset: int, L: int) -> int:
    if not (0 <= offset < MUSTANG_MAX_FEATURES):
        raise ValueError(
            f"feature offset {offset} does not fit MUSTANG's 12-bit offset "
            f"field (max {MUSTANG_MAX_FEATURES - 1}); the 16-bit MUSTANG "
            f"instruction cannot address more than {MUSTANG_MAX_FEATURES} "
            "features (use the classic REDRESS format, or a wider MUSTANG "
            "instruction, for larger models)"
        )
    return (P << 15) | (CC << 14) | (E << 13) | (offset << 1) | L


def _unpack_mustang_word(word: int) -> tuple[int, int, int, int, int]:
    P = (word >> 15) & 1
    CC = (word >> 14) & 1
    E = (word >> 13) & 1
    offset = (word >> 1) & ((1 << MUSTANG_OFFSET_BITS) - 1)
    L = word & 1
    return P, CC, E, offset, L


def mustang_encode(tmir: TMIR) -> list[int]:
    """Compress a TMIR model into a single MUSTANG instruction stream.

    Unlike encode(), there is no second return value -- class boundaries
    are carried in-band by the E bit (see module notes above), so there is
    nothing equivalent to Inc_per_class to return. The caller only needs
    n_classes (from tmir.architecture.n_classes, exactly as the real
    Instruction Header field would supply it) to decode this stream.

    Raises:
        MustangEmptyClassError: if any class has zero included literals
            across every one of its clauses (see module notes above).
    """
    tmir.to_includes()
    arch = tmir.architecture
    if arch.clause_organization != "per_class":
        raise NotImplementedError(
            "MUSTANG's Include-Encoding assumes per-class clause "
            f"organisation; got {arch.clause_organization!r}."
        )

    n_classes = arch.n_classes
    cpc = arch.n_clauses_per_class
    n_features = arch.n_features
    half = cpc // 2

    inc = np.asarray(tmir.representation.includes, dtype=bool)
    if inc.shape != (n_classes, cpc, 2 * n_features):
        raise ValueError(
            f"includes shape {inc.shape} != expected "
            f"{(n_classes, cpc, 2 * n_features)}"
        )

    E_words: list[int] = []
    E_bit = 0

    for c in range(n_classes):
        E_bit ^= 1  # toggles once per class, mirroring CC's once-per-clause flip
        CC = 0
        n_words_before = len(E_words)
        for n in range(cpc):
            P = 0 if n < half else 1
            clause = inc[c, n]
            first = True
            for f in range(n_features):
                if clause[f]:
                    if first:
                        CC ^= 1
                        first = False
                    E_words.append(_pack_mustang_word(P, CC, E_bit, f, 0))
                if clause[n_features + f]:
                    if first:
                        CC ^= 1
                        first = False
                    E_words.append(_pack_mustang_word(P, CC, E_bit, f, 1))
        if len(E_words) == n_words_before:
            raise MustangEmptyClassError(
                f"class {c} has zero included literals across all {cpc} "
                "clauses -- MUSTANG's in-band Class-Change bit cannot "
                "represent an empty class (see module notes above); use "
                "the classic REDRESS format (encode()/decode_and_infer()) "
                "for this model instead"
            )

    return E_words


def mustang_decode_and_infer(
    E_words: Sequence[int],
    n_classes: int,
    X_pack: Sequence[int],
    W: int,
) -> tuple[list[int], list[int]]:
    """Decode a MUSTANG instruction stream and classify one batch of up to
    W datapoints, in one pass -- the hardware dataflow of Figure 5.13,
    steps 4-6, transcribed directly: fetch+decode, literal-select AND into
    C1..C32 (clause_mask here), flush into Class Sums on a CC/E change,
    argmax.

    Args:
        E_words:   from mustang_encode() (or an RTL-loaded copy
                   reconstructed bit-for-bit the same way).
        n_classes: total class count -- supplied by the caller exactly as
                   the real Instruction Header field would (MUSTANG has no
                   Inc_per_class array to read this from).
        X_pack:    from pack_features(): W-bit word per feature.
        W:         batch width in bits.

    Returns:
        best_class[0..W-1], best_score[0..W-1] -- same shape as
        decode_and_infer().
    """
    FULL = (1 << W) - 1
    NEG_INF = -(1 << 62)

    best_score = [NEG_INF] * W
    best_class = [-1] * W

    y = [0] * W
    clause_mask = FULL
    have_clause = False
    have_class = False
    current_class = -1
    P_prev = 0
    CC_prev = 0
    E_prev = 0

    for word in E_words:
        P, CC, E, offset, L = _unpack_mustang_word(word)

        if have_class and E != E_prev:
            # New class: flush whatever clause was still open into y, then
            # flush the now-complete class into the running argmax.
            if have_clause:
                sign = -1 if P_prev == 1 else 1
                for j in range(W):
                    if (clause_mask >> j) & 1:
                        y[j] += sign
                clause_mask = FULL
                have_clause = False
            for j in range(W):
                if y[j] > best_score[j]:
                    best_score[j] = y[j]
                    best_class[j] = current_class
            y = [0] * W
            current_class += 1
            have_class = False  # forces E_prev to re-latch below
        elif not have_class:
            current_class = 0

        if have_clause and CC != CC_prev:
            # Clause boundary within the same class: flush just this clause.
            sign = -1 if P_prev == 1 else 1
            for j in range(W):
                if (clause_mask >> j) & 1:
                    y[j] += sign
            clause_mask = FULL
            have_clause = False

        if not have_clause:
            P_prev, CC_prev = P, CC
            have_clause = True
        if not have_class:
            E_prev = E
            have_class = True

        lit = X_pack[offset]
        if L == 1:
            lit = (~lit) & FULL
        clause_mask &= lit

    if have_clause:
        sign = -1 if P_prev == 1 else 1
        for j in range(W):
            if (clause_mask >> j) & 1:
                y[j] += sign
    if have_class:
        for j in range(W):
            if y[j] > best_score[j]:
                best_score[j] = y[j]
                best_class[j] = current_class

    return best_class, best_score


def mustang_infer(
    tmir: TMIR, X: np.ndarray, W: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """mustang_encode() + mustang_decode_and_infer(), in reference.predict()'s
    shape -- the correctness check this is verified against."""
    X = np.asarray(X, dtype=np.uint8)
    n_samples, n_features = X.shape
    n_classes = tmir.architecture.n_classes

    E_words = mustang_encode(tmir)

    predictions = np.empty(n_samples, dtype=np.int64)
    scores = np.empty((n_samples, n_classes), dtype=np.int64)

    for start in range(0, n_samples, W):
        batch = X[start : start + W]
        n = batch.shape[0]
        X_pack = pack_features(batch, W)
        best_class, best_score = mustang_decode_and_infer(E_words, n_classes, X_pack, W)
        for j in range(n):
            predictions[start + j] = best_class[j]
        scores[start : start + n] = _mustang_full_class_scores(
            E_words, n_classes, X_pack, W
        )[:n]

    return predictions, scores


def _mustang_full_class_scores(
    E_words: Sequence[int], n_classes: int, X_pack: Sequence[int], W: int,
) -> np.ndarray:
    """Same decode loop as mustang_decode_and_infer(), but keeping every
    class's score -- needed only to compare against reference.predict()'s
    full score vector; see _full_class_scores()'s docstring for why the
    real accelerator never needs this."""
    FULL = (1 << W) - 1
    all_scores = np.zeros((W, n_classes), dtype=np.int64)

    y = [0] * W
    clause_mask = FULL
    have_clause = False
    have_class = False
    current_class = -1
    P_prev = 0
    CC_prev = 0
    E_prev = 0

    for word in E_words:
        P, CC, E, offset, L = _unpack_mustang_word(word)

        if have_class and E != E_prev:
            if have_clause:
                sign = -1 if P_prev == 1 else 1
                for j in range(W):
                    if (clause_mask >> j) & 1:
                        y[j] += sign
                clause_mask = FULL
                have_clause = False
            all_scores[:, current_class] = y
            y = [0] * W
            current_class += 1
            have_class = False  # forces E_prev to re-latch below
        elif not have_class:
            current_class = 0

        if have_clause and CC != CC_prev:
            sign = -1 if P_prev == 1 else 1
            for j in range(W):
                if (clause_mask >> j) & 1:
                    y[j] += sign
            clause_mask = FULL
            have_clause = False

        if not have_clause:
            P_prev, CC_prev = P, CC
            have_clause = True
        if not have_class:
            E_prev = E
            have_class = True

        lit = X_pack[offset]
        if L == 1:
            lit = (~lit) & FULL
        clause_mask &= lit

    if have_clause:
        sign = -1 if P_prev == 1 else 1
        for j in range(W):
            if (clause_mask >> j) & 1:
                y[j] += sign
    if have_class:
        all_scores[:, current_class] = y

    return all_scores
