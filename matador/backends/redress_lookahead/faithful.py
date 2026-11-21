"""A software model of redress_accel.v that stays bit-exact under faults.

codec.mustang_decode_and_infer() is the *specification*: it is correct on clean
streams, but it is not the RTL. Once bits are flipped the two diverge (see
docs/lookahead_exercise/STAGE_2_faithful.md). decode_faithful() instead copies
the RTL's own datapath, register by register, so it predicts what the hardware
does with a corrupted Instruction Memory, not what the format intends.

Every step below names the redress_accel.v line it mirrors (line numbers as of
la-stage-1).

Conventions that the RTL leaves undefined:
  * Feature FIFO entries at or beyond n_features are never written for this
    model, so they read X in simulation. decode_faithful treats them as 0.
    Corrupted-RTL replays must therefore skip offset flips that land there.
  * Score registers are SCORE_WIDTH-bit signed counters. They cannot overflow:
    |score| <= number of flushes <= n_words + 1 <= max_includes + 1, and
    SCORE_WIDTH = clog2(max_includes + 1) + 2 (rtl.py _capacity_params). So
    plain Python ints are exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Geometry:
    """The compile-time RTL parameters decode_faithful depends on."""

    max_classes: int
    class_width: int
    feat_addr_w: int

    @classmethod
    def from_config(cls, config) -> Geometry:
        from matador.backends.redress.rtl import _clog2

        return cls(
            max_classes=config.max_classes,
            class_width=_clog2(config.max_classes),
            feat_addr_w=_clog2(config.max_features),
        )


def _field(word: int) -> tuple[int, int, int, int, int]:
    # redress_accel.v:200-204 -- P_cur, CC_cur, E_cur, offset_cur, L_cur
    return (word >> 15) & 1, (word >> 14) & 1, (word >> 13) & 1, (word >> 1) & 0xFFF, word & 1


def decode_faithful(
    words: Sequence[int],
    n_classes: int,
    X_pack: Sequence[int],
    W: int,
    *,
    class_width: int,
    max_classes: int,
    feat_addr_w: int,
    e_depth: int = 0,
    cc_lookahead: bool = False,
    e_init: int | None = None,
    class_trace: list[int] | None = None,
) -> tuple[list[int], np.ndarray]:
    """Run one batch through the decoder exactly as redress_accel.v does
    (e_depth=0, cc_lookahead=False) or as redress_la_accel.v does.

    Args:
        words:       Instruction Memory contents (possibly corrupted).
        n_classes:   the Instruction header's n_classes field.
        X_pack:      Feature FIFO contents: one W-bit int per loaded feature.
        W:           BATCH_WIDTH.
        class_width, max_classes, feat_addr_w: RTL parameters (see Geometry).
        e_depth:     E look-ahead depth. 0 = as built. d > 0 accepts an E change
                     only if the next d words exist and all carry the new E;
                     a rejected change leaves the E latch untouched.
        cc_lookahead: the same rule for CC, one word deep.
        e_init:      E latch value at batch start when e_depth > 0 (default 1,
                     class 0's E). The latch then counts as already set, so
                     word 0 is checked too. Ignored when e_depth == 0.
        class_trace: if given, the (wrapped) class index each word is
                     accumulated into is appended to it.

    Returns:
        (pred_class per lane, scores array of shape (W, max_classes)).
    """
    if e_depth < 0:
        raise ValueError(f"e_depth must be >= 0, got {e_depth}")
    n_words = len(words)

    def e_bit(w: int) -> int:
        return (w >> 13) & 1

    def cc_bit(w: int) -> int:
        return (w >> 14) & 1

    full = (1 << W) - 1
    class_mod = 1 << class_width
    feat_mask = (1 << feat_addr_w) - 1
    n_feat_loaded = len(X_pack)

    # Batch start, S_LOAD_FEAT TLAST (redress_accel.v:336-340). With E
    # look-ahead the E latch starts at a known value and counts as set.
    have_clause = False
    have_class = e_depth > 0
    clause_mask = full
    current_class = 0
    P_prev = CC_prev = 0  # X in the RTL; never read before the first latch
    E_prev = (1 if e_init is None else e_init) if e_depth > 0 else 0
    # score_acc_rt is cleared at batch start (acc_clear). Votes are recorded as
    # (class, sign, mask) and summed per lane at the end.
    votes: list[tuple[int, int, int]] = []

    def flush() -> None:
        # score_acc_rt.v:60-65: valid & active -> scores[cls] += +/-1.
        # acc_polarity = ~P_prev (redress_accel.v:234); acc_cls = current_class.
        # A write to scores[cls] with cls >= MAX_CLASSES has no target: dropped.
        if current_class < max_classes:
            votes.append((current_class, -1 if P_prev else 1, clause_mask))

    for i, word in enumerate(words):
        # S_INFER_DECODE (redress_accel.v:356-361): latch, issue feature read.
        # feat_raddr is FEAT_ADDR_W bits wide, so the 12-bit offset truncates.
        P, CC, E, offset, L = _field(word)
        addr = offset & feat_mask
        feat = X_pack[addr] if addr < n_feat_loaded else 0  # see module note
        # redress_accel.v:206
        selected = (~feat & full) if L else feat

        # S_INFER_ACCUM (redress_accel.v:371-411)
        class_boundary = have_class and E_prev != E        # :211
        if class_boundary and e_depth > 0:
            # Look-ahead: the next e_depth words must exist and agree. A
            # rejected change is treated as a flipped bit: no boundary, and the
            # latch below keeps its old value.
            class_boundary = i + e_depth < n_words and all(
                e_bit(words[i + k]) == E for k in range(1, e_depth + 1)
            )
        clause_boundary = have_clause and CC_prev != CC    # :212
        if clause_boundary and cc_lookahead:
            clause_boundary = i + 1 < n_words and cc_bit(words[i + 1]) == CC
        flush_this_cycle = class_boundary or clause_boundary  # :213

        if flush_this_cycle:  # flush_pulse uses the OLD clause_mask and class
            flush()
            clause_mask = selected                        # :385
        else:
            clause_mask &= selected                       # :387

        if class_boundary:                                # :390-391, wraps
            current_class = (current_class + 1) % class_mod
        if not have_class or class_boundary:              # :393-396
            E_prev = E
            have_class = True
        if not have_clause or flush_this_cycle:           # :397-401
            P_prev, CC_prev = P, CC
            have_clause = True
        if class_trace is not None:
            class_trace.append(current_class)

    # S_INFER_FINAL (redress_accel.v:413-423): unconditional final flush.
    if words:
        flush()

    scores = np.zeros((W, max_classes), dtype=np.int64)
    if votes:
        cls_arr = np.fromiter((v[0] for v in votes), dtype=np.int64, count=len(votes))
        signed = np.fromiter((v[1] for v in votes), dtype=np.int64, count=len(votes))
        masks = np.fromiter((v[2] for v in votes), dtype=np.uint64, count=len(votes))
        lanes = np.arange(W, dtype=np.uint64)
        bits = ((masks[:, None] >> lanes) & np.uint64(1)).astype(np.int64)  # (votes, W)
        for cls in np.unique(cls_arr):
            sel = cls_arr == cls
            scores[:, cls] = (signed[sel, None] * bits[sel]).sum(axis=0)

    # argmax_rt.v:48, fed argmax_n_classes (redress_accel.v:250-251): the 8-bit
    # header field zero-extended, then sliced to CLASS_WIDTH+1 bits. Class 0 is
    # always the starting candidate and the loop uses strict >, so ties go to
    # the lowest index -- exactly np.argmax's "first maximum" rule once the
    # classes argmax ignores (k >= n_classes, except class 0) are masked out.
    n_eff = (n_classes & 0xFF) & ((1 << (class_width + 1)) - 1)
    candidates = scores.copy()
    candidates[:, max(n_eff, 1):] = np.iinfo(np.int64).min
    pred = candidates.argmax(axis=1).tolist()
    return pred, scores


def predict_batches(words, n_classes, X, W, geometry: Geometry, **mode) -> np.ndarray:
    """decode_faithful over a whole (n_samples, n_features) matrix, in W-wide
    batches, returning one prediction per real sample."""
    from matador.backends.redress.codec import pack_features

    X = np.asarray(X, dtype=np.uint8)
    out = np.empty(X.shape[0], dtype=np.int64)
    for start in range(0, X.shape[0], W):
        batch = X[start:start + W]
        pred, _ = decode_faithful(
            words, n_classes, pack_features(batch, W), W,
            class_width=geometry.class_width, max_classes=geometry.max_classes,
            feat_addr_w=geometry.feat_addr_w, **mode,
        )
        out[start:start + batch.shape[0]] = pred[:batch.shape[0]]
    return out
