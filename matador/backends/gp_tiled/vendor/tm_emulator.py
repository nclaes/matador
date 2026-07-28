#!/usr/bin/env python3
# =============================================================================
# tm_emulator.py — bit-accurate golden model for the general-purpose TM
#                  accelerator (tm_accel_gp.v) and its AXI-Stream protocol
# =============================================================================
# WHAT THIS FILE PROVIDES
#   TMModel                 in-memory model (Include bits + geometry)
#   TMModel.infer()         cycle-exact inference semantics (unclamped, ties,
#                           empty-clause rule) — matches score_acc/argmax RTL
#   pack_tiles()            model -> list of TILE_WIDTH-bit ROM/BRAM rows
#   unpack_tiles()          inverse of pack_tiles (used for self-checks)
#   encode_load_packet()    model -> 32-bit AXIS words (header + payload)
#   encode_infer_packet()   feature vectors -> 32-bit AXIS words (batch)
#   expected_load_ack()     the single ack beat the RTL must emit after a load
#   extract_rom_model()     parse tm_accelerator.v -> TMModel of the frozen net
#   load_tmir()             TMIR adapter (STUB — needs the TMIR schema)
#   CLI: selftest | genvec  (see bottom of file)
#
# PROTOCOL (must match tm_accel_gp.v and README_GP.md)
#   Every s_axis packet begins with a header word:
#     word0  [31:24] MAGIC=0xA5   [23:16] CMD   [15:0] reserved (ignored)
#
#   CMD_LOAD (0x02) — reprogram the tile memory with a new model:
#     word1  [7:0]  n_classes     [15:8] clauses_per_class  [23:16] threshold
#                                        (VESTIGIAL: scores are unclamped,
#                                        kept only for wire compatibility)
#     word2  [7:0]  n_beats       [15:8] n_feat_slices      [23:16] n_clause_slices
#     word3  [15:0] n_clauses_total                         [31:16] n_tiles
#     payload: n_tiles rows x 64 words each; word w of a row carries row
#              bits [w*32 +: 32] (little-endian word order, LSW first).
#     TLAST on the final payload word.
#     RTL replies with one ack beat: [31:24]=0xA5 [23:16]=status [15:0]=info.
#
#   CMD_INFER (0x01) — run inference on one or more feature vectors:
#     payload: K frames of n_beats words each (batch), features packed
#              LSB-first exactly like the original tm_accelerator.
#     TLAST on the final word of the final frame.
#     RTL replies with one beat per frame: [31:0]={pad, class}; TLAST is
#     forwarded batch-style (1 only on the final frame's result).
#
#   STATUS / ERROR CODES (ack byte [23:16])
#     0x00 OK          info = n_tiles written
#     0xE0 ERR_MAGIC   header word0 magic mismatch
#     0xE1 ERR_CMD     unknown command byte
#     0xE2 ERR_RANGE   geometry field out of range / zero / odd cpc
#     0xE3 ERR_UNDERRUN TLAST arrived before the packet was complete
#     0xE4 ERR_OVERRUN  packet longer than the header promised
#     0xE5 ERR_NOCFG   CMD_INFER received before any successful CMD_LOAD
#
# HARDWARE SEMANTICS REPLICATED EXACTLY
#   * clause fires  iff  (Include mask nonzero) AND every Included literal == 1
#   * literal order: bit l (l < F) = feature l; bit F+f = NOT feature f
#   * scores: unclamped  (pol -> s+1) / (~pol -> s-1), raw vote sum, no
#             saturation. (An earlier version guarded each update against a
#             runtime `threshold`, clamping to [-T,+T]; removed as
#             order-dependent and unnecessary -- SCORE_WIDTH is sized to the
#             true worst-case magnitude, so the register never overflows.)
#   * clause g belongs to class g // cpc; positive iff (g % cpc) < cpc // 2
#   * argmax: linear scan, ties to the LOWER class index
#
# No third-party dependencies (plain-int bit twiddling), Python >= 3.8.
# =============================================================================

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# ── protocol constants (single source of truth for Python side) ─────────────
MAGIC          = 0xA5
CMD_INFER      = 0x01
CMD_LOAD       = 0x02
ST_OK          = 0x00
ERR_MAGIC      = 0xE0
ERR_CMD        = 0xE1
ERR_RANGE      = 0xE2
ERR_UNDERRUN   = 0xE3
ERR_OVERRUN    = 0xE4
ERR_NOCFG      = 0xE5

FEAT_SLICE     = 32           # fixed by the RTL tile geometry
CLAUSE_SLICE   = 32
AXIS_W         = 32
WORDS_PER_ROW  = None         # derived below
TILE_WIDTH     = CLAUSE_SLICE * 2 * FEAT_SLICE          # 2048
WORDS_PER_ROW  = TILE_WIDTH // AXIS_W                   # 64

# compile-time capacity of tm_accel_gp.v (keep in sync with the RTL params)
MAX_CLASSES        = 16
MAX_CLAUSES_TOTAL  = 256
MAX_FEAT_SLICES    = 16
MAX_CLAUSE_SLICES  = 8
MAX_TILES          = MAX_FEAT_SLICES * MAX_CLAUSE_SLICES   # 128
# threshold (header word1 byte [23:16]) is VESTIGIAL -- scores are unclamped
# (see score_acc_rt.v); this only bounds what the 8-bit wire field can hold.
MAX_THRESHOLD      = 255


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TMModel:
    """A trained vanilla TM: Include bits + geometry.

    include[g] is a plain int bitmask over 2*n_features literals for global
    clause g:  bit l (l < n_features)      -> positive literal of feature l
               bit n_features + f          -> negated  literal of feature f
    """
    n_features: int
    n_classes: int
    clauses_per_class: int
    threshold: int
    include: List[int] = field(default_factory=list)
    name: str = "model"

    # ── derived geometry (mirrors what the header words carry) ──────────────
    @property
    def n_clauses_total(self) -> int:
        return self.n_classes * self.clauses_per_class

    @property
    def n_feat_slices(self) -> int:
        return _ceil_div(self.n_features, FEAT_SLICE)

    @property
    def n_clause_slices(self) -> int:
        return _ceil_div(self.n_clauses_total, CLAUSE_SLICE)

    @property
    def n_tiles(self) -> int:
        return self.n_feat_slices * self.n_clause_slices

    @property
    def n_beats(self) -> int:
        return _ceil_div(self.n_features, AXIS_W)

    # ── validation (same checks the RTL performs on the header) ─────────────
    def validate(self) -> None:
        f = self
        problems = []
        if not (1 <= f.n_classes <= MAX_CLASSES):
            problems.append(f"n_classes={f.n_classes} outside 1..{MAX_CLASSES}")
        if f.clauses_per_class < 2 or f.clauses_per_class % 2:
            problems.append(f"clauses_per_class={f.clauses_per_class} must be even and >=2")
        if not (1 <= f.threshold <= MAX_THRESHOLD):
            problems.append(f"threshold={f.threshold} outside 1..{MAX_THRESHOLD}")
        if f.n_clauses_total > MAX_CLAUSES_TOTAL:
            problems.append(f"n_clauses_total={f.n_clauses_total} > {MAX_CLAUSES_TOTAL}")
        if f.n_feat_slices > MAX_FEAT_SLICES:
            problems.append(f"n_feat_slices={f.n_feat_slices} > {MAX_FEAT_SLICES}")
        if f.n_clause_slices > MAX_CLAUSE_SLICES:
            problems.append(f"n_clause_slices={f.n_clause_slices} > {MAX_CLAUSE_SLICES}")
        if len(f.include) != f.n_clauses_total:
            problems.append(f"include has {len(f.include)} rows, expected {f.n_clauses_total}")
        lit_mask = (1 << (2 * f.n_features)) - 1
        for g, inc in enumerate(f.include):
            if inc & ~lit_mask:
                problems.append(f"clause {g} has Include bits beyond 2*n_features")
                break
        if problems:
            raise ValueError("invalid TMModel: " + "; ".join(problems))

    # ── inference: global (reference) form ───────────────────────────────────
    def literals_of(self, features: int) -> int:
        """Pack a feature int (bit f = feature f) into the 2F literal vector."""
        fmask = (1 << self.n_features) - 1
        pos = features & fmask
        neg = (~features) & fmask
        return (neg << self.n_features) | pos

    def clause_active(self, g: int, literals: int) -> bool:
        inc = self.include[g]
        if inc == 0:
            return False                       # empty clause never fires
        return (literals & inc) == inc         # every Included literal is 1

    def infer(self, features: int) -> Tuple[int, List[int]]:
        """Returns (predicted_class, scores). Bit f of `features` = feature f."""
        lits = self.literals_of(features)
        half = self.clauses_per_class // 2
        scores = [0] * self.n_classes
        for g in range(self.n_clauses_total):
            if not self.clause_active(g, lits):
                continue
            cls = g // self.clauses_per_class
            pos = (g % self.clauses_per_class) < half
            if pos:
                scores[cls] += 1                # unclamped
            else:
                scores[cls] -= 1
        best, best_s = 0, scores[0]
        for k in range(1, self.n_classes):
            if scores[k] > best_s:             # strict > : ties keep lower index
                best, best_s = k, scores[k]
        return best, scores

    # ── inference: tiled form (mimics S_COMPUTE accumulation exactly) ───────
    def infer_tiled(self, features: int) -> Tuple[int, List[int]]:
        """Recomputes via the same per-tile AND/OR accumulation as the RTL.
        Used to cross-check pack_tiles() + the global rule against each other.
        """
        rows = self.pack_tiles()
        n_cs = self.n_clause_slices
        n_fs = self.n_feat_slices
        n_lanes = n_cs * CLAUSE_SLICE
        clause_pass = [True] * n_lanes
        clause_has = [False] * n_lanes
        fpad = features & ((1 << (n_fs * FEAT_SLICE)) - 1)
        for fs in range(n_fs):
            window = (fpad >> (fs * FEAT_SLICE)) & ((1 << FEAT_SLICE) - 1)
            lits = (((~window) & ((1 << FEAT_SLICE) - 1)) << FEAT_SLICE) | window
            for cs in range(n_cs):
                row = rows[fs * n_cs + cs]
                for k in range(CLAUSE_SLICE):
                    mask = (row >> (k * 2 * FEAT_SLICE)) & ((1 << (2 * FEAT_SLICE)) - 1)
                    has = mask != 0
                    tile_pass = has and ((lits & mask) == mask)
                    eff = tile_pass or (not has)          # empty tile: vacuous pass
                    g = cs * CLAUSE_SLICE + k
                    if fs == 0:
                        clause_pass[g] = eff
                        clause_has[g] = has
                    else:
                        clause_pass[g] = clause_pass[g] and eff
                        clause_has[g] = clause_has[g] or has
        half = self.clauses_per_class // 2
        scores = [0] * self.n_classes
        for g in range(self.n_clauses_total):
            if not (clause_pass[g] and clause_has[g]):
                continue
            cls = g // self.clauses_per_class
            pos = (g % self.clauses_per_class) < half
            if pos:
                scores[cls] += 1                # unclamped
            else:
                scores[cls] -= 1
        best, best_s = 0, scores[0]
        for k in range(1, self.n_classes):
            if scores[k] > best_s:
                best, best_s = k, scores[k]
        return best, scores

    # ── memory map: model <-> tile rows ──────────────────────────────────────
    def pack_tiles(self) -> List[int]:
        """Model -> list of n_tiles TILE_WIDTH-bit rows, address-ordered
        (row addr = fs * n_clause_slices + cs).  Within a row, clause lane k
        occupies bits [k*64 +: 64]: low 32 = Include(positive literals of the
        window), high 32 = Include(negated literals of the window).
        Lanes >= n_clauses_total and features >= n_features are zero-filled.
        """
        F = self.n_features
        rows: List[int] = []
        for fs in range(self.n_feat_slices):
            fbase = fs * FEAT_SLICE
            for cs in range(self.n_clause_slices):
                row = 0
                for k in range(CLAUSE_SLICE):
                    g = cs * CLAUSE_SLICE + k
                    if g >= self.n_clauses_total:
                        continue
                    inc = self.include[g]
                    pos_bits = 0
                    neg_bits = 0
                    top = min(FEAT_SLICE, F - fbase)
                    for l in range(max(0, top)):
                        f = fbase + l
                        if (inc >> f) & 1:
                            pos_bits |= 1 << l
                        if (inc >> (F + f)) & 1:
                            neg_bits |= 1 << l
                    row |= ((neg_bits << FEAT_SLICE) | pos_bits) << (k * 2 * FEAT_SLICE)
                rows.append(row)
        return rows

    @staticmethod
    def unpack_tiles(rows: Sequence[int], n_features: int, n_classes: int,
                     clauses_per_class: int, threshold: int,
                     name: str = "unpacked") -> "TMModel":
        """Inverse of pack_tiles(): rebuild a TMModel from address-ordered rows."""
        m = TMModel(n_features=n_features, n_classes=n_classes,
                    clauses_per_class=clauses_per_class, threshold=threshold,
                    include=[0] * (n_classes * clauses_per_class), name=name)
        F = n_features
        for fs in range(m.n_feat_slices):
            fbase = fs * FEAT_SLICE
            for cs in range(m.n_clause_slices):
                row = rows[fs * m.n_clause_slices + cs]
                for k in range(CLAUSE_SLICE):
                    g = cs * CLAUSE_SLICE + k
                    if g >= m.n_clauses_total:
                        continue
                    lane = (row >> (k * 2 * FEAT_SLICE)) & ((1 << (2 * FEAT_SLICE)) - 1)
                    pos_bits = lane & ((1 << FEAT_SLICE) - 1)
                    neg_bits = lane >> FEAT_SLICE
                    for l in range(FEAT_SLICE):
                        f = fbase + l
                        if f >= F:
                            break
                        if (pos_bits >> l) & 1:
                            m.include[g] |= 1 << f
                        if (neg_bits >> l) & 1:
                            m.include[g] |= 1 << (F + f)
        return m


# ─────────────────────────────────────────────────────────────────────────────
# AXIS packet encoding / expected responses
# ─────────────────────────────────────────────────────────────────────────────
def rows_to_words(rows: Iterable[int]) -> List[int]:
    """TILE_WIDTH-bit rows -> 32-bit words, LSW first within each row."""
    out: List[int] = []
    for row in rows:
        for w in range(WORDS_PER_ROW):
            out.append((row >> (w * AXIS_W)) & 0xFFFFFFFF)
    return out


def encode_load_packet(m: TMModel) -> List[int]:
    """CMD_LOAD packet (header + geometry + payload). TLAST = final word."""
    m.validate()
    w0 = (MAGIC << 24) | (CMD_LOAD << 16)
    w1 = (m.n_classes & 0xFF) | ((m.clauses_per_class & 0xFF) << 8) \
         | ((m.threshold & 0xFF) << 16)
    w2 = (m.n_beats & 0xFF) | ((m.n_feat_slices & 0xFF) << 8) \
         | ((m.n_clause_slices & 0xFF) << 16)
    w3 = (m.n_clauses_total & 0xFFFF) | ((m.n_tiles & 0xFFFF) << 16)
    return [w0, w1, w2, w3] + rows_to_words(m.pack_tiles())


def features_to_words(features: int, n_beats: int) -> List[int]:
    return [(features >> (b * AXIS_W)) & 0xFFFFFFFF for b in range(n_beats)]


def encode_infer_packet(m: TMModel, feature_vectors: Sequence[int]) -> List[int]:
    """CMD_INFER packet carrying a batch of frames. TLAST = final word."""
    words = [(MAGIC << 24) | (CMD_INFER << 16)]
    for fv in feature_vectors:
        words += features_to_words(fv, m.n_beats)
    return words


def expected_load_ack(m: TMModel) -> int:
    return (MAGIC << 24) | (ST_OK << 16) | (m.n_tiles & 0xFFFF)


def error_ack(code: int, info: int = 0) -> int:
    return (MAGIC << 24) | ((code & 0xFF) << 16) | (info & 0xFFFF)


# ─────────────────────────────────────────────────────────────────────────────
# Extraction of the frozen model from tm_accelerator.v
# ─────────────────────────────────────────────────────────────────────────────
_ROM_RE = re.compile(r"tile_rom\[\s*(\d+)\]\s*=\s*2048'h([0-9A-Fa-f]+)\s*;")


def extract_rom_model(rtl_path: Path, name: str = "frozen") -> TMModel:
    """Parse the initial-block ROM of the shipped tm_accelerator.v into a
    TMModel (geometry read from the frozen parameters of that design)."""
    text = Path(rtl_path).read_text()
    rows_by_addr = {int(m.group(1)): int(m.group(2), 16)
                    for m in _ROM_RE.finditer(text)}
    if len(rows_by_addr) != 112:
        raise ValueError(f"expected 112 ROM rows in {rtl_path}, found {len(rows_by_addr)}")
    rows = [rows_by_addr[a] for a in range(112)]
    m = TMModel.unpack_tiles(rows, n_features=512, n_classes=10,
                             clauses_per_class=20, threshold=8, name=name)
    # round-trip check: repacking must reproduce the ROM bit-for-bit
    if m.pack_tiles() != rows:
        raise AssertionError("pack_tiles(unpack_tiles(rom)) != rom — layout bug")
    return m


# ─────────────────────────────────────────────────────────────────────────────
# TMIR adapter (STUB)
# ─────────────────────────────────────────────────────────────────────────────
def load_tmir(path: Path) -> TMModel:
    """Adapter from your TMIR format to TMModel.

    TMIR.zip was not uploaded in this session, so the schema is unknown here.
    Once the TMIR format is available, implement the mapping to:
        n_features, n_classes, clauses_per_class, threshold,
        include[g] bitmask per global clause (positive literal f -> bit f,
        negated literal f -> bit n_features + f).
    Everything downstream (pack_tiles, encode_load_packet, infer) then works
    unchanged.
    """
    raise NotImplementedError(
        "TMIR schema unavailable in this session — upload TMIR.zip and fill in "
        "load_tmir(). See TMModel docstring for the target representation."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for generating tests
# ─────────────────────────────────────────────────────────────────────────────
def random_model(rng: random.Random, n_features: int, n_classes: int,
                 clauses_per_class: int, threshold: int,
                 include_density: float = 0.02, name: str = "rand") -> TMModel:
    """Random-but-plausible model: sparse Includes like a trained TM.
    Guarantees at least one non-empty clause overall so inference is non-trivial;
    individual clauses may legitimately be empty (exercises the empty rule)."""
    m = TMModel(n_features=n_features, n_classes=n_classes,
                clauses_per_class=clauses_per_class, threshold=threshold,
                include=[], name=name)
    for _ in range(n_classes * clauses_per_class):
        inc = 0
        for l in range(2 * n_features):
            if rng.random() < include_density:
                inc |= 1 << l
        m.include.append(inc)
    if all(i == 0 for i in m.include):
        m.include[0] = 1
    m.validate()
    return m


def parse_tb_system_vectors(tb_path: Path,
                            n_beats: int = 16) -> Tuple[List[int], List[int]]:
    """Pull tv_data/tv_exp out of the original tb_system.v.
    Returns (feature_ints, expected_classes)."""
    text = Path(tb_path).read_text()
    data = {int(m.group(1)): int(m.group(2), 16)
            for m in re.finditer(r"tv_data\[(\d+)\]\s*=\s*32'h([0-9A-Fa-f]+);", text)}
    exps = {int(m.group(1)): int(m.group(2))
            for m in re.finditer(r"tv_exp\[(\d+)\]\s*=\s*4'd(\d+);", text)}
    n_tests = len(exps)
    feats, expected = [], []
    for t in range(n_tests):
        fv = 0
        for b in range(n_beats):
            fv |= data[t * n_beats + b] << (b * AXIS_W)
        feats.append(fv)
        expected.append(exps[t])
    return feats, expected


# ─────────────────────────────────────────────────────────────────────────────
# memh emission for tb_system_gp.v
# ─────────────────────────────────────────────────────────────────────────────
def write_memh(path: Path, words: Sequence[int], lasts: Sequence[int],
               width_hex: int = 8) -> None:
    """One line per beat: 9 hex digits = {tlast[0], data[31:0]} packed as a
    33-bit value (tlast in bit 32)."""
    assert len(words) == len(lasts)
    with open(path, "w") as fh:
        for w, l in zip(words, lasts):
            fh.write(f"{(l << 32) | w:0{width_hex + 1}X}\n")


def stream_with_tlast(packets: Sequence[Sequence[int]]) -> Tuple[List[int], List[int]]:
    words, lasts = [], []
    for pkt in packets:
        for i, w in enumerate(pkt):
            words.append(w)
            lasts.append(1 if i == len(pkt) - 1 else 0)
    return words, lasts


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def cmd_selftest(args: argparse.Namespace) -> int:
    rtl = Path(args.rtl)
    tb = Path(args.tb)
    print("── extracting frozen model from", rtl.name)
    m = extract_rom_model(rtl)
    print(f"   {m.n_classes} classes, {m.clauses_per_class} clauses/class, "
          f"{m.n_features} features, T={m.threshold}, "
          f"{m.n_tiles} tiles ({m.n_feat_slices}x{m.n_clause_slices})")

    feats, exp = parse_tb_system_vectors(tb)
    print(f"── replaying {len(feats)} embedded tb_system vectors through the emulator")
    fails = 0
    for i, (fv, e) in enumerate(zip(feats, exp)):
        pred, scores = m.infer(fv)
        pred_t, scores_t = m.infer_tiled(fv)
        ok = (pred == e) and (pred_t == e) and (scores == scores_t)
        status = "ok " if ok else "FAIL"
        print(f"   test {i}: expected={e} global={pred} tiled={pred_t} "
              f"scores={scores}  [{status}]")
        fails += 0 if ok else 1

    print("── random-model cross-check (global vs tiled semantics)")
    rng = random.Random(0xC0FFEE)
    for trial in range(args.random_trials):
        nm = random_model(rng,
                          n_features=rng.choice([48, 96, 128, 512]),
                          n_classes=rng.choice([2, 3, 4, 10]),
                          clauses_per_class=rng.choice([4, 8, 20]),
                          threshold=rng.choice([2, 4, 8]),
                          include_density=rng.choice([0.0, 0.01, 0.05]))
        for _ in range(8):
            fv = rng.getrandbits(nm.n_features)
            a = nm.infer(fv)
            b = nm.infer_tiled(fv)
            if a != b:
                print(f"   FAIL trial {trial}: global {a} != tiled {b}")
                fails += 1
    print("   done")

    print("selftest:", "ALL PASSED" if fails == 0 else f"{fails} FAILURES")
    return 0 if fails == 0 else 1


def cmd_genvec(args: argparse.Namespace) -> int:
    """Emit stimulus/expected memh files consumed by tb_system_gp.v.

    Scenario (all one continuous s_axis stream, packet = TLAST-delimited):
      P0  LOAD  frozen model extracted from tm_accelerator.v
      P1  INFER batch: the 10 original tb_system vectors      (tlast batch)
      P2  LOAD  small random model (different geometry)       -> reprogram
      P3  INFER batch: random vectors on the small model
      P4  LOAD  frozen model again (prove full-size reload after small)
      P5  INFER single frame, one original vector
    Expected m_axis beats: ack, 10 results, ack, K results, ack, 1 result.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen = extract_rom_model(Path(args.rtl))
    feats, exp = parse_tb_system_vectors(Path(args.tb))

    rng = random.Random(args.seed)
    small = random_model(rng, n_features=args.small_features,
                         n_classes=args.small_classes,
                         clauses_per_class=args.small_cpc,
                         threshold=args.small_threshold,
                         include_density=0.06, name="small")
    small_vecs = [rng.getrandbits(small.n_features) for _ in range(args.small_frames)]

    packets = [
        encode_load_packet(frozen),
        encode_infer_packet(frozen, feats),
        encode_load_packet(small),
        encode_infer_packet(small, small_vecs),
        encode_load_packet(frozen),
        encode_infer_packet(frozen, [feats[0]]),
    ]
    s_words, s_lasts = stream_with_tlast(packets)

    exp_words: List[int] = []
    exp_lasts: List[int] = []

    def add(w: int, l: int) -> None:
        exp_words.append(w)
        exp_lasts.append(l)

    add(expected_load_ack(frozen), 1)
    for i, fv in enumerate(feats):
        pred, _ = frozen.infer(fv)
        if pred != exp[i]:
            print(f"internal error: emulator disagrees with tb_system on test {i}",
                  file=sys.stderr)
            return 1
        add(pred, 1 if i == len(feats) - 1 else 0)
    add(expected_load_ack(small), 1)
    for i, fv in enumerate(small_vecs):
        pred, _ = small.infer(fv)
        add(pred, 1 if i == len(small_vecs) - 1 else 0)
    add(expected_load_ack(frozen), 1)
    pred0, _ = frozen.infer(feats[0])
    add(pred0, 1)

    write_memh(out / "gp_stimulus.memh", s_words, s_lasts)
    write_memh(out / "gp_expected.memh", exp_words, exp_lasts)

    # standalone small-model packets for the directed error/recovery phases
    # of tb_system_gp.v (the TB mutates TLAST placement to inject underrun /
    # overrun, then replays the packets unmodified to prove recovery)
    sl_words, sl_lasts = stream_with_tlast([encode_load_packet(small)])
    write_memh(out / "gp_small_load.memh", sl_words, sl_lasts)
    si_words, si_lasts = stream_with_tlast(
        [encode_infer_packet(small, [small_vecs[0]])])
    write_memh(out / "gp_small_infer.memh", si_words, si_lasts)
    small_pred0, _ = small.infer(small_vecs[0])

    (out / "gp_sizes.vh").write_text(
        f"// generated by tm_emulator.py genvec (seed={args.seed})\n"
        f"localparam GP_N_STIM         = {len(s_words)};\n"
        f"localparam GP_N_EXP          = {len(exp_words)};\n"
        f"localparam GP_N_SMALL        = {len(sl_words)};\n"
        f"localparam GP_N_SMALL_INFER  = {len(si_words)};\n"
        f"localparam GP_SMALL_NBEATS   = {small.n_beats};\n"
        f"localparam [31:0] GP_SMALL_ACK       = 32'h{expected_load_ack(small):08X};\n"
        f"localparam [31:0] GP_SMALL_INFER_EXP = 32'h{small_pred0:08X};\n"
    )
    print(f"wrote {len(s_words)} stimulus beats, {len(exp_words)} expected beats -> {out}")
    print(f"small model: {small.n_classes} classes x {small.clauses_per_class} cpc, "
          f"{small.n_features} features, T={small.threshold}, "
          f"{small.n_tiles} tiles, {small.n_beats} beats/frame")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    st = sub.add_parser("selftest", help="validate emulator against the frozen RTL")
    st.add_argument("--rtl", default="../src/tm_accelerator.v")
    st.add_argument("--tb", default="../tb/tb_system.v")
    st.add_argument("--random-trials", type=int, default=40)
    st.set_defaults(func=cmd_selftest)

    gv = sub.add_parser("genvec", help="emit memh vectors for tb_system_gp.v")
    gv.add_argument("--rtl", default="../src/tm_accelerator.v")
    gv.add_argument("--tb", default="../tb/tb_system.v")
    gv.add_argument("--out", default="../sim/vectors")
    gv.add_argument("--seed", type=int, default=2026)
    gv.add_argument("--small-features", type=int, default=96)
    gv.add_argument("--small-classes", type=int, default=4)
    gv.add_argument("--small-cpc", type=int, default=8)
    gv.add_argument("--small-threshold", type=int, default=4)
    gv.add_argument("--small-frames", type=int, default=6)
    gv.set_defaults(func=cmd_genvec)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
