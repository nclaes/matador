"""Stage 3: look-ahead decoding (v3, v4) and the hardened encoder, in software."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from matador.backends.redress import codec
from matador.backends.redress_lookahead import campaign
from matador.backends.redress_lookahead.codec_la import (
    MIN_CLASS_WORDS,
    encode,
    harden,
    split_stream,
)
from matador.backends.redress_lookahead.faithful import Geometry, decode_faithful, predict_batches
from matador.inference.reference import predict

G16 = Geometry(max_classes=16, class_width=4, feat_addr_w=8)
VARIANTS = ["v0", "v3", "v4", "v3d2", "v4d2"]


def _X(tmir):
    return np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)


def _one_literal_clauses(words) -> int:
    return sum(len(c) == 1 for cls in split_stream(words) for c in cls)


# ---------------------------------------------------------------------------
# Clean data: every variant is still the same classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("density,n_classes,seed", [
    (0.02, 4, 0), (0.05, 3, 1), (0.08, 6, 2), (0.3, 2, 3),
])
def test_fault_free_predictions_equal_reference(variant, density, n_classes, seed):
    tmir = campaign.make_synthetic_tmir(n_classes=n_classes, cpc=20, density=density,
                                        n_samples=64, seed=seed)
    X = _X(tmir)
    v = campaign.VARIANTS[variant]
    words = v.encode(tmir)
    ref, _ = predict(tmir, X)
    assert (predict_batches(words, n_classes, X, 32, G16, **v.mode) == ref).all()


def test_sparse_fixture_really_has_one_literal_clauses():
    tmir = campaign.make_synthetic_tmir(n_classes=4, cpc=20, density=0.02, seed=0)
    words = codec.mustang_encode(tmir)
    assert _one_literal_clauses(words) >= 10
    assert _one_literal_clauses(encode(tmir, e_depth=1, cc_lookahead=True)) == 0


@pytest.mark.parametrize("e_depth,cc", [(0, True), (1, False), (1, True), (2, True)])
def test_hardened_stream_decodes_to_the_same_includes(e_depth, cc):
    tmir = campaign.make_synthetic_tmir(n_classes=5, cpc=12, density=0.04, seed=8)
    arch = tmir.architecture
    args = (arch.n_classes, arch.n_clauses_per_class, arch.n_features)
    plain = codec.mustang_decode_to_includes(codec.mustang_encode(tmir), *args)
    hard = codec.mustang_decode_to_includes(encode(tmir, e_depth=e_depth, cc_lookahead=cc), *args)
    assert (plain == hard).all()


def test_hardened_streams_are_backward_compatible_with_as_built_decoder():
    """40 random models: a v4-hardened stream decoded by the AS-BUILT decoder
    (and by the codec) still gives reference.predict's answers."""
    mismatches = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n_classes = int(rng.integers(2, 7))
        tmir = campaign.make_synthetic_tmir(
            n_features=int(rng.integers(5, 30)), n_classes=n_classes,
            cpc=int(rng.integers(2, 16)), density=float(rng.uniform(0.02, 0.3)),
            n_samples=40, seed=seed)
        X = _X(tmir)
        ref, _ = predict(tmir, X)
        hard = encode(tmir, e_depth=1, cc_lookahead=True)
        mismatches += int((predict_batches(hard, n_classes, X, 32, G16) != ref).sum())
        X_pack = codec.pack_features(X[:32], 32)
        best, _ = codec.mustang_decode_and_infer(hard, n_classes, X_pack, 32)
        mismatches += int((np.array(best[:len(X[:32])]) != ref[:32]).sum())
    assert mismatches == 0


def test_harden_rejects_unknown_depth():
    with pytest.raises(ValueError, match="e_depth=3"):
        harden([0x6000], e_depth=3, cc_lookahead=False)


# ---------------------------------------------------------------------------
# Minimum class length, found by exhaustive search on toy streams
# ---------------------------------------------------------------------------

def _toy_stream(lengths):
    """E-only toy stream: class c has E = 1 for even c (class 0 has E = 1,
    as mustang_encode produces), one word per unit of length."""
    return [((c + 1) % 2) << 13 for c, n in enumerate(lengths) for _ in range(n)]


def _trace(words, e_depth):
    t: list[int] = []
    decode_faithful(words, 2, [0], 1, class_width=8, max_classes=256, feat_addr_w=8,
                    e_depth=e_depth, class_trace=t)
    return t


def _cascades(lengths, e_depth, flips_per_case):
    """True if some flip set moves a word whose class is >= 2 away from every
    flipped word's class (a boundary moving by a word or two is local; a whole
    later class changing index is a cascade). Also True if the clean stream
    itself decodes wrongly."""
    words = _toy_stream(lengths)
    clean = [c for c, n in enumerate(lengths) for _ in range(n)]
    if _trace(words, e_depth) != clean:
        return True
    n = len(words)
    for start in range(n - flips_per_case + 1):
        idx = range(start, start + flips_per_case)
        w = list(words)
        for j in idx:
            w[j] ^= 1 << 13
        t = _trace(w, e_depth)
        near = {clean[j] + d for j in idx for d in (-1, 0, 1)}
        if any(t[k] != clean[k] for k in range(n) if clean[k] not in near):
            return True
    return False


def _min_class_length(e_depth, flips_per_case, k_classes=4, max_len=9):
    """Smallest L such that no stream whose classes all have L..L+2 words
    cascades. None if there is none up to max_len."""
    for L in range(1, max_len + 1):
        if not any(_cascades(ls, e_depth, flips_per_case)
                   for ls in itertools.product(range(L, L + 3), repeat=k_classes)):
            return L
    return None


def test_minimum_class_length_table(capsys):
    table = {(d, f): _min_class_length(d, f) for d in (0, 1, 2) for f in (1, 2)}
    with capsys.disabled():
        print("\n[min class length]  e_depth | single flip | two adjacent flips")
        for d in (0, 1, 2):
            single, adjacent = str(table[(d, 1)]), str(table[(d, 2)])
            print(f"[min class length]  {d:>7} | {single:>11} | {adjacent:>18}")
    assert table[(0, 1)] is None            # as built: interior E flips always cascade
    assert table[(1, 1)] == MIN_CLASS_WORDS[1] == 4
    assert table[(2, 1)] == 6
    assert table[(1, 2)] is None            # one-word look-ahead can't survive adjacent flips
    assert table[(2, 2)] == MIN_CLASS_WORDS[2] == 7  # depth 2 pads for adjacent flips


def test_three_word_class_fails_with_one_word_lookahead():
    """The concrete counterexample behind MIN_CLASS_WORDS[1] == 4."""
    assert _cascades([4, 3, 4, 4], e_depth=1, flips_per_case=1)
    assert not _cascades([4, 4, 4, 4], e_depth=1, flips_per_case=1)


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def test_campaign_lookahead_cuts_e_sensitivity():
    tmir = campaign.make_synthetic_tmir(n_classes=4, cpc=12, density=0.06, n_samples=32, seed=4)
    X = _X(tmir)
    res = {}
    for name in ("v0", "v3", "v4"):
        v = campaign.VARIANTS[name]
        words = v.encode(tmir)
        res[name] = campaign.summarise(name, campaign.run_campaign(words, 4, X, 32, G16, v.mode),
                                       len(X), len(codec.mustang_encode(tmir)))
    assert res["v3"].by_field["E"] < res["v0"].by_field["E"] / 10
    assert res["v4"].by_field["E"] < res["v0"].by_field["E"] / 10
    assert res["v3"].nxp < res["v0"].nxp and res["v4"].nxp < res["v0"].nxp
    # NOT asserted: v4 beating v3 on CC. On this very sparse model it doesn't
    # (see STAGE_3 README): the CC rule only protects interior clause words.


def test_cc_lookahead_protects_interior_clause_words():
    """Under v4 a CC flip on a clause's 3rd..second-to-last word changes
    nothing; as built, the same flips do change outputs. The 2nd word is NOT
    protected: flipping it makes the look-ahead reject the real boundary at
    the 1st word, so the boundary moves by two words (the CC analogue of the
    4-word minimum class length)."""
    tmir = campaign.make_synthetic_tmir(n_classes=4, cpc=12, density=0.2, n_samples=32, seed=4)
    X = _X(tmir)
    for name, expect_zero in (("v4", True), ("v0", False)):
        v = campaign.VARIANTS[name]
        words = v.encode(tmir)
        p = campaign.run_campaign(words, 4, X, 32, G16, v.mode)
        interior, pos = [], 0
        for clauses in split_stream(words):
            for c in clauses:
                interior += range(pos + 2, pos + len(c) - 1)
                pos += len(c)
        assert len(interior) > 20
        cc_interior = p[interior, 14].sum()
        assert (cc_interior == 0) if expect_zero else (cc_interior > 0), (name, cc_interior)
