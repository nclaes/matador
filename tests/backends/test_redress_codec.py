"""Verification of matador.backends.redress.codec against the shared TMIR
reference (matador.inference.reference.predict) -- the AcceleratorSpec every
vanilla-TM backend must agree with (see matador/backends/base.py).

This only tests codec.py (the pure-Python Algorithm 11/12 transcription),
not RTL -- that comes later, once this is proven correct."""

from __future__ import annotations

import random

import numpy as np
import pytest

from matador.backends.redress import codec
from matador.inference.reference import predict
from matador.ir.tm_ir import Architecture, Hyperparameters, Representation, TMIR


def _make_tmir(n_features: int, n_classes: int, cpc: int, include_density: float, seed: int) -> TMIR:
    rng = np.random.default_rng(seed)
    inc = rng.random((n_classes, cpc, 2 * n_features)) < include_density
    return TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=2 * n_features,
            n_classes=n_classes, n_clauses_per_class=cpc,
            n_clauses_total=n_classes * cpc,
            clause_organization="per_class", threshold=0,
        ),
        hyperparameters=Hyperparameters(s=1.0),
        representation=Representation(includes=inc),
    )


def _random_X(n_samples: int, n_features: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((n_samples, n_features)) < 0.5).astype(np.uint8)


@pytest.mark.parametrize(
    "n_features,n_classes,cpc,include_density",
    [
        (8, 2, 4, 0.3),      # tiny, dense-ish
        (16, 3, 6, 0.1),     # small, sparse
        (32, 5, 10, 0.05),   # sparser, more classes
        (64, 4, 8, 0.02),    # realistic sparsity
        (13, 2, 4, 0.0),     # all-exclude clauses (every clause empty)
        (5, 2, 2, 1.0),      # fully dense (every literal included)
    ],
)
def test_encode_decode_matches_reference(n_features, n_classes, cpc, include_density):
    tmir = _make_tmir(n_features, n_classes, cpc, include_density, seed=n_features * 97 + cpc)
    X = _random_X(n_samples=50, n_features=n_features, seed=1)

    ref_pred, ref_scores = predict(tmir, X)
    codec_pred, codec_scores = codec.infer(tmir, X, W=32)

    np.testing.assert_array_equal(codec_pred, ref_pred)
    np.testing.assert_array_equal(codec_scores, ref_scores)


def test_encode_decode_matches_reference_many_random_trials():
    """Broader random sweep: many small random models, many random batches,
    varying batch width W -- a single failure anywhere should be very hard
    to hide from this."""
    rng = random.Random(0xC0FFEE)
    for trial in range(30):
        n_features = rng.choice([4, 8, 12, 20, 33])
        n_classes = rng.choice([2, 3, 5])
        cpc = rng.choice([2, 4, 6, 8]) * 1  # already even
        density = rng.choice([0.0, 0.02, 0.1, 0.3, 0.7])
        W = rng.choice([16, 32, 64])

        tmir = _make_tmir(n_features, n_classes, cpc, density, seed=trial)
        n_samples = rng.choice([1, 5, W, W + 3, 2 * W + 1])
        X = _random_X(n_samples, n_features, seed=1000 + trial)

        ref_pred, ref_scores = predict(tmir, X)
        codec_pred, codec_scores = codec.infer(tmir, X, W=W)

        assert list(codec_pred) == list(ref_pred), (
            f"trial {trial}: prediction mismatch "
            f"(n_features={n_features}, n_classes={n_classes}, cpc={cpc}, "
            f"density={density}, W={W}, n_samples={n_samples})"
        )
        assert codec_scores.tolist() == ref_scores.tolist(), (
            f"trial {trial}: score mismatch "
            f"(n_features={n_features}, n_classes={n_classes}, cpc={cpc}, "
            f"density={density}, W={W}, n_samples={n_samples})"
        )


def test_empty_clauses_emit_no_words():
    """A fully-excluded model must encode to zero words per class, and
    every prediction must be a tie broken to class 0 (argmax of all-zero
    scores) -- exercises the exact case the thesis calls out as rare but
    important (Chapter 4.3.1)."""
    tmir = _make_tmir(n_features=6, n_classes=3, cpc=4, include_density=0.0, seed=1)
    E, inc_per_class = codec.encode(tmir)
    assert E == []
    assert inc_per_class == [0, 0, 0]

    X = _random_X(10, 6, seed=2)
    pred, scores = codec.infer(tmir, X, W=32)
    assert (scores == 0).all()
    assert (pred == 0).all()  # argmax ties go to the lowest class index


def test_word_format_roundtrip():
    """Direct check of the 16-bit word packing against Algorithm 11's own
    bit layout (Figure 4.8(c)), independent of the encode/decode loop."""
    for P in (0, 1):
        for C in (0, 1):
            for offset in (0, 1, 4095, 8191):
                for b0 in (0, 1):
                    word = codec._pack_word(P, C, offset, b0)
                    assert 0 <= word <= 0xFFFF
                    assert codec._unpack_word(word) == (P, C, offset, b0)


def test_offset_out_of_range_rejected():
    with pytest.raises(ValueError):
        codec._pack_word(0, 0, codec.MAX_FEATURES, 0)  # one past the 13-bit limit


# ---------------------------------------------------------------------------
# MUSTANG (the hardware instruction format -- Class-Change bit, no separate
# Inc_per_class array)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_features,n_classes,cpc,include_density",
    [
        (8, 2, 4, 0.3),
        (16, 3, 6, 0.1),
        (32, 5, 10, 0.05),
        (64, 4, 8, 0.02),
        (5, 2, 2, 1.0),  # fully dense -- no empty clauses/classes possible
    ],
)
def test_mustang_matches_reference(n_features, n_classes, cpc, include_density):
    tmir = _make_tmir(n_features, n_classes, cpc, include_density, seed=n_features * 97 + cpc)
    X = _random_X(n_samples=50, n_features=n_features, seed=1)

    ref_pred, ref_scores = predict(tmir, X)
    mustang_pred, mustang_scores = codec.mustang_infer(tmir, X, W=32)

    np.testing.assert_array_equal(mustang_pred, ref_pred)
    np.testing.assert_array_equal(mustang_scores, ref_scores)


def test_mustang_matches_classic_redress():
    """Sanity cross-check independent of reference.predict(): both codecs
    describe the same model, so they must agree with EACH OTHER too."""
    tmir = _make_tmir(24, 4, 8, 0.15, seed=42)
    X = _random_X(n_samples=80, n_features=24, seed=7)

    classic_pred, classic_scores = codec.infer(tmir, X, W=32)
    mustang_pred, mustang_scores = codec.mustang_infer(tmir, X, W=32)

    np.testing.assert_array_equal(mustang_pred, classic_pred)
    np.testing.assert_array_equal(mustang_scores, classic_scores)


def _canonical_class_masks(inc: np.ndarray, n_classes: int, cpc: int) -> list:
    """Per-class SET of non-empty clause literal-masks -- the right
    equivalence for comparing two include arrays that may legitimately
    differ in which clause SLOT a given non-empty clause landed in (see
    mustang_decode_to_includes()'s own docstring for why exact positional
    equality is the wrong bar whenever a class has an interior empty
    clause)."""
    return [
        {tuple(inc[c, k].tolist()) for k in range(cpc) if inc[c, k].any()}
        for c in range(n_classes)
    ]


@pytest.mark.parametrize(
    "n_features,n_classes,cpc,density",
    [
        (8, 2, 4, 0.3),
        (10, 3, 6, 0.02),    # sparse enough to reliably produce interior empty clauses
        (16, 3, 6, 0.1),
        (32, 5, 10, 0.05),
        (5, 2, 2, 1.0),      # fully dense
    ],
)
def test_mustang_decode_to_includes_round_trips(n_features, n_classes, cpc, density):
    tmir = _make_tmir(n_features, n_classes, cpc, density, seed=n_features * 31 + cpc)
    try:
        E_words = codec.mustang_encode(tmir)
    except codec.MustangEmptyClassError:
        pytest.skip("random draw hit an empty class -- covered by a dedicated test")

    inc = np.asarray(tmir.representation.includes, dtype=bool)
    decoded = codec.mustang_decode_to_includes(E_words, n_classes, cpc, n_features)

    assert decoded.shape == inc.shape
    assert _canonical_class_masks(decoded, n_classes, cpc) == _canonical_class_masks(inc, n_classes, cpc)

    # Inference-equivalence, not just "same set of clause masks exists
    # somewhere" -- this is the bar that actually matters (a real bug was
    # caught here: an earlier version compacted transmitted clauses across
    # the WHOLE [0, cpc) slot range in transmission order, silently
    # corrupting polarity whenever a class had an empty clause in only one
    # of its two polarity halves, since polarity is inferred from
    # `slot < cpc // 2`, never stored per-slot).
    decoded_tmir = _make_tmir(n_features, n_classes, cpc, 0.0, seed=0)
    decoded_tmir.representation.includes = decoded
    X = _random_X(n_samples=40, n_features=n_features, seed=99)
    orig_pred, orig_scores = predict(tmir, X)
    dec_pred, dec_scores = predict(decoded_tmir, X)
    np.testing.assert_array_equal(dec_pred, orig_pred)
    np.testing.assert_array_equal(dec_scores, orig_scores)


def test_mustang_decode_to_includes_rejects_cpc_too_small():
    tmir = _make_tmir(8, 2, 6, 0.9, seed=1)  # dense -> every clause non-empty
    E_words = codec.mustang_encode(tmir)
    with pytest.raises(ValueError, match="cpc"):
        codec.mustang_decode_to_includes(E_words, n_classes=2, cpc=3, n_features=8)


def test_mustang_decode_to_includes_rejects_wrong_n_classes():
    tmir = _make_tmir(8, 3, 4, 0.3, seed=2)
    for c in range(3):
        tmir.representation.includes[c, 0, 0] = True  # guarantee non-empty
    E_words = codec.mustang_encode(tmir)
    with pytest.raises(ValueError, match="n_classes"):
        codec.mustang_decode_to_includes(E_words, n_classes=2, cpc=4, n_features=8)


def test_mustang_matches_reference_many_random_trials():
    """Broader random sweep, same spirit as the classic-REDRESS one above.
    Trials that happen to draw a fully-empty class are skipped (that's a
    documented MUSTANG format limitation, tested explicitly below) rather
    than counted as failures."""
    rng = random.Random(0xDECAF)
    ran = 0
    skipped_empty_class = 0
    trial = 0
    while ran < 30:
        trial += 1
        n_features = rng.choice([4, 8, 12, 20, 33])
        n_classes = rng.choice([2, 3, 5])
        cpc = rng.choice([2, 4, 6, 8])
        density = rng.choice([0.05, 0.1, 0.3, 0.7])  # no 0.0 -- guaranteed empty
        W = rng.choice([16, 32, 64])

        tmir = _make_tmir(n_features, n_classes, cpc, density, seed=trial)
        n_samples = rng.choice([1, 5, W, W + 3, 2 * W + 1])
        X = _random_X(n_samples, n_features, seed=2000 + trial)

        try:
            mustang_pred, mustang_scores = codec.mustang_infer(tmir, X, W=W)
        except codec.MustangEmptyClassError:
            skipped_empty_class += 1
            continue

        ref_pred, ref_scores = predict(tmir, X)
        assert list(mustang_pred) == list(ref_pred), (
            f"trial {trial}: prediction mismatch "
            f"(n_features={n_features}, n_classes={n_classes}, cpc={cpc}, "
            f"density={density}, W={W}, n_samples={n_samples})"
        )
        assert mustang_scores.tolist() == ref_scores.tolist(), (
            f"trial {trial}: score mismatch "
            f"(n_features={n_features}, n_classes={n_classes}, cpc={cpc}, "
            f"density={density}, W={W}, n_samples={n_samples})"
        )
        ran += 1

    assert ran == 30
    print(f"\n(skipped {skipped_empty_class} empty-class trial(s))")


def test_mustang_rejects_empty_class():
    """The documented MUSTANG format limitation (see codec.py's module
    notes): a class with zero included literals across every clause can't
    be represented, because its Class-Change toggle would never be
    physically transmitted."""
    tmir = _make_tmir(n_features=6, n_classes=3, cpc=4, include_density=0.0, seed=1)
    with pytest.raises(codec.MustangEmptyClassError, match="class 0"):
        codec.mustang_encode(tmir)


def test_mustang_rejects_one_empty_class_among_others():
    """A single empty class in the MIDDLE of an otherwise-normal model must
    still be rejected -- this is the actually-dangerous case (a fully empty
    MODEL is at least obviously degenerate; one empty class hiding among
    real ones is the case that would otherwise silently mis-decode)."""
    rng = np.random.default_rng(3)
    n_features, n_classes, cpc = 10, 3, 4
    inc = rng.random((n_classes, cpc, 2 * n_features)) < 0.3
    inc[1, :, :] = False  # class 1 (the middle class) has no includes at all
    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=2 * n_features,
            n_classes=n_classes, n_clauses_per_class=cpc,
            n_clauses_total=n_classes * cpc,
            clause_organization="per_class", threshold=0,
        ),
        hyperparameters=Hyperparameters(s=1.0),
        representation=Representation(includes=inc),
    )
    with pytest.raises(codec.MustangEmptyClassError, match="class 1"):
        codec.mustang_encode(tmir)


def test_mustang_word_format_roundtrip():
    for P in (0, 1):
        for CC in (0, 1):
            for E in (0, 1):
                for offset in (0, 1, 2047, 4095):
                    for L in (0, 1):
                        word = codec._pack_mustang_word(P, CC, E, offset, L)
                        assert 0 <= word <= 0xFFFF
                        assert codec._unpack_mustang_word(word) == (P, CC, E, offset, L)


def test_mustang_offset_out_of_range_rejected():
    with pytest.raises(ValueError):
        codec._pack_mustang_word(0, 0, 0, codec.MUSTANG_MAX_FEATURES, 0)  # one past the 12-bit limit
