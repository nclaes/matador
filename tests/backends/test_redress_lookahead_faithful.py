"""Stage 2: decode_faithful() matches redress_accel.v, including under faults.

Three layers of evidence:
  1. Clean streams: decode_faithful == codec.mustang_decode_and_infer == reference.predict.
  2. Single flips: where decode_faithful and the codec disagree, and why.
  3. Corrupted-model replay: the as-built RTL agrees with decode_faithful on
     flipped streams (iverilog).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from matador.backends.redress import codec
from matador.backends.redress.config import RedressAcceleratorConfig
from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead import campaign
from matador.backends.redress_lookahead.faithful import Geometry, decode_faithful, predict_batches
from matador.backends.redress_lookahead.replay import (
    BIT_FIELD,
    build_replay_bundle,
    flip,
    replay_safe,
    run_iverilog,
)
from matador.inference.reference import predict

_HAVE_IVERILOG = shutil.which("iverilog") is not None


def _geometry(max_classes=16, max_features=256):
    from matador.backends.redress.rtl import _clog2

    return Geometry(max_classes=max_classes, class_width=_clog2(max_classes),
                    feat_addr_w=_clog2(max_features))


def _codec_predict(words, n_classes, X, W):
    out = np.empty(len(X), dtype=np.int64)
    for start in range(0, len(X), W):
        batch = X[start:start + W]
        best, _ = codec.mustang_decode_and_infer(words, n_classes, codec.pack_features(batch, W), W)
        out[start:start + len(batch)] = best[:len(batch)]
    return out


def _class_index_reached(words) -> int:
    """Highest (unwrapped) class index the as-built decoder reaches: every E
    change between consecutive words is a boundary, because the latch always
    takes the new value."""
    return sum(((a >> 13) ^ (b >> 13)) & 1 for a, b in zip(words, words[1:], strict=False))


# ---------------------------------------------------------------------------
# 1. Clean streams
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_classes,max_classes,density,seed", [
    (2, 16, 0.08, 0), (3, 4, 0.05, 1), (4, 4, 0.08, 2), (16, 16, 0.05, 3), (5, 8, 0.2, 4),
])
def test_clean_stream_matches_codec_and_reference(n_classes, max_classes, density, seed):
    tmir = campaign.make_synthetic_tmir(n_classes=n_classes, cpc=12, density=density,
                                        n_samples=70, seed=seed)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = codec.mustang_encode(tmir)
    ref, _ = predict(tmir, X)

    faithful = predict_batches(words, n_classes, X, 32, _geometry(max_classes))
    assert (faithful == ref).all()
    assert (_codec_predict(words, n_classes, X, 32) == ref).all()
    assert len(set(ref.tolist())) > 1  # a real check, not all class 0


def test_decode_faithful_returns_full_batch_scores():
    tmir = campaign.make_synthetic_tmir(n_classes=3, n_samples=10, seed=5)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = codec.mustang_encode(tmir)
    g = _geometry(4)
    pred, scores = decode_faithful(words, 3, codec.pack_features(X, 32), 32,
                                   class_width=g.class_width, max_classes=g.max_classes,
                                   feat_addr_w=g.feat_addr_w)
    _, ref_scores = predict(tmir, X)
    assert scores.shape == (32, 4)
    assert (scores[:10, :3] == ref_scores).all()
    assert (scores[:, 3] == 0).all()  # class 3 exists in hardware but is never voted
    assert len(pred) == 32


# ---------------------------------------------------------------------------
# 2. Single flips: where the codec stops describing the hardware
# ---------------------------------------------------------------------------

def test_single_flips_codec_vs_faithful_disagreements_are_explained(capsys):
    """decode_faithful and the codec must agree on every flip that leaves the
    decoder visiting exactly classes 0..n_classes-1. Every disagreement must
    come from a flip that pushes the class index past n_classes-1 (codec
    scores those extra segments; RTL wraps, drops or ignores them) or stops
    short of it (RTL still offers the unvisited classes to argmax at score 0)."""
    n_classes, max_classes = 4, 4
    tmir = campaign.make_synthetic_tmir(n_classes=n_classes, seed=0)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = codec.mustang_encode(tmir)
    g = _geometry(max_classes)

    counts = {"flips": 0, "agree": 0, "past_end": 0, "short": 0}
    for i in range(len(words)):
        for b in range(16):
            w = flip(words, i, b)
            if not replay_safe(w[i], 15, g.feat_addr_w) or ((w[i] >> 1) & 0xFFF) >= 15:
                continue  # the codec indexes X_pack[offset] directly
            counts["flips"] += 1
            faithful = predict_batches(w, n_classes, X, 32, g)
            same = (faithful == _codec_predict(w, n_classes, X, 32)).all()
            reached = _class_index_reached(w)
            if reached == n_classes - 1:
                assert same, (i, b, BIT_FIELD[b])
                counts["agree"] += 1
            elif not same:
                counts["past_end" if reached >= n_classes else "short"] += 1
    with capsys.disabled():
        print(f"\n[codec vs faithful] {counts}")
    assert counts["past_end"] > 0  # the test model really does exercise the difference


# ---------------------------------------------------------------------------
# 3. Corrupted-model replay through the as-built RTL
# ---------------------------------------------------------------------------

def _replay_cfg(tmp_path, tmir, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.yaml"
    tmir.to_yaml(path)
    payload = dict(model_path=str(path), output_dir=str(tmp_path / "bundle"), target_fpga="xc7z020",
                   max_includes=4096, max_features=256, max_classes=4, batch_width=32)
    payload.update(overrides)
    return RedressAcceleratorConfig.model_validate(payload)


def _pick_replay_flips(words, g, n_features, per_field=4, seed=7):
    """per_field flips of each field, all replay-safe, plus two flips where
    the codec and decode_faithful disagree."""
    rng = np.random.default_rng(seed)
    picks = []
    for field_bits in ([15], [14], [13], [0], list(range(1, 13))):
        found = 0
        for _ in range(500):
            i = int(rng.integers(len(words)))
            b = int(rng.choice(field_bits))
            if replay_safe(words[i] ^ (1 << b), n_features, g.feat_addr_w):
                picks.append((i, b))
                found += 1
                if found == per_field:
                    break
    return picks


@pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")
def test_corrupted_replay_as_built_rtl_matches_faithful(tmp_path):
    n_classes = 4
    tmir = campaign.make_synthetic_tmir(n_classes=n_classes, seed=0)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = codec.mustang_encode(tmir)
    g = _geometry(4)
    picks = _pick_replay_flips(words, g, 15)

    # Add flips where the codec is wrong about the hardware, so the replay
    # genuinely distinguishes the two software models.
    codec_wrong = []
    for i in range(len(words)):
        w = flip(words, i, 13)
        if (predict_batches(w, n_classes, X, 32, g) != _codec_predict(w, n_classes, X, 32)).any():
            codec_wrong.append((i, 13))
        if len(codec_wrong) == 2:
            break
    assert len(codec_wrong) == 2
    picks += codec_wrong

    failures = []
    for k, (i, b) in enumerate(picks):
        cfg = _replay_cfg(tmp_path / f"r{k}", tmir)
        artifacts = build_replay_bundle(tmir, flip(words, i, b), cfg)
        ok, out = run_iverilog(artifacts)
        if not ok:
            failures.append((i, b, out[-400:]))
    assert len(picks) >= 20
    assert not failures, failures

    # Negative control: the same harness with the CODEC's expectations must
    # fail on a flip where the codec is wrong -- proof the replay can fail.
    i, b = codec_wrong[0]
    corrupted = flip(words, i, b)

    class CodecExpectations(RedressBackend):
        def encode_stream(self, tmir, config):
            return list(corrupted)

    cfg = _replay_cfg(tmp_path / "control", tmir)
    ok, out = run_iverilog(CodecExpectations().generate(tmir, cfg))
    assert not ok and "FAIL out[" in out, out[-400:]


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------

def test_campaign_e_dominates_as_built():
    tmir = campaign.make_synthetic_tmir(n_classes=4, cpc=10, density=0.1, n_samples=32, seed=2)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = codec.mustang_encode(tmir)
    p = campaign.run_campaign(words, 4, X, 32, _geometry(16))
    res = campaign.summarise("v0", p, len(X), len(words))
    assert res.nxp > 0
    # E is one bit per word yet the largest field in total, and per bit it is
    # several times more sensitive than any other field.
    assert res.by_field["E"] == max(res.by_field.values())
    per_bit = {f: res.by_field[f] / len(campaign.FIELD_BITS[f]) for f in campaign.FIELDS}
    assert all(per_bit["E"] > 3 * v for f, v in per_bit.items() if f != "E")
