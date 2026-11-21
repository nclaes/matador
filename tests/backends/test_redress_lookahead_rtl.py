"""Stage 4: redress_la_accel.v against decode_faithful, clean and corrupted."""

from __future__ import annotations

import re
import shutil

import numpy as np
import pytest

from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead import campaign
from matador.backends.redress_lookahead.codec_la import split_stream
from matador.backends.redress_lookahead.config import RedressLookaheadConfig
from matador.backends.redress_lookahead.faithful import Geometry, predict_batches
from matador.backends.redress_lookahead.replay import (
    build_replay_bundle,
    flip,
    replay_safe,
    run_iverilog,
)
from matador.backends.redress_lookahead.rtl import RedressLookaheadBackend
from matador.inference.reference import predict

_HAVE_IVERILOG = shutil.which("iverilog") is not None
pytestmark = pytest.mark.skipif(not _HAVE_IVERILOG, reason="iverilog not installed")

COMBOS = [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)]


def _cfg(tmp_path, tmir, e_depth=1, cc=True, **overrides) -> RedressLookaheadConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.yaml"
    tmir.to_yaml(path)
    payload = dict(model_path=str(path), output_dir=str(tmp_path / "out"), target_fpga="xc7z020",
                   max_includes=4096, max_features=256, max_classes=4, batch_width=32,
                   e_lookahead_depth=e_depth, cc_lookahead=cc)
    payload.update(overrides)
    return RedressLookaheadConfig.model_validate(payload)


def _model(seed=0, n_samples=70):
    return campaign.make_synthetic_tmir(n_classes=4, cpc=12, density=0.08, n_samples=n_samples,
                                        seed=seed)


def _finish_time(output: str) -> int:
    m = re.search(r"\$finish called at (\d+)", output)
    assert m, output[-300:]
    return int(m.group(1))


@pytest.mark.parametrize("e_depth,cc", COMBOS, ids=lambda v: str(v))
def test_clean_simulation_all_parameter_combinations(tmp_path, e_depth, cc):
    tmir = _model()
    cfg = _cfg(tmp_path, tmir, e_depth, cc)
    backend = RedressLookaheadBackend()
    artifacts = backend.generate(tmir, cfg)

    top = (artifacts.rtl_dir / "src" / "redress_la_accel.v").read_text()
    assert f"parameter E_LA_DEPTH      = {e_depth}" in top
    assert f"parameter CC_LA           = {int(cc)}" in top
    ok, out = run_iverilog(artifacts)
    assert ok, out[-600:]
    assert "ALL 96 TESTS PASSED" in out  # 70 vectors -> 3 batches of 32

    # The testbench checks decode_faithful; tie that to the spec too.
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    words = backend.encode_stream(tmir, cfg)
    ref, _ = predict(tmir, X)
    pred = predict_batches(words, 4, X, 32, Geometry.from_config(cfg), **cfg.decoder_mode)
    assert (pred == ref).all()


def _replay_picks(words, n_features, feat_addr_w, seed, per_field=4):
    rng = np.random.default_rng(seed)
    picks = []
    for bits in ([15], [14], [13], [0], list(range(1, 13))):
        found = 0
        while found < per_field:
            i, b = int(rng.integers(len(words))), int(rng.choice(bits))
            if replay_safe(words[i] ^ (1 << b), n_features, feat_addr_w):
                picks.append((i, b))
                found += 1
    return picks


@pytest.mark.parametrize("e_depth,cc", [(1, True), (1, False), (2, True)], ids=lambda v: str(v))
def test_corrupted_replay_lookahead_rtl_matches_faithful(tmp_path, e_depth, cc):
    tmir = _model(seed=1, n_samples=40)
    cfg = _cfg(tmp_path / "base", tmir, e_depth, cc)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)
    picks = _replay_picks(words, 15, Geometry.from_config(cfg).feat_addr_w, seed=e_depth + 10 * cc)

    failures = []
    for k, (i, b) in enumerate(picks):
        c = cfg.model_copy(update={"output_dir": tmp_path / f"r{k}"})
        art = build_replay_bundle(tmir, flip(words, i, b), c, mode=cfg.decoder_mode,
                                  backend_cls=RedressLookaheadBackend)
        ok, out = run_iverilog(art)
        if not ok:
            failures.append((i, b, out[-300:]))
    assert len(picks) == 20
    assert not failures, failures


def test_interior_e_flips_change_nothing_with_lookahead(tmp_path):
    """Depth 1: an E flip on any word at least two words inside its class is
    invisible -- the RTL's predictions equal the fault-free ones."""
    tmir = _model(seed=2, n_samples=40)
    cfg = _cfg(tmp_path / "base", tmir, 1, True)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    g = Geometry.from_config(cfg)
    clean = predict_batches(words, 4, X, 32, g, **cfg.decoder_mode)

    interior, pos = [], 0
    for clauses in split_stream(words):
        n = sum(map(len, clauses))
        interior += range(pos + 2, pos + n - 2)
        pos += n
    rng = np.random.default_rng(3)
    for k, i in enumerate(rng.choice(interior, size=6, replace=False)):
        flipped = flip(words, int(i), 13)
        assert (predict_batches(flipped, 4, X, 32, g, **cfg.decoder_mode) == clean).all()
        c = cfg.model_copy(update={"output_dir": tmp_path / f"e{k}"})
        art = build_replay_bundle(tmir, flipped, c, mode=cfg.decoder_mode,
                                  backend_cls=RedressLookaheadBackend)
        ok, out = run_iverilog(art)  # RTL == faithful(flipped) == faithful(clean)
        assert ok, out[-300:]
        # As built, the same flip does change predictions.
        assert (predict_batches(flipped, 4, X, 32, g) != clean).any()


@pytest.mark.parametrize("depth,where,bit", [
    (1, "last", 13), (1, "last", 14), (1, "first", 13),
    (2, "last", 13), (2, "second-last", 13), (2, "last-two", 13), (2, "first", 13),
], ids=["d1-last-E", "d1-last-CC", "d1-first-E",
        "d2-last-E", "d2-second-last-E", "d2-last-two-E", "d2-first-E"])
def test_stream_edge_flips(tmp_path, depth, where, bit):
    """The two ends of the stream need their own rules, and only flips there
    show whether the RTL has them -- random replays almost never land there.

    last word:  there is no next word. Without the have_next gate,
                instr_rdata still holds the CURRENT word (no look-ahead read
                is issued), so the stale 'next' word always confirms.
    first word: E_prev must start at E_INIT = 1 with have_class set. If word 0
                were latched unchecked (as built), a flipped E on word 0 would
                be accepted at word 1 and every class would shift.
    last two (depth 2): the change at the second-to-last word IS confirmed by
                the last word, and only the missing word ptr+2 (have_next2)
                can reject it. Without that gate the RTL reads past the end
                of the stream."""
    tmir = _model(seed=5, n_samples=64)
    cfg = _cfg(tmp_path / "base", tmir, depth, True)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    g = Geometry.from_config(cfg)
    n = len(words)
    indices = {"last": [n - 1], "second-last": [n - 2], "last-two": [n - 2, n - 1],
               "first": [0]}[where]
    flipped = words
    for index in indices:
        flipped = flip(flipped, index, bit)

    la = predict_batches(flipped, 4, X, 32, g, **cfg.decoder_mode)
    accepted = predict_batches(flipped, 4, X, 32, g)  # as-built rule accepts the change
    assert (la != accepted).any(), "fixture must make the rejection observable"

    art = build_replay_bundle(tmir, flipped, cfg.model_copy(update={"output_dir": tmp_path / "r"}),
                              mode=cfg.decoder_mode, backend_cls=RedressLookaheadBackend)
    ok, out = run_iverilog(art)
    assert ok, out[-300:]


@pytest.mark.parametrize("depth", [1, 2])
def test_lookahead_adds_no_cycles(tmp_path, depth):
    """Same Instruction Memory contents, same vectors: redress_accel and
    redress_la_accel finish at exactly the same simulation time."""
    tmir = _model(seed=4, n_samples=64)
    cfg = _cfg(tmp_path / "base", tmir, depth, True)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)

    times = {}
    for name, cls, mode in (("as_built", RedressBackend, {}),
                            ("lookahead", RedressLookaheadBackend, cfg.decoder_mode)):
        c = cfg.model_copy(update={"output_dir": tmp_path / name})
        ok, out = run_iverilog(build_replay_bundle(tmir, words, c, mode=mode, backend_cls=cls))
        assert ok, out[-300:]
        times[name] = _finish_time(out)
    assert times["as_built"] == times["lookahead"], times


def test_gtkw_has_lookahead_group(tmp_path):
    tmir = _model(n_samples=8)
    art = RedressLookaheadBackend().generate(tmir, _cfg(tmp_path, tmir))
    gtkw = (art.rtl_dir / "sim" / "tb_redress_model.gtkw").read_text()
    for sig in ("E_cur", "E_prev", "E_next", "have_next", "class_boundary"):
        assert f"tb_redress_model.dut.{sig}" in gtkw


@pytest.mark.parametrize("depth,flip_last", [(1, 1), (2, 2)], ids=["d1", "d2"])
@pytest.mark.parametrize("fill", [0x0000, 0xFFFF], ids=["fill0", "fill1"])
def test_memory_beyond_the_stream_is_never_used(tmp_path, depth, flip_last, fill):
    """Instruction Memory past the loaded stream holds whatever an earlier,
    longer model left there. Simulation would show it as X, and `if (X)` takes
    the else branch -- which here happens to mean "reject" -- so an RTL that
    reads past the end can pass in simulation and fail in silicon. Pre-fill
    the memory with real data (all E = 0 and all E = 1) and flip the E of the
    last `flip_last` words, so a read past the end would be consulted."""
    tmir = _model(seed=5, n_samples=64)
    cfg = _cfg(tmp_path / "base", tmir, depth, True)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)
    flipped = words
    for i in range(len(words) - flip_last, len(words)):
        flipped = flip(flipped, i, 13)
    art = build_replay_bundle(tmir, flipped, cfg.model_copy(update={"output_dir": tmp_path / "r"}),
                              mode=cfg.decoder_mode, backend_cls=RedressLookaheadBackend)
    tb = art.rtl_dir / "tb" / "tb_redress_model.v"
    text = tb.read_text()
    head, sep, tail = text.rpartition("endmodule")
    tb.write_text(head + f"""
    // Stale Instruction Memory, as left by a previously loaded longer model.
    integer fill_k;
    initial for (fill_k = 0; fill_k < {cfg.max_includes}; fill_k = fill_k + 1)
        dut.instr_mem.mem[fill_k] = 16'h{fill:04x};
""" + sep + tail)
    ok, out = run_iverilog(art)
    assert ok, out[-300:]


def test_config_rejects_depth_three(tmp_path):
    tmir = _model(n_samples=8)
    with pytest.raises(ValueError, match="no RTL implementation"):
        _cfg(tmp_path, tmir, e_depth=3)


def _interior_starts(words, margin):
    """Word indices at least `margin` words inside their class, on both sides."""
    out, pos = [], 0
    for clauses in split_stream(words):
        n = sum(map(len, clauses))
        out += range(pos + margin, pos + n - margin - 1)
        pos += n
    return out


def test_adjacent_e_flips_change_nothing_at_depth_two(tmp_path):
    """Two flipped E bits in ADJACENT words, well inside a class: depth 2
    shrugs them off (the reason to choose it); depth 1 does not."""
    tmir = _model(seed=6, n_samples=40)
    cfg = _cfg(tmp_path / "base", tmir, 2, True)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    g = Geometry.from_config(cfg)
    clean = predict_batches(words, 4, X, 32, g, **cfg.decoder_mode)
    d1 = {"e_depth": 1, "cc_lookahead": True, "e_init": 1}

    rng = np.random.default_rng(8)
    depth1_broken = 0
    for k, i in enumerate(rng.choice(_interior_starts(words, 3), size=6, replace=False)):
        flipped = flip(flip(words, int(i), 13), int(i) + 1, 13)
        assert (predict_batches(flipped, 4, X, 32, g, **cfg.decoder_mode) == clean).all()
        depth1_broken += (predict_batches(flipped, 4, X, 32, g, **d1) != clean).any()
        c = cfg.model_copy(update={"output_dir": tmp_path / f"a{k}"})
        art = build_replay_bundle(tmir, flipped, c, mode=cfg.decoder_mode,
                                  backend_cls=RedressLookaheadBackend)
        ok, out = run_iverilog(art)  # RTL == faithful(flipped) == faithful(clean)
        assert ok, out[-300:]
    assert depth1_broken > 0
