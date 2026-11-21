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

COMBOS = [(0, False), (0, True), (1, False), (1, True)]


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


@pytest.mark.parametrize("e_depth,cc", [(1, True), (1, False)], ids=lambda v: str(v))
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


@pytest.mark.parametrize("where,bit", [("last", 13), ("last", 14), ("first", 13)],
                         ids=["last-E", "last-CC", "first-E"])
def test_stream_edge_flips(tmp_path, where, bit):
    """The two ends of the stream need their own rules, and only flips there
    show whether the RTL has them -- random replays almost never land there.

    last word:  there is no next word. Without the have_next gate,
                instr_rdata still holds the CURRENT word (no look-ahead read
                is issued), so the stale 'next' word always confirms.
    first word: E_prev must start at E_INIT = 1 with have_class set. If word 0
                were latched unchecked (as built), a flipped E on word 0 would
                be accepted at word 1 and every class would shift."""
    tmir = _model(seed=5, n_samples=64)
    cfg = _cfg(tmp_path / "base", tmir, 1, True)
    words = RedressLookaheadBackend().encode_stream(tmir, cfg)
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    g = Geometry.from_config(cfg)
    flipped = flip(words, len(words) - 1 if where == "last" else 0, bit)

    la = predict_batches(flipped, 4, X, 32, g, **cfg.decoder_mode)
    accepted = predict_batches(flipped, 4, X, 32, g)  # as-built rule accepts the change
    assert (la != accepted).any(), "fixture must make the rejection observable"

    art = build_replay_bundle(tmir, flipped, cfg.model_copy(update={"output_dir": tmp_path / "r"}),
                              mode=cfg.decoder_mode, backend_cls=RedressLookaheadBackend)
    ok, out = run_iverilog(art)
    assert ok, out[-300:]


def test_lookahead_adds_no_cycles(tmp_path):
    """Same Instruction Memory contents, same vectors: redress_accel and
    redress_la_accel finish at exactly the same simulation time."""
    tmir = _model(seed=4, n_samples=64)
    cfg = _cfg(tmp_path / "base", tmir, 1, True)
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


def test_config_rejects_depth_two(tmp_path):
    tmir = _model(n_samples=8)
    with pytest.raises(ValueError, match="no RTL implementation"):
        _cfg(tmp_path, tmir, e_depth=2)
