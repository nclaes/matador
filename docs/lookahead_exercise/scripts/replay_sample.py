#!/usr/bin/env python3
"""Replay a random sample of single-bit-corrupted Instruction Memories through
the redress (as-built) and redress_lookahead RTLs, under iverilog, and check
each against decode_faithful in the matching decoder mode.

    python docs/lookahead_exercise/scripts/replay_sample.py \\
        --model build/la_stage6/TMIR/mammographic/<model>.yaml \\
        --test-data build/la_stage6/booleanised/mammographic_test.txt \\
        --n 50 --vectors 64 --out build/la_stage6/replays

For every replay it also records whether the flip changed any prediction
(per decode_faithful), so the table shows both "RTL == model" and how many of
the sampled faults were actually visible.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from matador.backends.redress.config import RedressAcceleratorConfig
from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead.campaign import load_test_data
from matador.backends.redress_lookahead.config import RedressLookaheadConfig
from matador.backends.redress_lookahead.faithful import Geometry, predict_batches
from matador.backends.redress_lookahead.replay import (
    BIT_FIELD,
    FIELD_BITS,
    build_replay_bundle,
    flip,
    replay_safe,
    run_iverilog,
)
from matador.backends.redress_lookahead.rtl import RedressLookaheadBackend
from matador.inference.reference import predict
from matador.ir.tm_ir import TMIR, EvalVector, Verification


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--n", type=int, default=50, help="replays per RTL")
    ap.add_argument("--vectors", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fields", nargs="+", default=list(FIELD_BITS), choices=list(FIELD_BITS),
                    help="only flip bits of these fields (default: all 16 bits)")
    ap.add_argument("--out", default="build/replays")
    ap.add_argument("--max-includes", type=int, default=2048)
    ap.add_argument("--max-features", type=int, default=64)
    ap.add_argument("--max-classes", type=int, default=4)
    args = ap.parse_args(argv)

    tmir = TMIR.from_yaml(args.model)
    X = load_test_data(args.test_data, tmir.architecture.n_features, args.vectors, args.seed)
    preds, scores = predict(tmir, X)
    tmir.verification = Verification(test_vectors=[
        EvalVector(input=X[i].tolist(), expected_class=int(preds[i]),
                   expected_scores=scores[i].tolist()) for i in range(len(X))])
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    model_path = out / "model.yaml"
    tmir.to_yaml(model_path)

    base = dict(model_path=str(model_path), output_dir=str(out), target_fpga="xc7z020",
                max_includes=args.max_includes, max_features=args.max_features,
                max_classes=args.max_classes, batch_width=32)
    designs = {
        "redress (as built)": (RedressBackend, RedressAcceleratorConfig.model_validate(base), {}),
        "redress_lookahead (v4)": (RedressLookaheadBackend,
                                   RedressLookaheadConfig.model_validate(base), None),
    }
    n_feat = tmir.architecture.n_features
    rng = np.random.default_rng(args.seed)
    bits = [b for f in args.fields for b in FIELD_BITS[f]]
    print(f"{len(X)} vectors per replay, {args.n} random replay-safe single-bit flips per RTL"
          f" (fields: {' '.join(args.fields)})")
    print(f"{'design':<24}{'replays':>8}{'RTL==model':>12}{'visible':>9}   visible flips by field")
    for name, (cls, cfg, mode) in designs.items():
        mode = cfg.decoder_mode if mode is None else mode
        words = cls().encode_stream(tmir, cfg)
        g = Geometry.from_config(cfg)
        clean = predict_batches(words, tmir.architecture.n_classes, X, 32, g, **mode)
        ok = visible = 0
        by_field: dict[str, int] = {}
        for k in range(args.n):
            while True:
                i, b = int(rng.integers(len(words))), int(rng.choice(bits))
                if replay_safe(words[i] ^ (1 << b), n_feat, g.feat_addr_w):
                    break
            w = flip(words, i, b)
            if (predict_batches(w, tmir.architecture.n_classes, X, 32, g, **mode) != clean).any():
                visible += 1
                by_field[BIT_FIELD[b]] = by_field.get(BIT_FIELD[b], 0) + 1
            c = cfg.model_copy(update={"output_dir": out / f"{cls.__name__}_{k}"})
            passed, log = run_iverilog(build_replay_bundle(tmir, w, c, mode=mode, backend_cls=cls))
            ok += passed
            if not passed:
                print(f"  MISMATCH {name}: word {i} bit {b} ({BIT_FIELD[b]})\n{log[-400:]}")
            shutil.rmtree(out / f"{cls.__name__}_{k}", ignore_errors=True)
        fields = ", ".join(f"{f} {n}" for f, n in sorted(by_field.items())) or "-"
        print(f"{name:<24}{args.n:>8}{ok:>8}/{args.n:<3}{visible:>9}   {fields}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
