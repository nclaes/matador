#!/usr/bin/env python3
"""Generate a fixed set of redress-family bundles, for before/after refactor diffs.

Usage (from the repo root):
    python docs/lookahead_exercise/scripts/make_bundles.py build/bundles_before
    # ...change the code...
    python docs/lookahead_exercise/scripts/make_bundles.py build/bundles_after
    diff -r build/bundles_before build/bundles_after && echo IDENTICAL

Every bundle is generated into the same scratch folder (build/_bundle_work) and
then copied to OUT_DIR/<case>. Generated files embed absolute paths (model_path,
output_dir), so generating straight into two different folders would make every
run "different" for reasons that have nothing to do with the code.

Cases cover more than one config: a power-of-two class count, a multi-batch
model, a non-default batch_width, and a two-model reprogram suite.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

from matador.backends.base import ReprogramStep
from matador.backends.registry import config_class_for, get
from matador.inference.reference import predict
from matador.ir.tm_ir import (
    TMIR,
    Architecture,
    EvalVector,
    Hyperparameters,
    Representation,
    Verification,
)

WORK = Path("build/_bundle_work")


def make_tmir(n_features, n_classes, cpc, density, n_samples, seed) -> TMIR:
    """Same recipe as tests/backends/test_redress_rtl.py::_make_tmir."""
    rng = np.random.default_rng(seed)
    inc = rng.random((n_classes, cpc, 2 * n_features)) < density
    for c in range(n_classes):
        inc[c, 0, 0] = True
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
    X = (rng.random((n_samples, n_features)) < 0.5).astype(np.uint8)
    preds, scores = predict(tmir, X)
    tmir.verification = Verification(test_vectors=[
        EvalVector(input=X[i].tolist(), expected_class=int(preds[i]),
                   expected_scores=scores[i].tolist())
        for i in range(len(X))
    ])
    return tmir


CASES = {
    # name: (tmir kwargs, config overrides)
    "c4_of_4": (dict(n_features=15, n_classes=4, cpc=8, density=0.08, n_samples=32, seed=14),
                dict(max_classes=4)),
    "c3_multibatch": (dict(n_features=20, n_classes=3, cpc=6, density=0.15, n_samples=97, seed=3),
                      dict(max_classes=16)),
    "c2_bw16": (dict(n_features=15, n_classes=2, cpc=10, density=0.3, n_samples=20, seed=1),
                dict(max_classes=8, batch_width=16)),
}


def build(backend_name: str, out_dir: Path) -> None:
    backend = get(backend_name)()
    cfg_cls = config_class_for(backend_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    def fresh_cfg(model_path: Path, **overrides):
        payload = dict(model_path=str(model_path), output_dir=str(WORK),
                       target_fpga="xc7z020", max_includes=4096, max_features=256,
                       max_classes=16, batch_width=32)
        payload.update(overrides)
        return cfg_cls.model_validate(payload)

    for name, (tkw, ckw) in CASES.items():
        shutil.rmtree(WORK, ignore_errors=True)
        WORK.mkdir(parents=True)
        tmir = make_tmir(**tkw)
        tmir.to_yaml(WORK / "model.yaml")
        cfg = fresh_cfg(WORK / "model.yaml", **ckw)
        backend.generate(tmir, cfg)
        shutil.copytree(WORK, out_dir / name, dirs_exist_ok=False)

    # Two-model reprogram suite on one generated core.
    shutil.rmtree(WORK, ignore_errors=True)
    WORK.mkdir(parents=True)
    a = make_tmir(n_features=15, n_classes=2, cpc=8, density=0.3, n_samples=40, seed=10)
    b = make_tmir(n_features=15, n_classes=2, cpc=8, density=0.25, n_samples=50, seed=11)
    a.to_yaml(WORK / "a.yaml")
    b.to_yaml(WORK / "b.yaml")
    cfg = fresh_cfg(WORK / "a.yaml")
    backend.generate(a, cfg)
    steps = [ReprogramStep(tmir_path=WORK / "a.yaml", name="model_a"),
             ReprogramStep(tmir_path=WORK / "b.yaml", name="model_b")]
    backend.build_reprogram_suite(WORK / "RTL", steps, cfg)
    shutil.copytree(WORK, out_dir / "reprogram_suite", dirs_exist_ok=False)
    shutil.rmtree(WORK, ignore_errors=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--backend", default="redress")
    args = ap.parse_args(argv)
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    build(args.backend, args.out_dir)
    n = sum(1 for p in args.out_dir.rglob("*") if p.is_file())
    print(f"{args.backend}: wrote {n} files under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
