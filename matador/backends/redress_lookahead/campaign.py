"""Exhaustive single-bit fault campaign on the REDRESS instruction memory.

For every bit of every stored word: flip it, run the test vectors through
decode_faithful(), and count how many predictions differ from the fault-free
run. P(bit) is that fraction; N x P is the sum over all bits ("effective
sensitive bits", lower is better).

    python -m matador.backends.redress_lookahead.campaign --synthetic --variant v0
    python -m matador.backends.redress_lookahead.campaign --model m.yaml --variant v0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from matador.backends.redress_lookahead.faithful import Geometry, predict_batches
from matador.backends.redress_lookahead.replay import BIT_FIELD, FIELD_BITS

FIELDS = list(FIELD_BITS)


@dataclass
class Variant:
    name: str
    description: str
    e_depth: int = 0
    cc_lookahead: bool = False

    @property
    def mode(self) -> dict:
        if self.e_depth == 0 and not self.cc_lookahead:
            return {}
        return {"e_depth": self.e_depth, "cc_lookahead": self.cc_lookahead,
                "e_init": 1 if self.e_depth > 0 else None}

    def encode(self, tmir) -> list[int]:
        if self.e_depth == 0 and not self.cc_lookahead:
            from matador.backends.redress.codec import mustang_encode
            return mustang_encode(tmir)
        from matador.backends.redress_lookahead.codec_la import encode
        return encode(tmir, e_depth=self.e_depth, cc_lookahead=self.cc_lookahead)


VARIANTS = {
    "v0": Variant("v0", "as built"),
    "v3": Variant("v3", "E look-ahead", e_depth=1),
    "v4": Variant("v4", "E and CC look-ahead", e_depth=1, cc_lookahead=True),
    "v3d2": Variant("v3d2", "E look-ahead, two words (software only)", e_depth=2),
    "v4d2": Variant("v4d2", "E (two words) and CC look-ahead (software only)", e_depth=2,
                    cc_lookahead=True),
}


@dataclass
class CampaignResult:
    variant: str
    n_words: int
    n_vectors: int
    p: np.ndarray  # (n_words, 16) fraction of predictions changed
    clean_words: int = 0
    by_field: dict = field(default_factory=dict)

    @property
    def nxp(self) -> float:
        return float(self.p.sum())


def run_campaign(words, n_classes, X, W, geometry: Geometry, mode=None) -> np.ndarray:
    """P for every (word, bit), shape (len(words), 16)."""
    mode = dict(mode or {})
    clean = predict_batches(words, n_classes, X, W, geometry, **mode)
    p = np.zeros((len(words), 16))
    for i in range(len(words)):
        for b in range(16):
            flipped = list(words)
            flipped[i] ^= 1 << b
            pred = predict_batches(flipped, n_classes, X, W, geometry, **mode)
            p[i, b] = np.mean(pred != clean)
    return p


def summarise(variant: str, p: np.ndarray, n_vectors: int, clean_words: int) -> CampaignResult:
    res = CampaignResult(variant=variant, n_words=p.shape[0], n_vectors=n_vectors, p=p,
                         clean_words=clean_words)
    for f in FIELDS:
        res.by_field[f] = float(p[:, FIELD_BITS[f]].sum())
    return res


def make_synthetic_tmir(n_features=15, n_classes=4, cpc=40, density=0.08, n_samples=64, seed=0):
    """A small random model with varied predictions, ~380 stored words at the
    defaults. Every class gets at least one include (MUSTANG cannot encode an
    empty class)."""
    from matador.inference.reference import predict
    from matador.ir.tm_ir import (
        TMIR,
        Architecture,
        EvalVector,
        Hyperparameters,
        Representation,
        Verification,
    )

    rng = np.random.default_rng(seed)
    inc = rng.random((n_classes, cpc, 2 * n_features)) < density
    for c in range(n_classes):
        inc[c, 0, 0] = True
    tmir = TMIR(
        variant="vanilla",
        architecture=Architecture(
            n_features=n_features, n_literals=2 * n_features, n_classes=n_classes,
            n_clauses_per_class=cpc, n_clauses_total=n_classes * cpc,
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


def load_model_and_vectors(path, n_vectors=None, seed=0):
    from matador.ir.tm_ir import TMIR

    tmir = TMIR.from_yaml(path) if str(path).endswith((".yaml", ".yml")) else TMIR.from_npz(path)
    if not tmir.verification or not tmir.verification.test_vectors:
        raise SystemExit(f"{path} has no embedded verification.test_vectors")
    X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    if n_vectors is not None and n_vectors < len(X):
        idx = np.sort(np.random.default_rng(seed).choice(len(X), n_vectors, replace=False))
        X = X[idx]
    return tmir, X


def load_test_data(path, n_features, n_vectors=None, seed=0) -> np.ndarray:
    """A `matador booleanize` *_test.txt: one sample per line, space-separated
    bits, label in the last column (dropped: P compares against the model's
    own fault-free prediction, never the label)."""
    raw = np.loadtxt(path, dtype=np.uint8, ndmin=2)
    X = raw[:, :-1]
    if X.shape[1] != n_features:
        raise SystemExit(f"{path}: {X.shape[1]} feature columns, model expects {n_features}")
    if n_vectors is not None and n_vectors < len(X):
        X = X[np.sort(np.random.default_rng(seed).choice(len(X), n_vectors, replace=False))]
    return X


def format_table(results: list[CampaignResult]) -> str:
    base = results[0]
    head = f"{'variant':<8}{'words':>7}{'extra':>8}{'N x P':>10}{'gain':>8}  " + "".join(
        f"{f:>9}" for f in FIELDS)
    lines = [head, "-" * len(head)]
    for r in results:
        extra = 100.0 * (r.n_words - r.clean_words) / r.clean_words if r.clean_words else 0.0
        gain = f"{base.nxp / r.nxp:.0f}x" if r.nxp > 0 else "inf"
        lines.append(
            f"{r.variant:<8}{r.n_words:>7}{extra:>7.1f}%{r.nxp:>10.2f}{gain:>8}  "
            + "".join(f"{r.by_field[f]:>9.2f}" for f in FIELDS)
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Exhaustive single-bit instruction-memory campaign.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use the built-in synthetic model")
    src.add_argument("--model", help="TMIR .yaml/.npz with embedded test vectors")
    ap.add_argument("--variant", nargs="+", default=["v0"], choices=sorted(VARIANTS))
    ap.add_argument("--test-data", help="booleanised *_test.txt (last column = label, dropped); "
                    "default: the model's embedded verification vectors")
    ap.add_argument("--vectors", type=int, default=None, help="subsample this many test vectors")
    ap.add_argument("--batch-width", type=int, default=32)
    ap.add_argument("--max-classes", type=int, default=16)
    ap.add_argument("--max-features", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.synthetic:
        tmir = make_synthetic_tmir(seed=args.seed)
        X = np.array([tv.input for tv in tmir.verification.test_vectors], dtype=np.uint8)
    elif args.test_data:
        tmir, _ = load_model_and_vectors(args.model)
        X = load_test_data(args.test_data, tmir.architecture.n_features, args.vectors, args.seed)
    else:
        tmir, X = load_model_and_vectors(args.model, args.vectors, args.seed)

    from matador.backends.redress.codec import mustang_encode
    from matador.backends.redress.rtl import _clog2

    geometry = Geometry(max_classes=args.max_classes, class_width=_clog2(args.max_classes),
                        feat_addr_w=_clog2(args.max_features))
    n_classes = tmir.architecture.n_classes
    clean_words = len(mustang_encode(tmir))
    print(f"model: {n_classes} classes, {tmir.architecture.n_features} features, "
          f"{clean_words} as-built words; {len(X)} test vectors; "
          f"batch_width={args.batch_width}, max_classes={args.max_classes}")

    results = []
    for name in args.variant:
        v = VARIANTS[name]
        words = v.encode(tmir)
        p = run_campaign(words, n_classes, X, args.batch_width, geometry, v.mode)
        results.append(summarise(name, p, len(X), clean_words))
    print(format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["BIT_FIELD", "VARIANTS", "Variant", "run_campaign", "summarise", "main"]
