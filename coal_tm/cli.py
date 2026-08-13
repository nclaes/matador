"""coal_tm — CLI for the Coalesced TM RTL flow.

Subcommands:
  train      Train a TMCoalescedClassifier and export TAs.txt/weights.txt.
  validate   Check TAs.txt/weights.txt predictions against real labeled
             test vectors via the emulator -- run this before `generate`
             to catch a bad export or wrong config cheaply, before
             spending time on RTL.
  generate   Generate RTL from an existing TAs.txt/weights.txt pair
             (independent of `train` -- externally-supplied files work too).
  testbench  Generate a self-checking testbench + stimulus for a bundle
             generate() already produced.
  emulate    Run the standalone Python reference model, on either a single
             hand-supplied vector (--input) or real vectors read from a
             text file -- defaulting to the config's own test_data.
  sim        Compile and run a generated testbench under iverilog (default)
             or Verilator.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from coal_tm.config import RTLConfig, TrainingConfig


def cmd_train(args: argparse.Namespace) -> int:
    from coal_tm.train import train

    config = TrainingConfig.from_yaml(Path(args.config))
    result = train(config)
    print(f"[train] final accuracy: {result.accuracy:.2f}%")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from coal_tm.validate import validate

    config = RTLConfig.from_yaml(Path(args.config))
    test_data = Path(args.test_data) if args.test_data else None
    result = validate(config, n_vectors=args.n_vectors, test_data=test_data)

    for i in range(result.n_vectors):
        status = "OK" if result.predictions[i] == result.labels[i] else "MISMATCH"
        print(f"  [{status}] vector {i}: predicted {result.predictions[i]}, label {result.labels[i]}")
    print(f"[validate] {result.correct}/{result.n_vectors} correct ({result.accuracy:.2f}%)")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    from coal_tm import rtl

    config = RTLConfig.from_yaml(Path(args.config))
    result = rtl.generate(config)
    print(f"[generate] wrote {len(result.sources)} file(s) to {result.rtl_dir}")
    for path in result.sources:
        print(f"  {path}")
    print(f"  {result.readme_path}")
    return 0


def cmd_testbench(args: argparse.Namespace) -> int:
    from coal_tm import testbench

    config = RTLConfig.from_yaml(Path(args.config))
    test_data = Path(args.test_data) if args.test_data else None
    result = testbench.generate(config, n_vectors=args.n_vectors, test_data=test_data)
    print(f"[testbench] {result.n_vectors} vector(s), {result.n_blocks} packet(s) each")
    print(f"  {result.testbench_path}")
    print(f"  {result.stimulus_path}")
    print(f"  {result.expected_path}")
    print(f"  {result.run_iverilog_path}")
    print(f"  {result.run_verilator_path}")
    print(f"  {result.readme_path}")
    return 0


def cmd_emulate(args: argparse.Namespace) -> int:
    from coal_tm.emulate import emulate_batch, emulate_single

    config = RTLConfig.from_yaml(Path(args.config))

    if args.input:
        x = np.array([int(v) for v in args.input.split(",")], dtype=np.uint8)
        result = emulate_single(config, x)
        print(f"predicted class: {result.predicted_class}")
        print(f"class sums: {list(result.class_sums)}")
        print(f"clauses fired: {int(result.clause_outputs.sum())} / {config.clauses}")
        return 0

    test_data = Path(args.test_data) if args.test_data else None
    result = emulate_batch(config, n_vectors=args.n_vectors, test_data=test_data)
    for i, p in enumerate(result.predictions):
        print(f"row {i}: predicted class {p}")
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    from coal_tm.rtl import GENERATED_SOURCES

    rtl_dir = Path(args.rtl_dir)
    tb_path = Path(args.testbench) if args.testbench else rtl_dir / "tb" / "testbench.sv"
    if not tb_path.exists():
        print(f"error: testbench not found: {tb_path}", file=sys.stderr)
        return 1

    sources = [rtl_dir / name for name in GENERATED_SOURCES]
    missing = [s for s in sources if not s.exists()]
    if missing:
        print(f"error: missing generated source(s): {missing}", file=sys.stderr)
        return 1

    sim_dir = rtl_dir / "sim"
    if args.verilator:
        build_dir = sim_dir / "verilator_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["verilator", "--binary", "--timing", "-Wno-fatal", "-Wno-lint",
             "--top-module", "testbench", "-Mdir", str(build_dir), str(tb_path), *map(str, sources)],
            cwd=sim_dir, check=True,
        )
        subprocess.run([str(build_dir / "Vtestbench")], cwd=sim_dir, check=False)
    else:
        vvp_path = sim_dir / "sim.vvp"
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(vvp_path), str(tb_path), *map(str, sources)],
            check=True,
        )
        subprocess.run(["vvp", str(vvp_path)], cwd=sim_dir, check=False)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="coal_tm", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train a Coalesced TM and export TAs/weights")
    p_train.add_argument("--config", required=True, help="Path to a training YAML config")
    p_train.set_defaults(func=cmd_train)

    p_validate = sub.add_parser("validate", help="Check TAs/weights predictions against labeled test vectors")
    p_validate.add_argument("--config", required=True, help="Path to an RTL YAML config")
    p_validate.add_argument("--n-vectors", type=int, default=20)
    p_validate.add_argument("--test-data", default=None, help="Override config's test_data")
    p_validate.set_defaults(func=cmd_validate)

    p_generate = sub.add_parser("generate", help="Generate RTL from TAs.txt/weights.txt")
    p_generate.add_argument("--config", required=True, help="Path to an RTL YAML config")
    p_generate.set_defaults(func=cmd_generate)

    p_testbench = sub.add_parser("testbench", help="Generate a self-checking testbench")
    p_testbench.add_argument("--config", required=True, help="Path to an RTL YAML config")
    p_testbench.add_argument("--n-vectors", type=int, default=10)
    p_testbench.add_argument("--test-data", default=None, help="Override config's test_data")
    p_testbench.set_defaults(func=cmd_testbench)

    p_emulate = sub.add_parser("emulate", help="Run the Python reference model")
    p_emulate.add_argument("--config", required=True, help="Path to an RTL YAML config")
    p_emulate.add_argument("--input", default=None, help="Comma-separated 0/1 feature vector (single ad hoc row)")
    p_emulate.add_argument("--test-data", default=None, help="Override config's test_data (default source of vectors)")
    p_emulate.add_argument("--n-vectors", type=int, default=None, help="Limit to the first N rows (default: all)")
    p_emulate.set_defaults(func=cmd_emulate)

    p_sim = sub.add_parser("sim", help="Compile and run a generated testbench")
    p_sim.add_argument("--rtl-dir", required=True, help="The RTL/ directory generate() wrote")
    p_sim.add_argument("--testbench", default=None, help="Override the default tb/testbench.sv path")
    p_sim.add_argument("--verilator", action="store_true", help="Use Verilator instead of iverilog")
    p_sim.set_defaults(func=cmd_sim)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:  # noqa: BLE001 - CLI top-level error boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
