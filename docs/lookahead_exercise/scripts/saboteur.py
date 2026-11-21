#!/usr/bin/env python3
"""A mini saboteur, for learning: invert one bit of a named register at a
random inference cycle and record whether any prediction changed.

FOR LEARNING ONLY. This is not a substitute for EMBER:
  * it injects one hand-picked register at a time, not all state;
  * "changed" means "some output beat differed from the fault-free run";
    it does not detect reconvergence (a flip whose effect is later
    overwritten), latent corruption, or timing;
  * trials are uniform over inference cycles, so registers that are only
    live for a few cycles per instruction look less sensitive than their
    per-live-cycle sensitivity.

    python docs/lookahead_exercise/scripts/saboteur.py --trials 200 --out build/la_stage7

For each design it generates a bundle, rewrites the model testbench into
tb_saboteur.v (no VCD; a block that counts inference cycles -- state 3..7 --
and inverts `SAB_TARGET on the falling edge of cycle +FLIP_CYCLE), compiles
it once per target with iverilog -DSAB_TARGET=..., then runs vvp once per
trial.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from matador.backends.redress.config import RedressAcceleratorConfig
from matador.backends.redress.rtl import RedressBackend
from matador.backends.redress_lookahead import campaign
from matador.backends.redress_lookahead.config import RedressLookaheadConfig
from matador.backends.redress_lookahead.rtl import RedressLookaheadBackend

# (label, hierarchical reference inside the testbench, what it is)
TARGETS = [
    ("E_prev", "dut.E_prev", "E latch (class boundary reference)"),
    ("CC_prev", "dut.CC_prev", "CC latch (clause boundary reference)"),
    ("have_class", "dut.have_class", "E latch valid flag"),
    ("instr_latched[13]", "dut.instr_latched[13]", "current word's E (DECODE..ACCUM)"),
    ("instr_rdata[13]", "dut.instr_mem.rdata[13]", "BRAM output E (look-ahead word in ACCUM)"),
    ("current_class[0]", "dut.current_class[0]", "class counter LSB"),
]

SABOTEUR_BLOCK = """
    // ── mini saboteur (docs/lookahead_exercise/scripts/saboteur.py) ──────
    integer sab_cycle, sab_count;
    initial begin
        sab_count = 0;
        if (!$value$plusargs("FLIP_CYCLE=%d", sab_cycle)) sab_cycle = -1;
    end
    always @(negedge clk) begin
        if (rst_n && dut.state >= 3 && dut.state <= 7) begin
            if (sab_count == sab_cycle) begin
                `SAB_TARGET = ~`SAB_TARGET;
                $display("SABOTEUR: inverted `SAB_TARGET at inference cycle %0d (state %0d)",
                         sab_count, dut.state);
            end
            sab_count = sab_count + 1;
        end
    end

endmodule"""


def make_saboteur_tb(tb_text: str) -> str:
    tb = re.sub(r"^\s*\$dump(file|vars)\(.*\);\n", "", tb_text, flags=re.M)
    tb = tb.replace("module tb_redress_model;", "module tb_saboteur;", 1)
    head, sep, _ = tb.rpartition("endmodule")
    assert sep, "testbench has no endmodule"
    return head + SABOTEUR_BLOCK.lstrip("\n") + "\n"


def run(args) -> list[tuple[str, str, int, int]]:
    out = Path(args.out)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    tmir = campaign.make_synthetic_tmir(n_samples=32, seed=args.seed)  # one 32-wide batch
    tmir.to_yaml(out / "model.yaml")
    base = dict(model_path=str(out / "model.yaml"), output_dir=str(out), target_fpga="xc7z020",
                max_includes=4096, max_features=256, max_classes=16, batch_width=32)
    designs = [
        ("redress", RedressBackend, RedressAcceleratorConfig),
        ("redress_lookahead", RedressLookaheadBackend, RedressLookaheadConfig),
    ]
    rng = np.random.default_rng(args.seed)
    results = []
    for name, cls, cfg_cls in designs:
        cfg = cfg_cls.model_validate({**base, "output_dir": str(out / name)})
        backend = cls()
        art = backend.generate(tmir, cfg)
        n_words = len(backend.encode_stream(tmir, cfg))
        n_cycles = 4 * n_words + 1  # INSTR/DECODE/FEAT/ACCUM per word + FINAL; one batch
        sim = (art.rtl_dir / "sim").resolve()
        tb = sim / "tb_saboteur.v"
        tb.write_text(make_saboteur_tb((art.rtl_dir / "tb" / "tb_redress_model.v").read_text()))
        srcs = [str(Path(s).resolve()) for s in art.sources]
        for label, ref, _ in TARGETS:
            binary = sim / f"sab_{re.sub(r'[^A-Za-z0-9]', '_', label)}"
            cp = subprocess.run(["iverilog", "-g2001", "-Wno-timescale", f"-DSAB_TARGET={ref}",
                                 "-o", str(binary), *srcs, str(tb)], capture_output=True, text=True)
            if cp.returncode != 0:
                raise SystemExit(cp.stderr)
            changed = 0
            for cyc in rng.integers(n_cycles, size=args.trials):
                r = subprocess.run(["vvp", "-n", str(binary), f"+FLIP_CYCLE={int(cyc)}"],
                                   capture_output=True, text=True, cwd=str(sim))
                if "SABOTEUR:" not in r.stdout:
                    raise SystemExit(f"{name}/{label}: flip at cycle {cyc} never happened")
                changed += ("TESTS PASSED" not in r.stdout)
            results.append((name, label, changed, args.trials))
            print(f"{name:<18} {label:<18} {changed:>4}/{args.trials} trials changed an output",
                  flush=True)
    return results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="build/saboteur")
    args = ap.parse_args(argv)
    results = run(args)
    print("\n| register | what it is | redress | redress_lookahead |\n|---|---|---|---|")
    by = {(d, t): (c, n) for d, t, c, n in results}
    for label, _, what in TARGETS:
        cells = [f"{by[(d, label)][0]}/{by[(d, label)][1]}"
                 for d in ("redress", "redress_lookahead")]
        print(f"| `{label}` | {what} | {cells[0]} | {cells[1]} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
