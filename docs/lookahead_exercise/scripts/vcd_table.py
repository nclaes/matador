#!/usr/bin/env python3
"""Print selected DUT signals once per clock cycle from an iverilog .vcd, as a
table -- a text view of what GTKWave shows, handy for READMEs and diffs.

    python docs/lookahead_exercise/scripts/vcd_table.py RTL/sim/tb_redress_model.vcd \\
        --signals state,ptr,instr_latched,instr_rdata,E_prev,E_next,class_boundary \\
        --ptr 40 44 --first-batch

Values are sampled just before each rising clk edge, i.e. what the register
holds DURING that cycle. --ptr keeps only cycles whose `ptr` is in [lo, hi].
"""
from __future__ import annotations

import argparse


def read_vcd(path, wanted, scope="dut"):
    """Return (rows, widths): one {signal: binary string} per rising clk edge,
    and each wanted signal's declared width."""
    ids, stack, vals, rows, clk, widths = {}, [], {}, [], None, {}
    with open(path) as fh:
        it = iter(fh)
        for line in it:
            t = line.split()
            if not t:
                continue
            if t[0] == "$scope":
                stack.append(t[2])
            elif t[0] == "$upscope":
                stack.pop()
            elif t[0] == "$var" and stack and stack[-1] == scope:
                if t[4] in wanted:
                    ids.setdefault(t[3], t[4])
                    widths[t[4]] = int(t[2])
                if t[4] == "clk":
                    clk = t[3]
            elif t[0] == "$enddefinitions":
                break
        for line in it:
            line = line.strip()
            if not line or line[0] in "#$":
                continue
            v, code = line[1:].split() if line[0] in "bB" else (line[0], line[1:])
            if code == clk and v == "1":
                rows.append(dict(vals))
            if code in ids:
                vals[ids[code]] = v
    return rows, widths


def fmt(v, width=1):
    """Hex for buses wider than 4 bits (VCD drops leading zeros, so the
    declared width decides), decimal otherwise."""
    if v is None or "x" in v or "z" in v:
        return "x"
    n = int(v, 2)
    return f"{n:#06x}" if width > 4 else str(n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vcd")
    ap.add_argument("--signals", required=True)
    ap.add_argument("--ptr", nargs=2, type=int, metavar=("LO", "HI"))
    ap.add_argument("--first-batch", action="store_true",
                    help="stop at the first S_INFER_FINAL (state 7)")
    a = ap.parse_args(argv)
    sigs = a.signals.split(",")
    want = set(sigs) | {"ptr", "state"}
    print(" | ".join(["cyc"] + sigs))
    rows, widths = read_vcd(a.vcd, want)
    for n, r in enumerate(rows):
        if a.first_batch and r.get("state") and fmt(r["state"]) == "7":
            break
        p = r.get("ptr")
        if a.ptr and (p is None or "x" in p or not a.ptr[0] <= int(p, 2) <= a.ptr[1]):
            continue
        if r.get("state") is None or fmt(r["state"]) not in {"3", "4", "5", "6"}:
            continue
        print(" | ".join([str(n)] + [fmt(r.get(s), widths.get(s, 1)) for s in sigs]))


if __name__ == "__main__":
    main()
