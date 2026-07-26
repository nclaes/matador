#!/usr/bin/env python3
"""
gen_vectors.py — standalone CMD_LOAD/CMD_INFER vector generator for an
exported vanilla_gp_tiled RTL bundle.

Self-contained: needs only this file plus the sibling tm_emulator.py that
`matador generate --backend vanilla_gp_tiled` copies into the same
directory (RTL/sim/). No matador install required — stdlib only.

Use this to try a NEW model against the already-synthesized hardware
without resynthesizing, via the same runtime-reprogrammable CMD_LOAD path
`matador generate` itself used to build model_stimulus.memh. See
model_reference.json in this directory for a concrete, working example of
the JSON schema below — it's the exact model this bundle's own
model_stimulus.memh/model_expected.memh testbench vectors were built from.

Model JSON schema:
  {
    "n_features": int,
    "n_classes": int,
    "clauses_per_class": int,   # even
    "threshold": int,
    "include": [int, ...]       # one bitmask per global clause (n_classes *
                                 # clauses_per_class entries); bit f = positive
                                 # literal of feature f, bit n_features+f =
                                 # negated literal of feature f
  }

This bundle was synthesized with fixed capacity ceilings (see capacity.json
alongside this file) — any model you try here must fit within them, or the
RTL will reject the CMD_LOAD with ERR_RANGE (see capacity.json / RTL's
tm_accel_gp.v header comment for the exact fields).

Usage:
  # most users want this: one ready-to-replay LOAD+INFER stream
  python3 gen_vectors.py combined model.json vectors.txt -o stim.memh --expected exp.memh
  python3 gen_vectors.py combined model.json --random -n 20 -o stim.memh --expected exp.memh

  # add --testbench to also get a ready-to-compile Verilog testbench sized
  # for this model (same pass/fail idiom as the bundle's own
  # tb_system_gp_model.v), so you don't have to hand-write one:
  python3 gen_vectors.py combined model.json --random -n 20 \\
      -o stim.memh --expected exp.memh --testbench tb_my_model.v
  # (run from this directory, sim/ -- source files live one level up in ../src/)
  iverilog -g2001 -Wall -Wno-timescale -o tb_my_model \\
      ../src/axis_fifo.v ../src/clause_eval.v ../src/tile_mem.v \\
      ../src/score_acc_rt.v ../src/argmax_rt.v ../src/tm_accel_gp.v \\
      tb_my_model.v && vvp tb_my_model

  # building blocks, if you want LOAD and INFER as separate streams
  python3 gen_vectors.py load model.json -o load.memh
  python3 gen_vectors.py infer model.json vectors.txt -o infer.memh [--expected expected.memh]
  python3 gen_vectors.py random-infer model.json -n 10 -o infer.memh [--seed N]

  # prove reprogrammability with YOUR OWN models: chain several models into
  # one continuous LOAD->INFER->LOAD->INFER->... run (this is the concrete
  # answer to "can I swap models at runtime" -- see manifest.json schema
  # below):
  python3 gen_vectors.py sequence manifest.json \\
      -o seq_stim.memh --expected seq_exp.memh --testbench tb_sequence.v

vectors.txt: one feature vector per line, whitespace/comma-separated 0/1
             bits, exactly n_features bits per line.

manifest.json (for `sequence`): a JSON list of steps, applied in order,
each either
  {"model": "modelA.json", "vectors": "vecsA.txt"}
or
  {"model": "modelB.json", "random": true, "n": 10, "seed": 1}
Paths inside the manifest are resolved relative to the manifest file's own
directory, so a manifest + its referenced model/vector files can be kept
together and moved as a unit.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tm_emulator as te  # noqa: E402 (sibling module copied alongside this file)


def load_capacity(script_dir: Path) -> None:
    """Point tm_emulator's capacity globals at this bundle's synthesized
    capacity (capacity.json, written by `matador generate`), so validate()
    checks new models against the ACTUAL hardware, not tm_emulator.py's own
    built-in defaults."""
    cap_path = script_dir / "capacity.json"
    if not cap_path.exists():
        print(
            f"warning: {cap_path} not found — validating against "
            "tm_emulator.py's built-in defaults, which may not match this "
            "bundle's actual synthesized capacity",
            file=sys.stderr,
        )
        return
    cap = json.loads(cap_path.read_text())
    te.FEAT_SLICE = cap["feat_slice"]
    te.CLAUSE_SLICE = cap["clause_slice"]
    te.TILE_WIDTH = te.CLAUSE_SLICE * 2 * te.FEAT_SLICE
    te.WORDS_PER_ROW = te.TILE_WIDTH // te.AXIS_W
    te.MAX_CLASSES = cap["max_classes"]
    te.MAX_CLAUSES_TOTAL = cap["max_clauses_total"]
    te.MAX_FEAT_SLICES = cap["max_feat_slices"]
    te.MAX_CLAUSE_SLICES = cap["max_clause_slices"]
    te.MAX_TILES = cap["n_tiles_max"]


def load_model(path: Path) -> "te.TMModel":
    data = json.loads(path.read_text())
    model = te.TMModel(
        n_features=data["n_features"],
        n_classes=data["n_classes"],
        clauses_per_class=data["clauses_per_class"],
        threshold=data["threshold"],
        include=list(data["include"]),
        name=data.get("name", path.stem),
    )
    model.validate()
    return model


def _read_vectors(path: Path, n_features: int) -> list:
    vecs = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        bits = line.replace(",", " ").split()
        if len(bits) != n_features:
            raise ValueError(f"{path}:{lineno}: expected {n_features} bits, got {len(bits)}")
        vecs.append(sum((int(b) & 1) << f for f, b in enumerate(bits)))
    return vecs


def _vectors_from_args(args, model) -> list:
    if getattr(args, "random", False):
        rng = random.Random(args.seed)
        return [rng.getrandbits(model.n_features) for _ in range(args.n)]
    if not args.vectors:
        raise SystemExit("error: either supply a vectors file or pass --random")
    return _read_vectors(Path(args.vectors), model.n_features)


def cmd_load(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    words = te.encode_load_packet(model)
    lasts = [0] * (len(words) - 1) + [1]
    te.write_memh(Path(args.out), words, lasts)
    print(f"wrote {len(words)} CMD_LOAD words -> {args.out}")
    print(f"expected LOAD ack: 0x{te.expected_load_ack(model):08X}")
    return 0


def _emit_infer(model, feature_vectors, out_path, expected_path) -> None:
    words = te.encode_infer_packet(model, feature_vectors)
    lasts = [0] * (len(words) - 1) + [1]
    te.write_memh(Path(out_path), words, lasts)
    print(f"wrote {len(words)} CMD_INFER words ({len(feature_vectors)} frames) -> {out_path}")

    preds = [model.infer_tiled(fv)[0] for fv in feature_vectors]
    if expected_path:
        exp_lasts = [0] * (len(preds) - 1) + [1] if preds else []
        te.write_memh(Path(expected_path), preds, exp_lasts)
        print(f"wrote {len(preds)} expected prediction beats -> {expected_path}")
    else:
        for i, p in enumerate(preds):
            print(f"  vector[{i}] -> predicted class {p}")


def cmd_infer(args: argparse.Namespace) -> int:
    model = load_model(Path(args.model))
    vecs = _vectors_from_args(args, model)
    _emit_infer(model, vecs, args.out, args.expected)
    return 0


_TB_TEMPLATE = """\
`timescale 1ns/1ps
// Auto-generated by gen_vectors.py combined --testbench for model {model_name!r}.
// Loads it via CMD_LOAD, then replays the accompanying feature vectors via
// CMD_INFER, checking each predicted class against this tool's own
// TMModel.infer_tiled() (the RTL-exact reference) — same idiom as the
// matador-generated tb_system_gp_model.v this bundle already ships with.
module {tb_name};

    parameter AXIS_DATA_WIDTH = {axis_data_width};
    localparam N_STIM = {n_stim};
    localparam N_EXP  = {n_exp};
    localparam TO_MAX = {to_max};

    reg                        clk, rst_n;
    reg                        s_tvalid;
    wire                       s_tready;
    reg  [AXIS_DATA_WIDTH-1:0] s_tdata;
    reg                        s_tlast;
    wire                       m_tvalid;
    wire                       m_tready;
    wire [AXIS_DATA_WIDTH-1:0] m_tdata;
    wire                       m_tlast;
    wire                       busy, configured;

    tm_accel_gp #(
        {dut_params}, .AXIS_DATA_WIDTH(AXIS_DATA_WIDTH)
    ) dut (
        .clk(clk), .rst_n(rst_n),
        .s_axis_tvalid(s_tvalid), .s_axis_tready(s_tready),
        .s_axis_tdata(s_tdata),   .s_axis_tlast(s_tlast),
        .m_axis_tvalid(m_tvalid), .m_axis_tready(m_tready),
        .m_axis_tdata(m_tdata),   .m_axis_tlast(m_tlast),
        .busy(busy), .configured(configured)
    );

    initial clk = 0;
    always #5 clk = ~clk;
    assign m_tready = 1'b1;

    reg                        rdy_s;
    reg                        mv_s, ml_s;
    reg [AXIS_DATA_WIDTH-1:0]  md_s;
    always @(negedge clk) begin
        rdy_s <= s_tready;
        mv_s  <= m_tvalid;
        md_s  <= m_tdata;
        ml_s  <= m_tlast;
    end

    reg [32:0] stim [0:N_STIM-1];
    reg [32:0] expc [0:N_EXP-1];

    integer sent, fail_cnt, i, e, to;

    task send_beat;
        input [AXIS_DATA_WIDTH-1:0] d;
        input                       l;
        begin
            s_tvalid = 1'b1; s_tdata = d; s_tlast = l;
            @(posedge clk);
            while (!rdy_s) @(posedge clk);
            s_tvalid = 1'b0; s_tlast = 1'b0;
            sent = sent + 1;
        end
    endtask

    task expect_beat;
        input [AXIS_DATA_WIDTH-1:0] d;
        input                       l;
        input integer               idx;
        begin
            to = 0;
            @(posedge clk);
            while (!mv_s) begin
                @(posedge clk);
                to = to + 1;
                if (to >= TO_MAX) begin
                    $display("TIMEOUT waiting for beat[%0d]", idx);
                    fail_cnt = fail_cnt + 1;
                    $finish;
                end
            end
            if (md_s !== d || ml_s !== l) begin
                $display("FAIL beat[%0d]: exp data=%08h last=%b  got data=%08h last=%b",
                         idx, d, l, md_s, ml_s);
                fail_cnt = fail_cnt + 1;
            end else begin
                $display("PASS beat[%0d]: data=%08h last=%b", idx, md_s, ml_s);
            end
        end
    endtask

    initial begin
        $dumpfile("{tb_name}.vcd");
        $dumpvars(1, {tb_name});
        $dumpvars(1, dut);

        $readmemh("{stim_file}", stim);
        $readmemh("{exp_file}", expc);

        sent = 0; fail_cnt = 0;
        rst_n = 0; s_tvalid = 0; s_tdata = 0; s_tlast = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        fork
            begin
                for (i = 0; i < N_STIM; i = i + 1)
                    send_beat(stim[i][AXIS_DATA_WIDTH-1:0], stim[i][32]);
                s_tvalid = 1'b0; s_tlast = 1'b0;
            end
            begin
                for (e = 0; e < N_EXP; e = e + 1)
                    expect_beat(expc[e][AXIS_DATA_WIDTH-1:0], expc[e][32], e);
            end
        join

        if (fail_cnt == 0)
            $display("{tb_name}: ALL TESTS PASSED (%0d beats checked)", N_EXP);
        else
            $display("{tb_name}: FAILED (%0d errors)", fail_cnt);
        $finish;
    end

    initial begin
        #40_000_000;
        $display("TIMEOUT: global watchdog expired");
        $finish;
    end

endmodule
"""


def _emit_testbench(name: str, tb_path: Path, stim_file: str, exp_file: str, n_stim: int, n_exp: int) -> None:
    """name: a label for the testbench/module (e.g. a model name, or a
    manifest name for a multi-model `sequence`) -- purely cosmetic, doesn't
    affect capacity, which comes entirely from tm_emulator's globals
    (already synced by load_capacity() to this bundle's synthesized
    capacity) since the DUT is fixed regardless of which model is loaded."""
    dut_params = ", ".join(
        f".{pname}({getattr(te, pname)})"
        for pname in (
            "MAX_CLASSES", "MAX_CLAUSES_TOTAL", "MAX_FEAT_SLICES", "MAX_CLAUSE_SLICES",
            "FEAT_SLICE", "CLAUSE_SLICE",
        )
    )
    max_feat_padded = te.MAX_FEAT_SLICES * te.FEAT_SLICE
    n_tiles_max = te.MAX_FEAT_SLICES * te.MAX_CLAUSE_SLICES
    tile_width = te.TILE_WIDTH
    tile_aw = max(1, (max(n_tiles_max, 1) - 1).bit_length())
    word_cnt_w = max(1, (max(tile_width // te.AXIS_W, 1) - 1).bit_length())
    class_width = max(1, (max(te.MAX_CLASSES, 1) - 1).bit_length())
    dut_params += (
        f", .MAX_FEAT_PADDED({max_feat_padded}), .TILE_WIDTH({tile_width}), "
        f".N_TILES_MAX({n_tiles_max}), .TILE_AW({tile_aw}), .WORD_CNT_W({word_cnt_w}), "
        f".CLASS_WIDTH({class_width}), .FIFO_DEPTH(16)"
    )
    tb_name = "tb_" + "".join(c if c.isalnum() else "_" for c in name)
    text = _TB_TEMPLATE.format(
        model_name=name, tb_name=tb_name, axis_data_width=te.AXIS_W,
        n_stim=n_stim, n_exp=n_exp, to_max=max(200_000, 4 * n_stim),
        dut_params=dut_params, stim_file=stim_file, exp_file=exp_file,
    )
    tb_path.write_text(text)
    print(f"wrote testbench module {tb_name} -> {tb_path}")
    print(f"  compile (run from this directory, sim/): iverilog -g2001 -Wall -Wno-timescale "
          f"-o {tb_name} ../src/axis_fifo.v ../src/clause_eval.v ../src/tile_mem.v "
          f"../src/score_acc_rt.v ../src/argmax_rt.v ../src/tm_accel_gp.v "
          f"{tb_path.name} && vvp {tb_name}")


def cmd_combined(args: argparse.Namespace) -> int:
    """Build one ready-to-replay LOAD-then-INFER stimulus stream, matching
    the shape of the tb_system_gp_model.v testbench generate() itself
    builds — the recommended entry point for trying a new model."""
    model = load_model(Path(args.model))
    vecs = _vectors_from_args(args, model)

    load_words = te.encode_load_packet(model)
    infer_words = te.encode_infer_packet(model, vecs)
    stim_words, stim_lasts = te.stream_with_tlast([load_words, infer_words])
    te.write_memh(Path(args.out), stim_words, stim_lasts)
    print(f"wrote {len(stim_words)} combined LOAD+INFER words -> {args.out}")

    preds = [model.infer_tiled(fv)[0] for fv in vecs]
    exp_words = [te.expected_load_ack(model), *preds]
    exp_lasts = [1] + ([0] * (len(preds) - 1) + [1] if preds else [])
    if args.expected:
        te.write_memh(Path(args.expected), exp_words, exp_lasts)
        print(f"wrote {len(exp_words)} expected beats (ack + {len(preds)} predictions) -> {args.expected}")
    else:
        print(f"expected LOAD ack: 0x{te.expected_load_ack(model):08X}")
        for i, p in enumerate(preds):
            print(f"  vector[{i}] -> predicted class {p}")

    if args.testbench:
        if not args.expected:
            print("error: --testbench requires --expected (the testbench reads it)", file=sys.stderr)
            return 1
        _emit_testbench(
            model.name, Path(args.testbench),
            stim_file=Path(args.out).name, exp_file=Path(args.expected).name,
            n_stim=len(stim_words), n_exp=len(exp_words),
        )
    return 0


def cmd_sequence(args: argparse.Namespace) -> int:
    """Chain MULTIPLE user-supplied models into one continuous
    LOAD -> INFER -> LOAD -> INFER -> ... stream -- the concrete way to
    prove runtime reprogrammability with YOUR OWN models, not just the two
    built-in demo models tb_system_gp.v ships with (same idea, generalized:
    each step is still just encode_load_packet()+encode_infer_packet(),
    concatenated via stream_with_tlast() same as `combined` does for one
    model).

    manifest.json: a JSON list of steps, each either
      {"model": "modelA.json", "vectors": "vecsA.txt"}
    or
      {"model": "modelB.json", "random": true, "n": 10, "seed": 1}
    Paths are resolved relative to the manifest file's own directory."""
    manifest_path = Path(args.manifest)
    steps = json.loads(manifest_path.read_text())
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{manifest_path}: expected a non-empty JSON list of steps")

    base_dir = manifest_path.resolve().parent
    packets: list = []
    exp_words: list = []
    exp_lasts: list = []

    for idx, step in enumerate(steps):
        model = load_model(base_dir / step["model"])

        if step.get("random"):
            rng = random.Random(step.get("seed", 0))
            vecs = [rng.getrandbits(model.n_features) for _ in range(step.get("n", 10))]
        elif "vectors" in step:
            vecs = _read_vectors(base_dir / step["vectors"], model.n_features)
        else:
            raise ValueError(f"manifest step {idx}: need either \"vectors\" or \"random\": true")

        packets.append(te.encode_load_packet(model))
        if vecs:
            packets.append(te.encode_infer_packet(model, vecs))

        preds = [model.infer_tiled(fv)[0] for fv in vecs]
        exp_words.append(te.expected_load_ack(model))
        exp_lasts.append(1)
        if preds:
            exp_words.extend(preds)
            exp_lasts.extend([0] * (len(preds) - 1) + [1])

        print(f"  step {idx}: model={model.name!r} ({model.n_classes} classes, "
              f"{model.clauses_per_class} clauses/class) -- {len(vecs)} vector(s), "
              f"predictions={preds}")

    stim_words, stim_lasts = te.stream_with_tlast(packets)
    te.write_memh(Path(args.out), stim_words, stim_lasts)
    print(f"wrote {len(stim_words)} words across {len(steps)} reprogram step(s) -> {args.out}")

    if args.expected:
        te.write_memh(Path(args.expected), exp_words, exp_lasts)
        print(f"wrote {len(exp_words)} expected beats -> {args.expected}")

    if args.testbench:
        if not args.expected:
            print("error: --testbench requires --expected (the testbench reads it)", file=sys.stderr)
            return 1
        _emit_testbench(
            manifest_path.stem, Path(args.testbench),
            stim_file=Path(args.out).name, exp_file=Path(args.expected).name,
            n_stim=len(stim_words), n_exp=len(exp_words),
        )
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser(
        "combined",
        help="build a full LOAD+INFER stimulus stream, ready to replay (recommended)",
    )
    pc.add_argument("model")
    pc.add_argument("vectors", nargs="?", help="text file of feature vectors (or use --random)")
    pc.add_argument("-o", "--out", required=True)
    pc.add_argument("--expected", default=None, help="also write expected ack+prediction beats here")
    pc.add_argument("--random", action="store_true", help="use N random feature vectors instead of a file")
    pc.add_argument("-n", type=int, default=10, help="number of random vectors (with --random)")
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--testbench", default=None,
                    help="also write a ready-to-compile Verilog testbench here (requires --expected)")
    pc.set_defaults(func=cmd_combined)

    pl = sub.add_parser("load", help="build a standalone CMD_LOAD .memh from a model JSON file")
    pl.add_argument("model")
    pl.add_argument("-o", "--out", required=True)
    pl.set_defaults(func=cmd_load)

    pi = sub.add_parser("infer", help="build a standalone CMD_INFER .memh for feature vectors from a file")
    pi.add_argument("model")
    pi.add_argument("vectors", nargs="?")
    pi.add_argument("-o", "--out", required=True)
    pi.add_argument("--expected", default=None, help="also write expected prediction beats here")
    pi.add_argument("--random", action="store_true")
    pi.add_argument("-n", type=int, default=10)
    pi.add_argument("--seed", type=int, default=0)
    pi.set_defaults(func=cmd_infer)

    ps = sub.add_parser(
        "sequence",
        help="chain multiple of YOUR OWN models into one continuous reprogram-and-verify run",
    )
    ps.add_argument("manifest", help="JSON list of {model, vectors} or {model, random, n, seed} steps")
    ps.add_argument("-o", "--out", required=True)
    ps.add_argument("--expected", default=None, help="also write expected ack+prediction beats here")
    ps.add_argument("--testbench", default=None,
                    help="also write a ready-to-compile Verilog testbench here (requires --expected)")
    ps.set_defaults(func=cmd_sequence)

    args = p.parse_args(argv)
    load_capacity(Path(__file__).resolve().parent)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
