"""GP-tiled backend RTL generator.

Wraps GP_TM_Inference_Accelerator's tm_accel_gp.v — a runtime-reprogrammable
core. Unlike vanilla_tiled/vanilla_hardwired, generate() does not bake model
weights into freshly generated RTL: the datapath sources are static (only
capacity parameters vary), and the specific trained model is delivered
separately as an AXI-Stream CMD_LOAD packet, generated here as a .memh file.

generate() writes:
  RTL/src/*.v            — capacity-parameterized static datapath
  RTL/tb/*.v               — generic protocol-conformance testbench (always
                            the same regardless of model) + a model-specific
                            testbench that loads *this* TMIR model and
                            replays its embedded verification.test_vectors
  RTL/sim/*.memh, *.vh   — memh vectors for both testbenches
  RTL/sim/*.sh            — iverilog run / waves scripts
  RTL/sim/gen_vectors.py, tm_emulator.py, capacity.json, model_reference.json
                            — standalone (no matador install needed) tooling to
                            build CMD_LOAD/CMD_INFER vectors for a NEW model
                            against this already-synthesized, runtime-
                            reprogrammable core
  RTL/provenance.json     — software manifest identifying which trained model's
                            weights model_stimulus.memh currently encodes
                            (fingerprint, geometry, training provenance)
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

from matador.backends.base import ResourceEstimate, RTLArtifacts, RTLBackend

_VENDOR_DIR = Path(__file__).parent / "vendor"

_STATIC_SOURCES = ("axis_fifo.v", "clause_eval.v", "tile_mem.v", "score_acc_rt.v", "argmax_rt.v")
_STATIC_TESTBENCHES = ("tb_tile_mem.v", "tb_score_acc_rt.v", "tb_argmax_rt.v")
_STATIC_VECTORS = (
    "gp_stimulus.memh", "gp_expected.memh",
    "gp_small_load.memh", "gp_small_infer.memh", "gp_sizes.vh",
)
# Exported standalone tooling — copied verbatim into RTL/sim/ so someone with
# just the exported RTL folder (no matador install) can regenerate CMD_LOAD/
# CMD_INFER vectors for a NEW model against this already-synthesized core.
# gen_vectors.py is executable; tm_emulator.py is its sibling import.
_EXPORTED_TOOLS = ("tm_emulator.py", "gen_vectors.py")

_INCLUDE_RE = re.compile(r'`include\s+"vectors/gp_sizes\.vh"')
_READMEMH_RE = re.compile(r'\$readmemh\("vectors/([^"]+)"')


def _clog2(n: int) -> int:
    """Number of bits needed to represent 0..n-1 (matches $clog2 semantics)."""
    return max(1, (max(n, 1) - 1).bit_length())


def _sub_param(text: str, name: str, value: int) -> str:
    pattern = re.compile(rf"(parameter\s+{name}\s*=\s*)\d+")
    new_text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one 'parameter {name} = ...' in vendored "
            f"tm_accel_gp.v, found {count}"
        )
    return new_text


class GPTiledBackend(RTLBackend):
    """Runtime-reprogrammable tiled accelerator (GP_TM_Inference_Accelerator).

    One synthesis, sized by GPTiledAcceleratorConfig's capacity fields, can
    be reprogrammed at runtime with any per_class TM model that fits inside
    that capacity — no resynthesis needed to swap models. Tuning knobs:

      target_fpga, bram_bits_budget
        — the real capacity constraint: tile_mem must fit the device's BRAM
      feat_slice, clause_slice
        — per-cycle SIMD width ("how many rounds of computation" per inference)
      max_features, max_clauses_total, max_classes
        — compile-time capacity, in real units
      fifo_depth
        — input FIFO depth in beats

    Trade-off: larger capacity -> more BRAM (tile_mem); the AXI-Stream header
    format's own field widths cap how far this can go regardless of device
    size (see matador.backends.gp_tiled.tmir_bridge.HARD_MAX_*).
    """

    @property
    def name(self) -> str:
        return "vanilla_gp_tiled"

    @property
    def config_class(self) -> type:
        from matador.backends.gp_tiled.config import GPTiledAcceleratorConfig
        return GPTiledAcceleratorConfig

    def generate(self, tmir, config) -> RTLArtifacts:
        from matador.backends.gp_tiled.tmir_bridge import check_capacity, tmir_to_tmmodel

        check_capacity(tmir, config)
        model = tmir_to_tmmodel(tmir, name=config.model_path.stem)

        rtl_dir = config.output_dir / "RTL"
        src_dir = rtl_dir / "src"
        tb_dir = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"
        for d in (src_dir, tb_dir, sim_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── sources: static, only tm_accel_gp.v's capacity params vary ─────
        sources = []
        for fname in _STATIC_SOURCES:
            dst = src_dir / fname
            dst.write_text((_VENDOR_DIR / "src" / fname).read_text())
            sources.append(dst)

        top_path = src_dir / "tm_accel_gp.v"
        top_path.write_text(self._render_top(config))
        sources.append(top_path)

        # ── generic protocol-conformance testbench + its fixed vectors ─────
        testbenches = []
        for fname in _STATIC_TESTBENCHES:
            dst = tb_dir / fname
            dst.write_text((_VENDOR_DIR / "tb" / fname).read_text())
            testbenches.append(dst)

        proto_tb_path = tb_dir / "tb_system_gp.v"
        proto_tb_path.write_text(
            self._flatten_includes((_VENDOR_DIR / "tb" / "tb_system_gp.v").read_text())
        )
        testbenches.append(proto_tb_path)

        sim_files = []
        for fname in _STATIC_VECTORS:
            dst = sim_dir / fname
            dst.write_text((_VENDOR_DIR / "vectors" / fname).read_text())
            sim_files.append(dst)

        # ── model-specific testbench: loads *this* model, replays its
        #    embedded verification vectors ───────────────────────────────
        model_tb_path, model_vec_paths = self._gen_model_testbench(
            tmir, model, config, tb_dir, sim_dir
        )
        testbenches.append(model_tb_path)
        sim_files.extend(model_vec_paths)

        # ── exported standalone tooling: try a NEW model against this
        #    bitstream without matador installed ───────────────────────────
        for fname in _EXPORTED_TOOLS:
            dst = sim_dir / fname
            dst.write_text((_VENDOR_DIR / fname).read_text())
            if fname == "gen_vectors.py":
                dst.chmod(0o755)
            sim_files.append(dst)

        capacity_path = sim_dir / "capacity.json"
        capacity_path.write_text(self._gen_capacity_json(config))
        sim_files.append(capacity_path)

        model_ref_path = sim_dir / "model_reference.json"
        model_ref_path.write_text(self._gen_model_reference_json(model))
        sim_files.append(model_ref_path)

        # ── provenance manifest: which trained model's weights are packed
        #    into model_stimulus.memh, for traceability on a runtime-
        #    reprogrammable core where RTL != model 1:1 ───────────────────
        provenance_path = rtl_dir / "provenance.json"
        provenance_path.write_text(self._gen_provenance(tmir, config, model))
        sim_files.append(provenance_path)

        readme_path = rtl_dir / "README.md"
        readme_path.write_text(self._gen_readme(config, model))

        for name, content in self._gen_sim_scripts(rtl_dir).items():
            p = sim_dir / name
            p.write_text(content)
            p.chmod(0o755)
            sim_files.append(p)

        sim_files.append(readme_path)

        return RTLArtifacts(
            rtl_dir=rtl_dir,
            sources=sources,
            testbenches=testbenches,
            sim_scripts=sim_files,
        )

    @property
    def emulator_class(self):
        from matador.backends.gp_tiled.emulator import GPTiledEmulator
        return GPTiledEmulator

    def resource_estimate(self, tmir, config) -> ResourceEstimate:
        from matador.backends.gp_tiled.fpga_budget import bram_budget_bits

        n_tiles_max = config.n_tiles_max
        tile_width = config.clause_slice * 2 * config.feat_slice
        bram_bits = config.required_bram_bits
        budget_bits = bram_budget_bits(config.target_fpga, config.bram_bits_budget)
        pct = 100.0 * bram_bits / budget_bits if budget_bits else 0.0
        return ResourceEstimate(
            bram_bits=bram_bits,
            notes=(
                f"tile_mem: {n_tiles_max} rows x {tile_width} bits = {bram_bits:,} bits "
                f"({bram_bits // 8192} Xilinx 36Kb BRAMs at typical packing) = "
                f"{pct:.1f}% of the {budget_bits:,}-bit tile_mem budget for "
                f"target_fpga={config.target_fpga!r}. "
                "Runtime-reprogrammable: this capacity is fixed at synthesis time; "
                "any model within max_features/max_clauses_total/max_classes loads "
                "without resynthesis. Run Vivado synthesis for exact LUT/BRAM numbers."
            ),
        )

    # ------------------------------------------------------------------
    # RTL rendering
    # ------------------------------------------------------------------

    def _render_top(self, config) -> str:
        text = (_VENDOR_DIR / "src" / "tm_accel_gp.v").read_text()
        for pname, value in self._capacity_params(config):
            text = _sub_param(text, pname, value)
        return text

    def _capacity_params(self, config) -> list[tuple[str, int]]:
        """Every `parameter NAME = ...` in tm_accel_gp.v that must be
        re-derived from this config, in substitution order. Shared by
        _render_top() (the generated src/tm_accel_gp.v) and
        _render_model_testbench() (the DUT instantiation in the generated
        testbench) so the two can never drift apart."""
        max_feat_padded = config.max_feat_slices * config.feat_slice
        tile_width = config.clause_slice * 2 * config.feat_slice
        n_tiles_max = config.n_tiles_max
        words_per_tile = tile_width // config.axis_data_width
        return [
            ("MAX_CLASSES", config.max_classes),
            ("MAX_CLAUSES_TOTAL", config.max_clauses_total),
            ("MAX_FEAT_SLICES", config.max_feat_slices),
            ("MAX_CLAUSE_SLICES", config.max_clause_slices),
            ("FEAT_SLICE", config.feat_slice),
            ("CLAUSE_SLICE", config.clause_slice),
            ("MAX_FEAT_PADDED", max_feat_padded),
            ("TILE_WIDTH", tile_width),
            ("N_TILES_MAX", n_tiles_max),
            ("TILE_AW", _clog2(n_tiles_max)),
            ("WORD_CNT_W", _clog2(words_per_tile)),
            ("AXIS_DATA_WIDTH", config.axis_data_width),
            ("CLASS_WIDTH", _clog2(config.max_classes)),
            ("FIFO_DEPTH", config.fifo_depth),
        ]

    def _flatten_includes(self, text: str) -> str:
        """Inline `include "vectors/gp_sizes.vh" and flatten $readmemh
        "vectors/X" paths to "X" — matches sim_dir holding vectors directly
        rather than a nested vectors/ subdirectory (avoids depending on
        `include search-path semantics under an unspecified compile-time
        CWD; see matador.models.validator.validate_rtl)."""
        sizes_vh = (_VENDOR_DIR / "vectors" / "gp_sizes.vh").read_text()
        text = _INCLUDE_RE.sub(lambda _m: sizes_vh, text)
        text = _READMEMH_RE.sub(r'$readmemh("\1"', text)
        return text

    def _gen_model_testbench(self, tmir, model, config, tb_dir: Path, sim_dir: Path):
        from matador.backends.gp_tiled.vendor.tm_emulator import (
            encode_infer_packet,
            encode_load_packet,
            expected_load_ack,
            stream_with_tlast,
            write_memh,
        )

        test_vectors = list(tmir.verification.test_vectors) if tmir.verification else []
        fv_ints = [
            sum((int(b) & 1) << f for f, b in enumerate(tv.input))
            for tv in test_vectors
        ]

        packets = [encode_load_packet(model)]
        if fv_ints:
            packets.append(encode_infer_packet(model, fv_ints))
        stim_words, stim_lasts = stream_with_tlast(packets)

        exp_words = [expected_load_ack(model)]
        exp_lasts = [1]
        for i, tv in enumerate(test_vectors):
            exp_words.append(tv.expected_class)
            exp_lasts.append(1 if i == len(test_vectors) - 1 else 0)

        stim_path = sim_dir / "model_stimulus.memh"
        exp_path = sim_dir / "model_expected.memh"
        write_memh(stim_path, stim_words, stim_lasts)
        write_memh(exp_path, exp_words, exp_lasts)

        tb_path = tb_dir / "tb_system_gp_model.v"
        tb_path.write_text(
            self._render_model_testbench(
                config, n_stim=len(stim_words), n_exp=len(exp_words)
            )
        )
        return tb_path, [stim_path, exp_path]

    def _render_model_testbench(self, config, n_stim: int, n_exp: int) -> str:
        params = dict(self._capacity_params(config))
        dut_params = ", ".join(f".{name}({value})" for name, value in params.items() if name != "AXIS_DATA_WIDTH")
        # TO_MAX is a PER-BEAT wait budget inside expect_beat, but since the
        # sender/receiver run concurrently (fork/join), the receiver's very
        # first wait (the LOAD ack) blocks until the ENTIRE load payload has
        # streamed in — up to n_stim beats. Scale the watchdog off n_stim
        # (dominated by n_tiles*64 for large capacity configs) rather than a
        # fixed constant sized only for the small default-capacity demo model.
        to_max = max(200_000, 4 * n_stim)
        return textwrap.dedent(f"""\
            `timescale 1ns/1ps
            // Auto-generated by matador generate (vanilla_gp_tiled backend).
            // Loads the trained TMIR model via CMD_LOAD, then replays its embedded
            // verification.test_vectors via CMD_INFER, checking each predicted
            // class against tmir.verification.test_vectors[i].expected_class.
            module tb_system_gp_model;

                parameter AXIS_DATA_WIDTH = {config.axis_data_width};
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

                // race-free handshake sampling (negedge shadows) — same idiom as
                // the vendored tb_system_gp.v
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
                    $dumpfile("tb_system_gp_model.vcd");
                    $dumpvars(1, tb_system_gp_model);
                    $dumpvars(1, dut);

                    $readmemh("model_stimulus.memh", stim);
                    $readmemh("model_expected.memh", expc);

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
                        $display("tb_system_gp_model: ALL TESTS PASSED (%0d beats checked)", N_EXP);
                    else
                        $display("tb_system_gp_model: FAILED (%0d errors)", fail_cnt);
                    $finish;
                end

                initial begin
                    #40_000_000;
                    $display("TIMEOUT: global watchdog expired");
                    $finish;
                end

            endmodule
            """)

    def _gen_capacity_json(self, config) -> str:
        """This bitstream's synthesized capacity, in the shape the exported
        gen_vectors.py needs to configure tm_emulator's globals correctly
        for any NEW model it's asked to try — must match _capacity_params()."""
        payload = {
            "target_fpga": config.target_fpga,
            "axis_data_width": config.axis_data_width,
            "feat_slice": config.feat_slice,
            "clause_slice": config.clause_slice,
            "max_features": config.max_features,
            "max_clauses_total": config.max_clauses_total,
            "max_classes": config.max_classes,
            "max_feat_slices": config.max_feat_slices,
            "max_clause_slices": config.max_clause_slices,
            "n_tiles_max": config.n_tiles_max,
            "required_bram_bits": config.required_bram_bits,
        }
        return json.dumps(payload, indent=2) + "\n"

    def _gen_model_reference_json(self, model) -> str:
        """The model this bundle was generated with, in gen_vectors.py's
        model JSON schema — doubles as a concrete, working schema example."""
        payload = {
            "name": model.name,
            "n_features": model.n_features,
            "n_classes": model.n_classes,
            "clauses_per_class": model.clauses_per_class,
            "threshold": model.threshold,
            "include": list(model.include),
        }
        return json.dumps(payload, indent=2) + "\n"

    def _gen_provenance(self, tmir, config, model) -> str:
        """Software manifest identifying exactly which trained model's
        weights are packed into model_stimulus.memh / model_reference.json.

        On a runtime-reprogrammable core, RTL != model 1:1 the way it does
        for vanilla_tiled/vanilla_hardwired — the same bitstream can later
        be reprogrammed with a different model via a fresh CMD_LOAD (see
        gen_vectors.py). This manifest describes the model current AT
        GENERATE TIME only; it goes stale the moment someone replays a
        different model's CMD_LOAD against the live hardware. Regenerate it
        by re-running `matador generate`, or track provenance for a new
        model yourself alongside whatever gen_vectors.py output you use to
        reprogram it.
        """
        prov = tmir.provenance
        payload = {
            "matador_backend": self.name,
            "model_fingerprint": tmir.fingerprint(),
            "model_name": model.name,
            "architecture": {
                "n_features": model.n_features,
                "n_classes": model.n_classes,
                "clauses_per_class": model.clauses_per_class,
                "n_clauses_total": model.n_clauses_total,
                "threshold": model.threshold,
            },
            "synthesized_capacity": {
                "target_fpga": config.target_fpga,
                "feat_slice": config.feat_slice,
                "clause_slice": config.clause_slice,
                "max_features": config.max_features,
                "max_clauses_total": config.max_clauses_total,
                "max_classes": config.max_classes,
            },
            "training_provenance": (
                {
                    "framework": prov.framework,
                    "framework_version": prov.framework_version,
                    "dataset_id": prov.dataset_id,
                    "trained_at": prov.trained_at.isoformat(),
                    "epochs": prov.epochs,
                }
                if prov is not None
                else None
            ),
        }
        if prov is None:
            payload["_note"] = (
                "tmir.provenance was not set on the source TMIR model "
                "(e.g. a hand-constructed or externally-authored model) — "
                "training_provenance is unavailable, but model_fingerprint "
                "above still uniquely identifies this exact model."
            )
        return json.dumps(payload, indent=2) + "\n"

    def _gen_sim_scripts(self, rtl_dir: Path) -> dict[str, str]:
        src = rtl_dir / "src"
        tb = rtl_dir / "tb"
        srcs = " ".join(f"{src}/{f}" for f in (*_STATIC_SOURCES, "tm_accel_gp.v"))
        return {
            "run_iverilog.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                set -e
                cd "$(dirname "$0")"
                SRCS="{srcs}"
                iverilog -g2001 -Wall -Wno-timescale -o tb_tile_mem      $SRCS {tb}/tb_tile_mem.v      && vvp tb_tile_mem
                iverilog -g2001 -Wall -Wno-timescale -o tb_score_acc_rt $SRCS {tb}/tb_score_acc_rt.v  && vvp tb_score_acc_rt
                iverilog -g2001 -Wall -Wno-timescale -o tb_argmax_rt    $SRCS {tb}/tb_argmax_rt.v     && vvp tb_argmax_rt
                iverilog -g2001 -Wall -Wno-timescale -o tb_system_gp    $SRCS {tb}/tb_system_gp.v     && vvp tb_system_gp
                iverilog -g2001 -Wall -Wno-timescale -o tb_system_gp_model $SRCS {tb}/tb_system_gp_model.v && vvp tb_system_gp_model
                """),
            "waves.sh": textwrap.dedent("""\
                #!/usr/bin/env bash
                # Usage: bash waves.sh [tb_name]
                set -euo pipefail
                TB="${1:-tb_system_gp_model}"
                SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
                WF="$SCRIPT_DIR/$TB.vcd"
                if [ ! -f "$WF" ]; then
                    echo "No waveform for $TB. Run simulation first: bash run_iverilog.sh"; exit 1
                fi
                echo "Opening GTKWave: $WF"
                gtkwave "$WF" &
                """),
        }

    def _gen_readme(self, config, model) -> str:
        from matador.backends.gp_tiled.fpga_budget import bram_budget_bits

        budget_bits = bram_budget_bits(config.target_fpga, config.bram_bits_budget)
        pct = 100.0 * config.required_bram_bits / budget_bits if budget_bits else 0.0
        return textwrap.dedent(f"""\
            # vanilla_gp_tiled — runtime-reprogrammable TM accelerator

            Vendored from GP_TM_Inference_Accelerator (tm_accel_gp.v). Unlike
            vanilla_tiled/vanilla_hardwired, this core is **not** regenerated per
            model: it is synthesized once at the capacity below, then reprogrammed
            at runtime over `s_axis` with a CMD_LOAD packet (see `src/tm_accel_gp.v`
            header comment for the full AXI-Stream protocol).

            ## Target device
              target_fpga        = {config.target_fpga}
              tile_mem bits       = {config.required_bram_bits:,} / {budget_bits:,} budget ({pct:.1f}%)

            ## Compile-time capacity (this synthesis)
              feat_slice          = {config.feat_slice}   (features/cycle)
              clause_slice        = {config.clause_slice}   (clauses/cycle)
              max_features        = {config.max_features}
              max_clauses_total   = {config.max_clauses_total}
              max_classes         = {config.max_classes}
              max_feat_slices     = {config.max_feat_slices}  (derived)
              max_clause_slices   = {config.max_clause_slices}  (derived)
              fifo_depth          = {config.fifo_depth}

            ## Model loaded for simulation ({model.name})
              n_features         = {model.n_features}
              n_classes          = {model.n_classes}
              clauses_per_class  = {model.clauses_per_class}
              threshold          = {model.threshold}
              n_tiles            = {model.n_tiles}

            ## Testbenches
              tb_tile_mem, tb_score_acc_rt, tb_argmax_rt — unit tests
              tb_system_gp        — generic protocol-conformance suite (LOAD/INFER,
                                    batching, error injection + recovery); independent
                                    of the model above
              tb_system_gp_model  — loads the model above and replays its embedded
                                    verification.test_vectors; this is the
                                    AcceleratorSpec == RTL leg of Matador's
                                    correctness contract

            Run: `bash sim/run_iverilog.sh`

            ## Trying a new model without resynthesizing
            This core is runtime-reprogrammable: swap in a different model with a
            fresh CMD_LOAD packet, no rebuild needed, as long as it fits the
            capacity above. `sim/gen_vectors.py` is a standalone tool (only needs
            `sim/tm_emulator.py` alongside it — no matador install) that builds
            CMD_LOAD/CMD_INFER `.memh` streams for any model you describe in its
            JSON schema; `sim/model_reference.json` is a working example (the
            exact model this bundle's own `model_stimulus.memh` was built from),
            and `sim/capacity.json` is what it reads to validate a new model
            against *this* bitstream's actual synthesized capacity:

                python3 sim/gen_vectors.py combined my_model.json --random -n 20 \\
                    -o my_stim.memh --expected my_expected.memh

            `provenance.json` (repo root) identifies which trained model's
            weights are packed into `model_stimulus.memh` right now — model
            fingerprint, geometry, and training provenance if the source TMIR
            had it. It describes the model at generate time only; if you replay
            a different model's CMD_LOAD against the live hardware, it goes
            stale for that hardware state (there's no on-chip readback to
            re-verify what's actually loaded — this is a software record, not a
            hardware memory-integrity check).
            """)
