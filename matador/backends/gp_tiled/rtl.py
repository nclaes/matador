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
  RTL/sim/test_vectors.txt — human-readable companion to model_stimulus.memh:
                            which inputs are embedded and what class each one
                            should predict (the .memh files are opaque hex)
  RTL/sim/tb_system_gp_model.gtkw
                            — curated GTKWave layout grouping AXI-Stream I/O so
                            the predicted class (m_axis_tdata) is easy to find
                            and compare against test_vectors.txt; waves.sh uses
                            it automatically
  RTL/sim/*.sh            — iverilog run / waves scripts
  RTL/sim/verilator/{Makefile,tb_top.cpp}
                            — Verilator harness (static, no per-config templating:
                            it replays whatever --stim/--exp .memh it's given at
                            runtime, so the same binary works for this model, a
                            brand-new one, or a gen_vectors.py sequence)
  RTL/sim/gen_vectors.py, tm_emulator.py, capacity.json, model_reference.json
                            — standalone (no matador install needed) tooling to
                            build CMD_LOAD/CMD_INFER vectors — including
                            multi-model reprogramming sequences (`sequence`
                            subcommand) — for NEW models against this
                            already-synthesized, runtime-reprogrammable core
  RTL/provenance.json     — software manifest identifying which trained model's
                            weights model_stimulus.memh currently encodes
                            (fingerprint, geometry, training provenance)
"""

from __future__ import annotations

import json
import math
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

        # ── human-readable test-vector reference + a curated waveform view,
        #    so "which signal shows the predicted class" and "what's it
        #    supposed to be" are both answered without spelunking the RTL ──
        test_vectors_path = sim_dir / "test_vectors.txt"
        test_vectors_path.write_text(self._gen_test_vectors_txt(tmir, model))
        sim_files.append(test_vectors_path)

        model_gtkw_path = sim_dir / "tb_system_gp_model.gtkw"
        model_gtkw_path.write_text(self._gen_model_tb_gtkw())
        sim_files.append(model_gtkw_path)

        # ── Verilator harness: static, no per-config templating needed (it
        #    replays whatever --stim/--exp .memh it's given at runtime; see
        #    tb_top.cpp header). matador.models.validator.validate_rtl picks
        #    this up automatically once sim/verilator/Makefile exists.
        verilator_dir = sim_dir / "verilator"
        verilator_dir.mkdir(exist_ok=True)
        for fname in ("Makefile", "tb_top.cpp"):
            dst = verilator_dir / fname
            dst.write_text((_VENDOR_DIR / "verilator" / fname).read_text())
            sim_files.append(dst)

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

        for name, content in self._gen_sim_scripts().items():
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

    @property
    def supports_reprogramming(self) -> bool:
        return True

    def build_reprogram_suite(self, rtl_dir: Path, steps, config) -> RTLArtifacts:
        """Build a testbench + stimulus that reprograms an ALREADY-GENERATED
        bundle (rtl_dir) across multiple {model, dataset} steps in one
        continuous LOAD -> INFER -> LOAD -> INFER -> ... run — generalizing
        the single-model path (_gen_model_testbench) and the vendored
        gen_vectors.py `sequence` subcommand's own algorithm, reusing the
        SAME matador-side infra (tmir_bridge, vendor.tm_emulator) rather
        than duplicating either.
        """
        from matador.backends.gp_tiled.tmir_bridge import check_capacity, tmir_to_tmmodel
        from matador.backends.gp_tiled.vendor.tm_emulator import (
            encode_infer_packet,
            encode_load_packet,
            expected_load_ack,
            stream_with_tlast,
            write_memh,
        )
        from matador.ir.tm_ir import TMIR

        if not steps:
            raise ValueError("build_reprogram_suite: steps must be non-empty")

        tb_dir = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"
        if not sim_dir.exists():
            raise FileNotFoundError(
                f"{rtl_dir} has no sim/ directory — run `matador generate` for "
                "this backend/config first (build_reprogram_suite augments an "
                "already-generated bundle, it doesn't create one)."
            )

        packets = []
        exp_words: list[int] = []
        exp_lasts: list[int] = []
        step_reports = []

        for i, step in enumerate(steps):
            suffix = step.tmir_path.suffix.lower()
            tmir = TMIR.from_yaml(step.tmir_path) if suffix in {".yaml", ".yml"} else TMIR.from_npz(step.tmir_path)
            check_capacity(tmir, config)
            model_name = step.name or step.tmir_path.stem
            model = tmir_to_tmmodel(tmir, name=model_name)

            pairs = self._resolve_step_vectors(step, tmir, model)

            packets.append(encode_load_packet(model))
            if pairs:
                packets.append(encode_infer_packet(model, [fv for fv, _ in pairs]))

            exp_words.append(expected_load_ack(model))
            exp_lasts.append(1)
            if pairs:
                preds = [cls for _, cls in pairs]
                exp_words.extend(preds)
                exp_lasts.extend([0] * (len(preds) - 1) + [1])

            step_reports.append({
                "index": i, "model": model_name, "tmir_path": str(step.tmir_path),
                "dataset": str(step.dataset_path) if step.dataset_path else None,
                "vectors": str(step.vectors_path) if step.vectors_path else None,
                "expected_classes": [cls for _, cls in pairs],
            })

        stim_words, stim_lasts = stream_with_tlast(packets)

        stim_path = sim_dir / "reprogram_stimulus.memh"
        exp_path = sim_dir / "reprogram_expected.memh"
        write_memh(stim_path, stim_words, stim_lasts)
        write_memh(exp_path, exp_words, exp_lasts)

        tb_path = tb_dir / "tb_reprogram_suite.v"
        tb_path.write_text(self._render_model_testbench(
            config, n_stim=len(stim_words), n_exp=len(exp_words),
            tb_name="tb_reprogram_suite",
            stim_file="reprogram_stimulus.memh", exp_file="reprogram_expected.memh",
            header_comment=(
                "// Multi-model/dataset reprogramming suite (matador reprogram-suite).\n"
                "    // Reprograms this ALREADY-SYNTHESIZED core across several models in one\n"
                "    // continuous LOAD -> INFER -> LOAD -> INFER -> ... run -- see\n"
                "    // sim/reprogram_manifest.txt for which beat belongs to which step."
            ),
        ))

        manifest_path = sim_dir / "reprogram_manifest.txt"
        manifest_path.write_text(self._gen_reprogram_manifest_txt(step_reports))

        gtkw_path = sim_dir / "tb_reprogram_suite.gtkw"
        gtkw_path.write_text(self._gen_model_tb_gtkw(tb_name="tb_reprogram_suite"))

        return RTLArtifacts(
            rtl_dir=rtl_dir,
            sources=[],
            testbenches=[tb_path],
            sim_scripts=[stim_path, exp_path, manifest_path, gtkw_path],
        )

    def _resolve_step_vectors(self, step, tmir, model) -> "list[tuple[int, int]]":
        """(feature_vector_int, expected_class) pairs for one step, per the
        precedence documented on ReprogramStep: vectors_path > dataset_path
        > this TMIR's own embedded verification.test_vectors. expected_class
        is always the MODEL's own (software-reference-matching) prediction
        — via model.infer_tiled(), same as gen_vectors.py computes it for
        any new model — except for the embedded-vectors case, where
        tv.expected_class is already exactly that (see _gen_model_testbench)
        and recomputing it would be redundant."""
        if step.vectors_path:
            fv_ints = self._read_vector_file(step.vectors_path, model.n_features)
            return [(fv, model.infer_tiled(fv)[0]) for fv in fv_ints]
        if step.dataset_path:
            fv_ints = self._read_dataset_features(step.dataset_path, model.n_features, step.n_samples, step.seed)
            return [(fv, model.infer_tiled(fv)[0]) for fv in fv_ints]

        test_vectors = list(tmir.verification.test_vectors) if tmir.verification else []
        return [
            (sum((int(b) & 1) << f for f, b in enumerate(tv.input)), tv.expected_class)
            for tv in test_vectors
        ]

    def _read_vector_file(self, path: Path, n_features: int) -> list[int]:
        """Raw bit-vector file: one vector per line, whitespace/comma-
        separated 0/1 bits — the same format gen_vectors.py's --vectors
        flag reads."""
        vecs = []
        for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            bits = line.replace(",", " ").split()
            if len(bits) != n_features:
                raise ValueError(f"{path}:{lineno}: expected {n_features} bits, got {len(bits)}")
            vecs.append(sum((int(b) & 1) << f for f, b in enumerate(bits)))
        return vecs

    def _read_dataset_features(self, path: Path, n_features: int, n_samples, seed: int) -> list[int]:
        """A booleanized *_test.txt-shaped file (space-separated uint, last
        column = label — matador.models.trainer's own format). The label
        column is dropped: reprogram-suite always checks RTL against the
        model's OWN prediction (model.infer_tiled), not the dataset's
        ground-truth label — consistent with how every other testbench in
        this backend closes the AcceleratorSpec == RTL contract."""
        import numpy as np

        raw = np.genfromtxt(path, dtype=np.uint32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        X = raw[:, :-1]
        if X.shape[1] != n_features:
            raise ValueError(
                f"{path}: dataset has {X.shape[1]} feature columns, model expects {n_features}"
            )
        if n_samples is not None and n_samples < X.shape[0]:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(X.shape[0], size=n_samples, replace=False))
            X = X[idx]
        return [sum((int(b) & 1) << f for f, b in enumerate(row)) for row in X]

    def _gen_reprogram_manifest_txt(self, step_reports: list[dict]) -> str:
        """Human-readable companion to reprogram_stimulus.memh/
        reprogram_expected.memh — same role as test_vectors.txt, but for a
        multi-step reprogramming run: which step each beat belongs to, and
        what the beat immediately before each step's is (its LOAD ack)."""
        lines = [
            "Multi-model/dataset reprogramming suite -- one continuous",
            "LOAD -> INFER -> LOAD -> INFER -> ... stream across the steps below,",
            "in order, packed into reprogram_stimulus.memh / reprogram_expected.memh",
            "and replayed by tb_reprogram_suite.v (iverilog) or the Verilator",
            "harness (sim/verilator --stim reprogram_stimulus.memh",
            "--exp reprogram_expected.memh).",
            "",
            "HOW TO READ THIS AGAINST THE WAVEFORM/CONSOLE OUTPUT:",
            "  Each step contributes one LOAD ack beat, then one prediction beat",
            "  per vector, in order. Beat indices accumulate across steps -- e.g.",
            "  if step 0 has 3 vectors, its beats are e=0 (ack) .. e=3 (last",
            "  vector), and step 1's ack is e=4.",
            "",
        ]
        beat = 0
        for step in step_reports:
            lines.append(f"step {step['index']}: model={step['model']!r}  tmir={step['tmir_path']}")
            if step["dataset"]:
                lines.append(f"  vectors from dataset: {step['dataset']}")
            elif step["vectors"]:
                lines.append(f"  vectors from file: {step['vectors']}")
            else:
                lines.append("  vectors from: this model's own embedded verification.test_vectors")
            lines.append(f"  beat {beat}: LOAD ack")
            beat += 1
            for i, cls in enumerate(step["expected_classes"]):
                lines.append(f"  beat {beat}: vector[{i}] -> expected_class {cls}")
                beat += 1
            lines.append("")
        return "\n".join(lines) + "\n"

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
        # score_width is sized to the TRUE worst-case vote magnitude any
        # model within this backend's compile-time capacity could ever
        # produce: a single class holding half of max_clauses_total
        # clauses, all one polarity, all firing (e.g. n_classes=1,
        # clauses_per_class=max_clauses_total). score_acc_rt.v no longer
        # clamps (see its own header comment), so the register must never
        # overflow regardless of what threshold a loaded model happens to
        # carry. Mirrors vanilla_tiled's TMAccelerator.score_width sizing.
        max_half_clauses = config.max_clauses_total // 2
        score_width = max(4, _clog2(max_half_clauses + 1) + 2)
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
            ("SCORE_WIDTH", score_width),
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

    def _render_model_testbench(
        self, config, n_stim: int, n_exp: int,
        tb_name: str = "tb_system_gp_model",
        stim_file: str = "model_stimulus.memh",
        exp_file: str = "model_expected.memh",
        header_comment: str = (
            "// Loads the trained TMIR model via CMD_LOAD, then replays its embedded\n"
            "    // verification.test_vectors via CMD_INFER, checking each predicted\n"
            "    // class against tmir.verification.test_vectors[i].expected_class."
        ),
    ) -> str:
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
            {header_comment}
            module {tb_name};

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

    def _gen_test_vectors_txt(self, tmir, model) -> str:
        """Human-readable companion to model_stimulus.memh/model_expected.memh
        (which are opaque hex) — exactly which test vectors are embedded, what
        class each one should predict, and how to line that up against the
        waveform beat-by-beat. This is the direct answer to "where are the
        test vectors" / "how do I check predicted vs actual" for someone
        opening this bundle for the first time."""
        vecs = list(tmir.verification.test_vectors) if tmir.verification else []
        lines = [
            "Test vectors embedded in this bundle's model, from the source TMIR's",
            "verification.test_vectors. These are the EXACT vectors packed into",
            "model_stimulus.memh (as CMD_INFER frames) and checked against",
            "model_expected.memh -- the same vectors tb_system_gp_model.v",
            "(iverilog) and the Verilator harness (sim/verilator, default",
            "--stim/--exp) both replay to prove",
            "AcceleratorSpec.simulate(x) == RTL_simulation(x).",
            "",
            "HOW TO READ THIS AGAINST THE WAVEFORM (bash sim/waves.sh):",
            "  Watch m_axis_tdata every time m_axis_tvalid fires (the curated",
            "  waveform view groups these under 'AXI-Stream Output' -- that's",
            "  the PREDICTED CLASS, in the low bits, each time it pulses).",
            "  Beat 0 is the LOAD ack (data = 0xA5000xxx -- not a prediction,",
            "  ignore it). Beat 1 is vector[0]'s prediction, beat 2 is",
            "  vector[1]'s, and so on in order -- match each beat's m_axis_tdata",
            "  against the expected_class column below.",
            "",
            f"model: {model.name}  ({model.n_features} features, {model.n_classes} classes, "
            f"{model.clauses_per_class} clauses/class, threshold={model.threshold})",
            f"{len(vecs)} test vector(s):",
            "",
            f"{'idx':>4}  {'expected_class':>14}  input (bit 0 = feature 0, left-to-right)",
        ]
        for i, tv in enumerate(vecs):
            bits = "".join(str(int(b) & 1) for b in tv.input)
            lines.append(f"{i:>4}  {tv.expected_class:>14}  {bits}")
        if not vecs:
            lines.append("(none embedded -- this TMIR model has no verification.test_vectors,")
            lines.append(" so model_stimulus.memh only contains the CMD_LOAD packet, no INFER frames)")
        return "\n".join(lines) + "\n"

    def _gen_model_tb_gtkw(self, tb_name: str = "tb_system_gp_model") -> str:
        """Curated GTKWave layout for <tb_name>.vcd, grouping signals so the
        AXI-Stream output (where the predicted class appears) is
        immediately obvious rather than buried in an unsorted signal tree —
        reuses matador.waves.gtkwave's low-level .gtkw helpers (already
        used by TMAccelerator's own waveform generation)."""
        from matador.waves.gtkwave import _BIN, _DEC, _HEX, _blank, _group, _group_end, _header, _sig

        T = tb_name
        lines = [_header(f"{T}.vcd")]

        lines.append(_group("AXI-Stream Input (LOAD payload, then feature vectors)"))
        lines.append(_sig(f"{T}.s_tvalid", _BIN))
        lines.append(_sig(f"{T}.s_tready", _BIN))
        lines.append(_sig(f"{T}.s_tdata", _HEX))
        lines.append(_sig(f"{T}.s_tlast", _BIN))
        lines.append(_group_end("AXI-Stream Input (LOAD payload, then feature vectors)"))

        lines.append(_blank())
        lines.append(_group("AXI-Stream Output (PREDICTED CLASS appears in m_tdata here)"))
        lines.append(_sig(f"{T}.m_tvalid", _BIN))
        lines.append(_sig(f"{T}.m_tready", _BIN))
        lines.append(_sig(f"{T}.m_tdata", _HEX))
        lines.append(_sig(f"{T}.m_tlast", _BIN))
        lines.append(_group_end("AXI-Stream Output (PREDICTED CLASS appears in m_tdata here)"))

        lines.append(_blank())
        lines.append(_group("Status"))
        lines.append(_sig(f"{T}.busy", _BIN))
        lines.append(_sig(f"{T}.configured", _BIN))
        lines.append(_sig(f"{T}.dut.state", _HEX))
        lines.append(_group_end("Status"))

        lines.append(_blank())
        lines.append(_group("Testbench bookkeeping (cross-reference sim/test_vectors.txt by this index)"))
        lines.append(_sig(f"{T}.e", _DEC))
        lines.append(_sig(f"{T}.sent", _DEC))
        lines.append(_sig(f"{T}.fail_cnt", _DEC))
        lines.append(_group_end("Testbench bookkeeping (cross-reference sim/test_vectors.txt by this index)"))

        return "".join(lines)

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

    def _gen_sim_scripts(self) -> dict[str, str]:
        srcs = " ".join(f"$SRC_DIR/{f}" for f in (*_STATIC_SOURCES, "tm_accel_gp.v"))
        return {
            "run_iverilog.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                # Portable: paths are resolved relative to this script's own location
                # at runtime, not baked in at generation time -- works regardless of
                # where the RTL folder is copied/moved to, inside or outside a
                # container (this used to hardcode absolute paths derived from
                # output_dir at `matador generate` time, e.g. /work/..., which broke
                # as soon as the bundle was used somewhere /work wasn't mounted).
                set -e
                SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
                SRC_DIR="$SCRIPT_DIR/../src"
                TB_DIR="$SCRIPT_DIR/../tb"
                cd "$SCRIPT_DIR"
                SRCS="{srcs}"
                iverilog -g2001 -Wall -Wno-timescale -o tb_tile_mem      $SRCS $TB_DIR/tb_tile_mem.v      && vvp tb_tile_mem
                iverilog -g2001 -Wall -Wno-timescale -o tb_score_acc_rt $SRCS $TB_DIR/tb_score_acc_rt.v  && vvp tb_score_acc_rt
                iverilog -g2001 -Wall -Wno-timescale -o tb_argmax_rt    $SRCS $TB_DIR/tb_argmax_rt.v     && vvp tb_argmax_rt
                iverilog -g2001 -Wall -Wno-timescale -o tb_system_gp    $SRCS $TB_DIR/tb_system_gp.v     && vvp tb_system_gp
                iverilog -g2001 -Wall -Wno-timescale -o tb_system_gp_model $SRCS $TB_DIR/tb_system_gp_model.v && vvp tb_system_gp_model
                """),
            "waves.sh": textwrap.dedent("""\
                #!/usr/bin/env bash
                # Usage: bash waves.sh [tb_name|verilator]
                #   tb_name    — an iverilog testbench's .vcd (default: tb_system_gp_model)
                #   verilator  — the Verilator harness's .fst (sim/verilator/tb_top.fst),
                #                produced by `make -C verilator run`
                # With no argument, prefers the Verilator trace if one exists (usually
                # the most recent/most relevant run for large-capacity models), falling
                # back to the default iverilog testbench's .vcd otherwise.
                set -euo pipefail
                SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
                ARG="${1:-}"

                if [ "$ARG" = "verilator" ]; then
                    WF="$SCRIPT_DIR/verilator/tb_top.fst"
                elif [ -n "$ARG" ]; then
                    WF="$SCRIPT_DIR/$ARG.vcd"
                elif [ -f "$SCRIPT_DIR/verilator/tb_top.fst" ]; then
                    WF="$SCRIPT_DIR/verilator/tb_top.fst"
                else
                    WF="$SCRIPT_DIR/tb_system_gp_model.vcd"
                fi

                if [ ! -f "$WF" ]; then
                    echo "No waveform at $WF. Run simulation first:"
                    echo "  bash run_iverilog.sh          (produces .vcd)"
                    echo "  make -C verilator run          (produces verilator/tb_top.fst)"
                    exit 1
                fi

                # If a curated .gtkw layout exists for this waveform (groups
                # AXI-Stream input/output so the predicted class is easy to
                # find instead of an unsorted signal tree -- see
                # sim/test_vectors.txt for what to compare it against), open
                # with it. Falls back to GTKWave's default view otherwise.
                GTKW="${WF%.vcd}.gtkw"
                if [ -f "$GTKW" ]; then
                    echo "Opening GTKWave: $WF  (layout: $(basename "$GTKW"))"
                    gtkwave "$WF" "$GTKW" &
                else
                    echo "Opening GTKWave: $WF"
                    gtkwave "$WF" &
                fi
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
            model: it is synthesized once at the capacity in the reference table
            below, then reprogrammed at runtime over `s_axis` with a CMD_LOAD
            packet (see `src/tm_accel_gp.v`'s header comment for the full
            AXI-Stream protocol). This README walks through using this bundle in
            the order you'll actually need it — start at 0 if you're new here.

            ## 0. Orientation — what's in this folder, and where the test vectors are

              src/                        Datapath RTL. tm_accel_gp.v is the top
                                          module (see its header comment for the
                                          full LOAD/INFER protocol); the other 5
                                          files are shared sub-modules (FIFO,
                                          per-clause evaluator, tile memory, score
                                          accumulator, argmax).
              tb/                         Testbenches. tb_system_gp_model.v is
                                          built for *this* model (below);
                                          tb_system_gp.v is a generic protocol
                                          suite; the rest are per-module unit
                                          tests.
              sim/                        Everything you actually run.

              sim/test_vectors.txt         <- THE TEST VECTORS. Human-readable:
                                          which inputs are embedded and what
                                          class each one should predict. Start
                                          here if you're trying to find them.
              sim/model_stimulus.memh      The same vectors, machine-readable —
              sim/model_expected.memh      what the RTL/testbenches actually
                                          read. test_vectors.txt is what YOU
                                          read; these are what the RTL reads.
              sim/model_reference.json     This model's raw parameters (geometry
                                          + compiled weights), in
                                          gen_vectors.py's JSON schema — also a
                                          working example of that schema for a
                                          DIFFERENT model (step 3): `matador
                                          export-model-json` produces the same
                                          shape from any trained TMIR.
              sim/capacity.json            This bitstream's synthesized capacity
                                          ceiling (what a new model must fit
                                          inside — see step 3).
              sim/gen_vectors.py           Standalone tool: build vectors for a
                                          NEW model (step 3) or a multi-model
                                          reprogramming run (step 4).
              sim/verilator/               Verilator build (step 1).
              provenance.json (repo root)  Which trained model this bundle
                                          currently encodes.

            If you only read one file before diving in, read `sim/test_vectors.txt`.

            ## 1. Compile & run

            Two independent simulators are shipped; either is enough to check
            this bundle. iverilog is the reference path (works everywhere);
            Verilator is faster and produces `.fst` waveforms.

            ```
            bash sim/run_iverilog.sh          # iverilog: all 5 testbenches
            make -C sim/verilator run          # Verilator: replays model_stimulus.memh
            ```

            Both check the SAME correctness contract for the model below —
            `tb_system_gp_model` (iverilog) and the Verilator harness both load it
            via CMD_LOAD and replay its embedded verification vectors, checking
            predictions against `tmir.verification.test_vectors` (the
            `AcceleratorSpec == RTL` leg of Matador's three-layer contract).
            `tb_system_gp` (iverilog only) is a generic protocol-conformance
            suite — LOAD/INFER framing, batching, error injection/recovery —
            independent of which model is loaded. `tb_tile_mem`/`tb_score_acc_rt`/
            `tb_argmax_rt` are per-module unit tests.

            The Verilator harness (`sim/verilator/tb_top.cpp`) is generic: it
            replays whatever `--stim`/`--exp` `.memh` pair it's given (default:
            this model's own vectors), so the same compiled binary also replays
            anything `gen_vectors.py` builds below — no rebuild needed to try a
            new model, only a re-run:

            ```
            sim/verilator/obj_dir/Vtm_accel_gp --stim my_stim.memh --exp my_expected.memh
            ```

            `matador simulate --backend vanilla_gp_tiled --config ...` runs both
            paths automatically and reports PASS/FAIL per testbench.

            ## 2. View waveforms, and check predicted vs. expected classes

            ```
            bash sim/waves.sh
            ```

            Opens `tb_system_gp_model.vcd` with a curated signal layout
            (`tb_system_gp_model.gtkw`) instead of GTKWave's default unsorted
            signal tree, grouped as:

              AXI-Stream Input     s_tvalid/s_tready/s_tdata/s_tlast — the LOAD
                                   packet, then the feature vectors, streaming in
              AXI-Stream Output    m_tvalid/m_tready/m_tdata/m_tlast — the
                                   PREDICTED CLASS appears in m_tdata every time
                                   m_tvalid pulses
              Status               busy, configured, dut.state (the FSM state)
              Testbench bookkeeping  e (the beat index), sent, fail_cnt

            **To check predicted vs. expected:** open `sim/test_vectors.txt`
            alongside the waveform. Scrub to each `m_tvalid` pulse in the
            "AXI-Stream Output" group and read `m_tdata`:
              - the FIRST pulse is the LOAD ack (`m_tdata` = `0xA5000xxx`) — not
                a prediction, skip it
              - the 2nd pulse is `test_vectors.txt` row 0's prediction, the 3rd
                is row 1's, and so on in order — compare each `m_tdata` against
                that row's `expected_class` column
            `e` (in "Testbench bookkeeping") is the same index `run_iverilog.sh`
            prints as `PASS beat[e]`/`FAIL beat[e]` to the console, so console
            output, waveform, and `test_vectors.txt` all line up: `e=0` is the
            LOAD ack, `e=k+1` is `test_vectors.txt` row `k`.

            ```
            bash sim/waves.sh                    # prefers the Verilator .fst if present, else the iverilog .vcd
            bash sim/waves.sh verilator           # explicitly open sim/verilator/tb_top.fst (no curated layout yet)
            bash sim/waves.sh tb_system_gp        # explicitly open a specific iverilog testbench's .vcd
            bash sim/waves.sh tb_reprogram_suite  # the multi-model reprogramming run, once built -- see § 4
            ```

            ## 3. Try a new model without resynthesizing

            This core is runtime-reprogrammable: swap in a different model with a
            fresh CMD_LOAD packet, no rebuild needed, as long as it fits the
            capacity in the reference table below. `sim/gen_vectors.py` is a
            standalone tool (only needs `sim/tm_emulator.py` alongside it — no
            matador install) that builds CMD_LOAD/CMD_INFER `.memh` streams (and,
            with `--testbench`, a ready-to-compile Verilog testbench) for any
            model you describe in its JSON schema. `sim/model_reference.json` is
            a working example — the exact model this bundle's own
            `model_stimulus.memh` was built from — and `sim/capacity.json` is
            what it reads to validate a new model against *this* bitstream's
            actual synthesized capacity, not generic defaults.

            **Getting a NEW model's JSON:** `gen_vectors.py` deliberately has no
            matador dependency, so it can't read a trained TMIR file directly
            (its `tm_emulator.py::load_tmir()` is an unimplemented stub, by
            design — this tool isn't meant to know matador's internal formats).
            Whoever trained the new model (i.e. has matador installed) runs this
            once and hands you the resulting JSON — no RTL regeneration involved,
            just a format conversion:

            ```
            matador export-model-json --backend vanilla_gp_tiled \\
                --model path/to/TM_TMIR_....yaml -o my_model.json
            ```

            With that JSON in hand (verified end to end: exported this way, then
            built and simulated from a completely standalone copy of this `sim/`
            directory with no matador on the Python path — it works):

            ```
            cd sim
            python3 gen_vectors.py combined my_model.json --random -n 20 \\
                -o my_stim.memh --expected my_expected.memh --testbench tb_my_model.v
            iverilog -g2001 -Wall -Wno-timescale -o tb_my_model \\
                ../src/axis_fifo.v ../src/clause_eval.v ../src/tile_mem.v \\
                ../src/score_acc_rt.v ../src/argmax_rt.v ../src/tm_accel_gp.v \\
                tb_my_model.v && vvp tb_my_model
            ```

            ## 4. Prove reprogrammability with your own models

            One model at a time only shows this core *can* be reprogrammed in
            principle. To prove it end to end with models you actually wrote,
            chain several into one continuous LOAD → INFER → LOAD → INFER → ...
            run.

            **If you have matador installed** (the common case — matador is what
            built this bundle), the CLI-native path builds and runs it for you,
            writing `tb/tb_reprogram_suite.v` and
            `sim/reprogram_stimulus.memh`/`reprogram_expected.memh`/
            `reprogram_manifest.txt` into *this* bundle alongside the
            single-model artifacts above:

            ```
            matador reprogram-suite --backend vanilla_gp_tiled \\
                --config vanilla_gp_tiled.yaml \\
                --reprogram-config reprogram_config.yaml
            ```

            where `reprogram_config.yaml` lists an ordered `steps:` sequence,
            each a trained TMIR model plus (optionally) a dataset or vector
            file to draw test vectors from — omit both to fall back to that
            model's own embedded test vectors (the most independent check:
            those expected classes come from matador's separate mathematical
            reference model, not from any code this bundle's RTL was built
            with). See `docs/Usage.md` § Step 8 in the matador repo for the
            full config schema.

            **Without matador installed** (just this exported bundle),
            `gen_vectors.py sequence` builds the same kind of multi-step
            stream standalone:

            ```
            cd sim   # if not already there
            python3 gen_vectors.py sequence my_manifest.json \\
                -o seq_stim.memh --expected seq_exp.memh --testbench tb_sequence.v
            ```

            where `my_manifest.json` is a JSON list of `{{"model": ..., "vectors":
            ...}}` (or `{{"model": ..., "random": true, "n": N}}`) steps — see
            `sim/gen_vectors.py --help` / its module docstring for the exact
            schema.

            Either way, the resulting testbench streams a LOAD ack + predictions
            for *each* step in order, so a PASS there is a direct, checkable
            demonstration that the live hardware reprograms correctly across your
            own models, not just the two hardcoded demo models `tb_system_gp`
            ships with.

            **Viewing it:** `sim/reprogram_manifest.txt` (or the equivalent
            `my_manifest.json`-derived one from `gen_vectors.py`) maps each beat
            index to which {{model, vector}} it belongs to — same `e` numbering
            as the console's `PASS beat[e]`/`FAIL beat[e]` and the waveform, per
            § 2 above:

            ```
            bash sim/waves.sh tb_reprogram_suite
            ```

            For Verilator, the same `sim/verilator/obj_dir/Vtm_accel_gp` binary
            from § 1 replays this too — point it at the reprogram-suite vectors
            with `--stim`/`--exp`:

            ```
            make -C sim/verilator run ARGS="--stim ../reprogram_stimulus.memh --exp ../reprogram_expected.memh"
            ```

            `make run` always writes to the same `sim/verilator/tb_top.fst`
            regardless of which `--stim`/`--exp` you passed — running this
            overwrites whatever trace (e.g. the single-model one from § 2) was
            there before. Copy `tb_top.fst` aside first if you want to keep both.

            ## 5. Provenance

            `provenance.json` (repo root) identifies which trained model's
            weights are packed into `model_stimulus.memh` right now — model
            fingerprint, geometry, and training provenance if the source TMIR had
            it. It describes the model at generate time only; if you replay a
            different model's CMD_LOAD against the live hardware (steps 3-4
            above), it goes stale for that hardware state — there's no on-chip
            readback to re-verify what's actually loaded, this is a software
            record, not a hardware memory-integrity check.

            ## Reference: capacity & model (this synthesis)

              target_fpga         = {config.target_fpga}
              tile_mem bits        = {config.required_bram_bits:,} / {budget_bits:,} budget ({pct:.1f}%)

              feat_slice           = {config.feat_slice}   (features/cycle)
              clause_slice         = {config.clause_slice}   (clauses/cycle)
              max_features         = {config.max_features}
              max_clauses_total    = {config.max_clauses_total}
              max_classes          = {config.max_classes}
              max_feat_slices      = {config.max_feat_slices}  (derived)
              max_clause_slices    = {config.max_clause_slices}  (derived)
              fifo_depth           = {config.fifo_depth}

              model loaded for simulation: {model.name}
              n_features           = {model.n_features}
              n_classes            = {model.n_classes}
              clauses_per_class    = {model.clauses_per_class}
              threshold            = {model.threshold}  (vestigial -- scores are unclamped, see score_acc_rt.v)
              n_tiles              = {model.n_tiles}
            """)
