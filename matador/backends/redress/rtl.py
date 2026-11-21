"""REDRESS backend RTL generator.

Wraps redress_accel.v — MUSTANG's Single-Core configuration (Chapter 5.3 of
the source thesis): a runtime-reprogrammable decoder for the 16-bit
Include-Encoded (Chapter 4.3) instruction stream codec.py produces. Like
vanilla_gp_tiled, generate() does not bake model weights into freshly
generated RTL: the datapath sources are static (only capacity parameters
vary), and the specific trained model is delivered separately as an
AXI-Stream Instruction-stream packet, generated here as a .memh file. Unlike
gp_tiled's dense tile_mem, the loaded representation here is the sparse
MUSTANG instruction stream itself (see codec.py) — Instruction Memory holds
one 16-bit word per included literal, nothing for excludes or empty clauses.

generate() writes:
  RTL/src/*.v             — capacity-parameterized static datapath
  RTL/tb/tb_redress_model.v — testbench that loads *this* TMIR model and
                             replays its embedded verification.test_vectors,
                             batch_width datapoints at a time
  RTL/sim/*.memh           — stimulus/expected vectors for that testbench
  RTL/sim/test_vectors.txt — human-readable companion to model_stimulus.memh
  RTL/sim/tb_redress_model.gtkw
                            — curated GTKWave layout; waves.sh uses it
  RTL/sim/run_iverilog.sh, waves.sh
  RTL/sim/verilator/{Makefile,tb_top.cpp}
                            — Verilator harness (static, no per-config templating:
                            it replays whatever --stim/--exp .memh it's given at
                            runtime, so the same binary works for this model, a
                            brand-new one, or a reprogram-suite run)
  RTL/provenance.json      — which trained model's weights model_stimulus.memh
                             currently encodes
  RTL/README.md
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import numpy as np

from matador.backends.base import ResourceEstimate, RTLArtifacts, RTLBackend

_SRC_DIR = Path(__file__).parent / "src"
_VERILATOR_DIR = Path(__file__).parent / "verilator"
_COMMON_DIR = Path(__file__).parent.parent / "common" / "src"
_STATIC_SOURCES = ("redress_bram.v", "score_acc_rt.v", "argmax_rt.v")
# Sourced from matador/backends/common/src/ (shared across backends) rather
# than this backend's own src/ — see common/README.md.
_COMMON_SOURCES = {"argmax_rt.v"}
# The top-module name as it appears in the static Verilator files
# (verilator/Makefile, verilator/tb_top.cpp). Subclasses with a different top
# module get it substituted when those files are copied into a bundle.
_VERILATOR_TOP_TOKEN = "redress_accel"


def _clog2(n: int) -> int:
    """Number of bits needed to represent 0..n-1 (matches $clog2 semantics)."""
    return max(1, (max(n, 1) - 1).bit_length())


def _sub_param(text: str, name: str, value: int, fname: str = "redress_accel.v") -> str:
    pattern = re.compile(rf"(parameter\s+{name}\s*=\s*)\d+")
    new_text, count = pattern.subn(rf"\g<1>{value}", text, count=1)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one 'parameter {name} = ...' in {fname}, "
            f"found {count}"
        )
    return new_text


def _write_memh(path: Path, words: list[int], lasts: list[int]) -> None:
    """One 33-bit hex word per line: bit32 = tlast, bits[31:0] = data —
    matches $readmemh's expectation for a `reg [32:0]` array, the same
    convention gp_tiled's vendored tm_emulator.write_memh() uses."""
    lines = [f"{(tlast << 32) | data:09X}" for data, tlast in zip(words, lasts)]
    path.write_text("\n".join(lines) + "\n")


class RedressBackend(RTLBackend):
    """MUSTANG Single-Core accelerator (see codec.py's module docstring for
    the full REDRESS/MUSTANG derivation this implements).

    One synthesis, sized by RedressAcceleratorConfig's capacity fields, can
    be reprogrammed at runtime with any per_class TM model whose MUSTANG
    encoding fits inside that capacity — no resynthesis needed. Tuning
    knobs:

      target_fpga, bram_bits_budget
        — the real capacity constraint: Instruction Memory + Feature FIFO
          must fit the device's BRAM
      max_includes
        — Instruction Memory depth in 16-bit words (DATA-dependent: the
          compressed size of a model, not derivable from geometry alone)
      max_features, max_classes
        — compile-time capacity, in real units (max_features is further
          bounded by MUSTANG's 12-bit offset field, codec.MUSTANG_MAX_FEATURES)
      batch_width
        — datapoints classified per pass through the instruction stream
          (Algorithm 12's W); also the parallel AND-accumulate lane count
      fifo_depth
        — currently unused by redress_accel.v (no separate input FIFO —
          Instruction Memory / Feature FIFO ARE the buffering); kept on the
          config for symmetry with gp_tiled and forward compatibility.

    Extension hooks (for subclasses such as a hardened-decoder variant):

      _src_dir          where the top module (and any overridden submodule) lives
      _verilator_dir    where the static Verilator Makefile/tb_top.cpp live
      _top_module       top module name; its file is <_src_dir>/<_top_module>.v
      _static_sources   submodules copied verbatim into every bundle
      encode_stream()         TMIR -> instruction words (default: mustang_encode)
      expected_predictions()  words + one packed batch -> expected classes
      _extra_params()         extra `parameter NAME = value` substitutions
    """

    _src_dir: Path = _SRC_DIR
    _verilator_dir: Path = _VERILATOR_DIR
    _top_module: str = "redress_accel"
    _static_sources: tuple[str, ...] = _STATIC_SOURCES

    @property
    def name(self) -> str:
        return "redress"

    @property
    def config_class(self) -> type:
        from matador.backends.redress.config import RedressAcceleratorConfig
        return RedressAcceleratorConfig

    @property
    def emulator_class(self):
        from matador.backends.redress.emulator import RedressEmulator
        return RedressEmulator

    @property
    def supports_reprogramming(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # generate()
    # ------------------------------------------------------------------

    def generate(self, tmir, config) -> RTLArtifacts:
        from matador.backends.redress import codec

        E_words = self._check_capacity(tmir, config)

        rtl_dir = config.output_dir / "RTL"
        src_dir = rtl_dir / "src"
        tb_dir = rtl_dir / "tb"
        sim_dir = rtl_dir / "sim"
        for d in (src_dir, tb_dir, sim_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── sources: static submodules verbatim, redress_accel.v capacity-
        #    parameterized ───────────────────────────────────────────────
        sources = []
        for fname in self._static_sources:
            dst = src_dir / fname
            dst.write_text(self._static_source_path(fname).read_text())
            sources.append(dst)

        top_path = src_dir / f"{self._top_module}.v"
        top_path.write_text(self._render_top(config))
        sources.append(top_path)

        # ── model-specific testbench: one Instruction-stream load, then
        #    this model's embedded verification vectors replayed in
        #    batch_width-sized Feature-stream packets ────────────────────
        stim_words, stim_lasts, exp_words, exp_lasts, n_batches = self._build_stimulus(
            tmir, config, E_words
        )

        stim_path = sim_dir / "model_stimulus.memh"
        exp_path = sim_dir / "model_expected.memh"
        _write_memh(stim_path, stim_words, stim_lasts)
        _write_memh(exp_path, exp_words, exp_lasts)

        tb_path = tb_dir / "tb_redress_model.v"
        tb_path.write_text(self._render_model_testbench(
            config, n_stim=len(stim_words), n_exp=len(exp_words),
            n_instr=len(E_words), n_batches=n_batches,
        ))

        sim_files = [stim_path, exp_path]
        testbenches = [tb_path]

        test_vectors_path = sim_dir / "test_vectors.txt"
        test_vectors_path.write_text(self._gen_test_vectors_txt(tmir, config, E_words))
        sim_files.append(test_vectors_path)

        gtkw_path = sim_dir / "tb_redress_model.gtkw"
        gtkw_path.write_text(self._gen_tb_gtkw("tb_redress_model"))
        sim_files.append(gtkw_path)

        provenance_path = rtl_dir / "provenance.json"
        provenance_path.write_text(self._gen_provenance(tmir, config, E_words))
        sim_files.append(provenance_path)

        # ── Verilator harness: static, no per-config templating needed (it
        #    replays whatever --stim/--exp .memh it's given at runtime; see
        #    verilator/tb_top.cpp's header). matador.models.validator.
        #    validate_rtl picks this up automatically once sim/verilator/
        #    Makefile exists.
        verilator_dir = sim_dir / "verilator"
        verilator_dir.mkdir(exist_ok=True)
        for fname in ("Makefile", "tb_top.cpp"):
            dst = verilator_dir / fname
            dst.write_text(self._render_verilator_file(fname))
            sim_files.append(dst)

        readme_path = rtl_dir / "README.md"
        readme_path.write_text(self._gen_readme(config, tmir, E_words))

        for fname, content in self._gen_sim_scripts().items():
            p = sim_dir / fname
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

    # ------------------------------------------------------------------
    # build_reprogram_suite()
    # ------------------------------------------------------------------

    def build_reprogram_suite(self, rtl_dir: Path, steps, config) -> RTLArtifacts:
        """Build a testbench + stimulus that reprograms an ALREADY-GENERATED
        bundle (rtl_dir) across multiple {model, dataset} steps in one
        continuous Instruction-load -> Feature-batches -> Instruction-load
        -> ... run. redress_accel.v accepts a fresh Instruction-stream
        header from S_IDLE at any time (the same state a Feature-stream
        header is accepted from), so reprogramming mid-run needs no special
        handling beyond concatenating each step's own stimulus in order —
        the same principle gp_tiled's build_reprogram_suite uses, just with
        a per-step instruction reload instead of a per-step CMD_LOAD.
        """
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

        all_stim_words: list[int] = []
        all_stim_lasts: list[int] = []
        all_exp_words: list[int] = []
        all_exp_lasts: list[int] = []
        step_reports = []
        beat = 0

        for i, step in enumerate(steps):
            suffix = step.tmir_path.suffix.lower()
            tmir = TMIR.from_yaml(step.tmir_path) if suffix in {".yaml", ".yml"} else TMIR.from_npz(step.tmir_path)
            E_words = self._check_capacity(tmir, config)
            model_name = step.name or step.tmir_path.stem

            X = self._resolve_step_vectors(step, tmir)
            stim_words, stim_lasts, exp_words, exp_lasts, n_batches = self._build_stimulus(
                tmir, config, E_words, X=X
            )
            all_stim_words.extend(stim_words)
            all_stim_lasts.extend(stim_lasts)
            all_exp_words.extend(exp_words)
            all_exp_lasts.extend(exp_lasts)

            step_reports.append({
                "index": i, "model": model_name, "tmir_path": str(step.tmir_path),
                "dataset": str(step.dataset_path) if step.dataset_path else None,
                "vectors": str(step.vectors_path) if step.vectors_path else None,
                "n_instr": len(E_words), "n_vectors": X.shape[0], "n_batches": n_batches,
                "first_beat": beat,
            })
            beat += len(stim_words)

        tb_path = tb_dir / "tb_reprogram_suite.v"
        tb_path.write_text(self._render_model_testbench(
            config, n_stim=len(all_stim_words), n_exp=len(all_exp_words),
            n_instr=max((s["n_instr"] for s in step_reports), default=0),
            n_batches=sum(s["n_batches"] for s in step_reports),
            tb_name="tb_reprogram_suite",
            stim_file="reprogram_stimulus.memh", exp_file="reprogram_expected.memh",
            header_comment=(
                "// Multi-model/dataset reprogramming suite (matador reprogram-suite).\n"
                "    // Reprograms this ALREADY-SYNTHESIZED core across several models in one\n"
                "    // continuous Instruction-load -> Feature-batches -> ... run -- see\n"
                "    // sim/reprogram_manifest.txt for which beat belongs to which step."
            ),
        ))

        stim_path = sim_dir / "reprogram_stimulus.memh"
        exp_path = sim_dir / "reprogram_expected.memh"
        _write_memh(stim_path, all_stim_words, all_stim_lasts)
        _write_memh(exp_path, all_exp_words, all_exp_lasts)

        manifest_path = sim_dir / "reprogram_manifest.txt"
        manifest_path.write_text(self._gen_reprogram_manifest_txt(step_reports))

        gtkw_path = sim_dir / "tb_reprogram_suite.gtkw"
        gtkw_path.write_text(self._gen_tb_gtkw("tb_reprogram_suite"))

        return RTLArtifacts(
            rtl_dir=rtl_dir,
            sources=[],
            testbenches=[tb_path],
            sim_scripts=[stim_path, exp_path, manifest_path, gtkw_path],
        )

    def _resolve_step_vectors(self, step, tmir) -> np.ndarray:
        """(n_samples, n_features) uint8 matrix for one step, per the
        precedence documented on ReprogramStep: vectors_path > dataset_path
        > this TMIR's own embedded verification.test_vectors."""
        n_features = tmir.architecture.n_features
        if step.vectors_path:
            return self._read_vector_file(step.vectors_path, n_features)
        if step.dataset_path:
            return self._read_dataset_features(step.dataset_path, n_features, step.n_samples, step.seed)
        test_vectors = list(tmir.verification.test_vectors) if tmir.verification else []
        if not test_vectors:
            return np.zeros((0, n_features), dtype=np.uint8)
        return np.array([tv.input for tv in test_vectors], dtype=np.uint8)

    def _read_vector_file(self, path: Path, n_features: int) -> np.ndarray:
        rows = []
        for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            bits = line.replace(",", " ").split()
            if len(bits) != n_features:
                raise ValueError(f"{path}:{lineno}: expected {n_features} bits, got {len(bits)}")
            rows.append([int(b) & 1 for b in bits])
        return np.array(rows, dtype=np.uint8) if rows else np.zeros((0, n_features), dtype=np.uint8)

    def _read_dataset_features(self, path: Path, n_features: int, n_samples, seed: int) -> np.ndarray:
        """A booleanized *_test.txt-shaped file (space-separated uint, last
        column = label — matador.models.trainer's own format). The label
        column is dropped: reprogram-suite always checks RTL against the
        model's OWN prediction (codec.mustang_decode_and_infer), not the
        dataset's ground-truth label."""
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
        return X.astype(np.uint8)

    def _gen_reprogram_manifest_txt(self, step_reports: list[dict]) -> str:
        lines = [
            "Multi-model/dataset reprogramming suite -- one continuous",
            "Instruction-load -> Feature-batches -> Instruction-load -> ... stream",
            "across the steps below, in order, packed into",
            "reprogram_stimulus.memh / reprogram_expected.memh and replayed by",
            "tb_reprogram_suite.v (iverilog).",
            "",
            "HOW TO READ THIS AGAINST THE WAVEFORM/CONSOLE OUTPUT:",
            "  Each step contributes n_instr instruction-load beats (no output),",
            "  then n_batches x batch_width prediction beats (batch_width output",
            "  beats per batch, including any zero-padded lanes beyond the real",
            "  vector count in that batch's last, possibly-partial batch).",
            "",
        ]
        for step in step_reports:
            lines.append(f"step {step['index']}: model={step['model']!r}  tmir={step['tmir_path']}")
            if step["dataset"]:
                lines.append(f"  vectors from dataset: {step['dataset']}")
            elif step["vectors"]:
                lines.append(f"  vectors from file: {step['vectors']}")
            else:
                lines.append("  vectors from: this model's own embedded verification.test_vectors")
            lines.append(
                f"  n_instr={step['n_instr']}  n_vectors={step['n_vectors']}  "
                f"n_batches={step['n_batches']}  first stimulus beat={step['first_beat']}"
            )
            lines.append("")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # capacity check (data-dependent: must actually encode the model)
    # ------------------------------------------------------------------

    def _check_capacity(self, tmir, config) -> list[int]:
        """Encode tmir via MUSTANG and verify it fits this config's
        compile-time capacity AND the wire protocol's own field widths.
        Returns the encoded instruction stream (callers reuse it rather
        than re-encoding)."""
        # MustangEmptyClassError is itself a ValueError (see codec.py) with
        # a clear, complete message -- let it propagate as-is rather than
        # wrapping it, so callers that want to distinguish "format can't
        # represent this model" from "model doesn't fit this config's
        # capacity" still can.
        E_words = self.encode_stream(tmir, config)

        arch = tmir.architecture
        if len(E_words) > config.max_includes:
            raise ValueError(
                f"model encodes to {len(E_words):,} MUSTANG instruction words, "
                f"exceeding this config's max_includes={config.max_includes:,}. "
                "Raise max_includes (and re-check the BRAM budget), or use a "
                "sparser/smaller model."
            )
        if arch.n_features > config.max_features:
            raise ValueError(
                f"model has n_features={arch.n_features}, exceeding this "
                f"config's max_features={config.max_features}."
            )
        if arch.n_classes > config.max_classes:
            raise ValueError(
                f"model has n_classes={arch.n_classes}, exceeding this "
                f"config's max_classes={config.max_classes}."
            )
        if arch.n_classes > 255:
            raise ValueError(
                f"model has n_classes={arch.n_classes}, exceeding the "
                "Instruction Header's 8-bit n_classes field (bits[29:22], max 255)."
            )
        if len(E_words) > (1 << 22):
            raise ValueError(
                f"model encodes to {len(E_words):,} instruction words, exceeding "
                "the Instruction Header's 22-bit n_instructions field (max 4,194,304)."
            )
        return E_words

    def resource_estimate(self, tmir, config) -> ResourceEstimate:
        from matador.backends.fpga_budget import bram_budget_bits
        from matador.backends.redress.config import REDRESS_BRAM_FRACTION

        bram_bits = config.required_bram_bits
        budget_bits = bram_budget_bits(config.target_fpga, config.bram_bits_budget, REDRESS_BRAM_FRACTION)
        pct = 100.0 * bram_bits / budget_bits if budget_bits else 0.0
        return ResourceEstimate(
            bram_bits=bram_bits,
            notes=(
                f"Instruction Memory: {config.max_includes:,} x 16-bit words = "
                f"{config.instr_mem_bits:,} bits. Feature FIFO: {config.max_features:,} x "
                f"{config.batch_width}-bit words = {config.feature_fifo_bits:,} bits. "
                f"Total {bram_bits:,} bits ({pct:.1f}% of the {budget_bits:,}-bit budget "
                f"for target_fpga={config.target_fpga!r}). Runtime-reprogrammable: this "
                "capacity is fixed at synthesis time; any model whose MUSTANG-encoded "
                "size (codec.mustang_encode) fits max_includes, and whose geometry fits "
                "max_features/max_classes, loads without resynthesis. Run Vivado "
                "synthesis for exact LUT/BRAM numbers."
            ),
        )

    # ------------------------------------------------------------------
    # RTL rendering
    # ------------------------------------------------------------------

    def _render_top(self, config) -> str:
        fname = f"{self._top_module}.v"
        text = (self._src_dir / fname).read_text()
        for pname, value in self._capacity_params(config):
            text = _sub_param(text, pname, value, fname)
        return text

    def _static_source_path(self, fname: str) -> Path:
        """Where a static submodule is copied from: common/src for shared
        modules, else this backend's _src_dir if it has its own copy, else
        redress's src/ (so a subclass only ships the files it changes)."""
        if fname in _COMMON_SOURCES:
            return _COMMON_DIR / fname
        own = self._src_dir / fname
        return own if own.exists() else _SRC_DIR / fname

    def _render_verilator_file(self, fname: str) -> str:
        """A static Verilator file with the top-module name substituted.
        For redress itself this is an identity copy."""
        text = (self._verilator_dir / fname).read_text()
        return text.replace(_VERILATOR_TOP_TOKEN, self._top_module)

    # ------------------------------------------------------------------
    # Extension hooks (see the class docstring)
    # ------------------------------------------------------------------

    def encode_stream(self, tmir, config) -> list[int]:
        """TMIR -> the instruction words loaded into Instruction Memory."""
        from matador.backends.redress import codec
        return codec.mustang_encode(tmir)

    def expected_predictions(self, words, n_classes: int, X_pack, config) -> list[int]:
        """Expected predicted class for every one of the batch_width lanes of
        one packed Feature batch -- what the testbench checks the RTL against."""
        from matador.backends.redress import codec
        best_class, _ = codec.mustang_decode_and_infer(words, n_classes, X_pack, config.batch_width)
        return best_class

    def _extra_params(self, config) -> list[tuple[str, int]]:
        """Additional `parameter NAME = value` substitutions for the top
        module, applied after the capacity parameters. redress has none."""
        return []

    def _capacity_params(self, config) -> list[tuple[str, int]]:
        """Every `parameter NAME = ...` in redress_accel.v that must be
        re-derived from this config, in substitution order. Shared by
        _render_top() and _render_model_testbench()'s DUT instantiation so
        the two can never drift apart."""
        # SCORE_WIDTH: redress has no explicit max-clauses-per-class field
        # (capacity is expressed in max_includes, a word count, not a clause
        # count — see config.py's docstring). The true worst case is every
        # included word being its own single-literal clause, all one
        # polarity, all firing -- i.e. up to max_includes vote increments
        # landing on one class. Conservative but safe, same clog2(...)+2
        # sizing idiom as gp_tiled's _capacity_params().
        score_width = max(4, _clog2(config.max_includes + 1) + 2)
        return [
            ("AXIS_DATA_WIDTH", config.axis_data_width),
            ("MAX_CLASSES", config.max_classes),
            ("CLASS_WIDTH", _clog2(config.max_classes)),
            ("MAX_INCLUDES", config.max_includes),
            ("INSTR_ADDR_W", _clog2(config.max_includes)),
            ("MAX_FEATURES", config.max_features),
            ("FEAT_ADDR_W", _clog2(config.max_features)),
            ("BATCH_WIDTH", config.batch_width),
            ("OUT_IDX_W", _clog2(config.batch_width)),
            ("SCORE_WIDTH", score_width),
        ] + self._extra_params(config)

    def _build_stimulus(self, tmir, config, E_words: list[int], X: "np.ndarray | None" = None):
        """One Instruction-stream packet (this model's MUSTANG encoding),
        then this model's test vectors replayed batch_width datapoints at a
        time as Feature-stream packets. Expected outputs are computed
        directly via codec.mustang_decode_and_infer() for the FULL batch
        width each time (not codec.mustang_infer(), which truncates to the
        real sample count) -- the RTL always emits batch_width output beats
        per batch regardless of how many lanes held real data, and the
        zero-padded lanes' predictions are still fully deterministic, so
        checking them too is a real (if minor) extra assurance, not
        guesswork.
        """
        from matador.backends.redress import codec

        arch = tmir.architecture
        n_classes = arch.n_classes
        n_features = arch.n_features
        n_instr = len(E_words)
        W = config.batch_width

        if X is None:
            test_vectors = list(tmir.verification.test_vectors) if tmir.verification else []
            X = (
                np.array([tv.input for tv in test_vectors], dtype=np.uint8)
                if test_vectors else np.zeros((0, n_features), dtype=np.uint8)
            )

        stim_words: list[int] = []
        stim_lasts: list[int] = []
        exp_words: list[int] = []
        exp_lasts: list[int] = []

        hdr = (1 << 31) | (1 << 30) | (n_classes << 22) | (n_instr - 1 if n_instr else 0)
        stim_words.append(hdr)
        stim_lasts.append(0 if n_instr else 1)
        for i, w in enumerate(E_words):
            stim_words.append(w)
            stim_lasts.append(1 if i == n_instr - 1 else 0)

        n_samples = X.shape[0]
        n_batches = 0
        for start in range(0, n_samples, W):
            batch = X[start:start + W]
            X_pack = codec.pack_features(batch, W)

            hdr2 = (1 << 31) | (0 << 30) | (n_features - 1)
            stim_words.append(hdr2)
            stim_lasts.append(0)
            for i, w in enumerate(X_pack):
                stim_words.append(w)
                stim_lasts.append(1 if i == n_features - 1 else 0)

            best_class = self.expected_predictions(E_words, n_classes, X_pack, config)
            for j in range(W):
                exp_words.append(best_class[j])
                exp_lasts.append(1 if j == W - 1 else 0)
            n_batches += 1

        return stim_words, stim_lasts, exp_words, exp_lasts, n_batches

    def _render_model_testbench(
        self, config, n_stim: int, n_exp: int, n_instr: int, n_batches: int,
        tb_name: str = "tb_redress_model",
        stim_file: str = "model_stimulus.memh",
        exp_file: str = "model_expected.memh",
        header_comment: str = (
            "// Loads the trained TMIR model via one Instruction-stream packet\n"
            "    // (MUSTANG encoding), then replays its embedded\n"
            "    // verification.test_vectors in batch_width-sized Feature-stream\n"
            "    // packets, checking every predicted class -- including zero-padded\n"
            "    // lanes in a partial final batch -- against\n"
            "    // codec.mustang_decode_and_infer()'s own output."
        ),
    ) -> str:
        params = dict(self._capacity_params(config))
        dut_params = ", ".join(f".{name}({value})" for name, value in params.items())

        # Watchdog: n_stim beats to load, then n_batches passes each costing
        # up to ~4 cycles/instruction (S_INFER_INSTR/DECODE/FEAT/ACCUM) plus
        # batch_width output beats plus slack -- a generous per-batch upper
        # bound, not a tight one.
        cycles_per_batch_upper_bound = 4 * max(n_instr, 1) + config.batch_width + 32
        watchdog_cycles = n_stim + n_batches * cycles_per_batch_upper_bound + 32
        watchdog_ns = max(2_000_000, watchdog_cycles * 10 + 200_000)

        return textwrap.dedent(f"""\
            `timescale 1ns/1ps
            // Auto-generated by matador generate ({self.name} backend).
            {header_comment}
            module {tb_name};

                parameter AXIS_DATA_WIDTH = {config.axis_data_width};
                localparam N_STIM   = {n_stim};
                localparam N_EXP    = {n_exp};
                localparam WATCHDOG = {watchdog_ns};  // #time-units, not cycles

                reg                        clk, rst_n;
                reg  [AXIS_DATA_WIDTH-1:0] s_tdata;
                reg                        s_tvalid;
                wire                       s_tready;
                reg                        s_tlast;
                wire [AXIS_DATA_WIDTH-1:0] m_tdata;
                wire                       m_tvalid;
                reg                        m_tready;
                wire                       m_tlast;
                wire                       busy;

                {self._top_module} #(
                    {dut_params}
                ) dut (
                    .clk(clk), .rst_n(rst_n),
                    .s_axis_tdata(s_tdata), .s_axis_tvalid(s_tvalid),
                    .s_axis_tready(s_tready), .s_axis_tlast(s_tlast),
                    .m_axis_tdata(m_tdata), .m_axis_tvalid(m_tvalid),
                    .m_axis_tready(m_tready), .m_axis_tlast(m_tlast),
                    .busy(busy)
                );

                initial clk = 0;
                always #5 clk = ~clk;

                reg [32:0] stim [0:(N_STIM > 0 ? N_STIM : 1) - 1];
                reg [32:0] expc [0:(N_EXP > 0 ? N_EXP : 1) - 1];

                integer ei, pass_cnt, fail_cnt;

                // Synchronous AXI-Stream source: holds each beat until accepted,
                // then advances -- a plain procedural loop with blocking
                // assignments and no delay between iterations is NOT safe here
                // (found the hard way while developing this core): it races the
                // DUT's own registered tready/state transitions on the same
                // clock edge and can re-present or misalign a beat. This idiom
                // (a single always @(posedge clk) block, nonblocking
                // assignments, same clocking discipline as the DUT itself)
                // avoids that entirely.
                reg [31:0] wr_ptr;
                reg        started;

                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        wr_ptr   <= 0;
                        s_tvalid <= 1'b0;
                        s_tdata  <= {{AXIS_DATA_WIDTH{{1'b0}}}};
                        s_tlast  <= 1'b0;
                        started  <= 1'b0;
                    end else begin
                        if (!started) begin
                            started  <= 1'b1;
                            wr_ptr   <= 0;
                            s_tdata  <= stim[0][AXIS_DATA_WIDTH-1:0];
                            s_tlast  <= stim[0][32];
                            s_tvalid <= (N_STIM > 0);
                        end else if (s_tvalid && s_tready) begin
                            if (wr_ptr == N_STIM-1) begin
                                s_tvalid <= 1'b0;
                            end else begin
                                wr_ptr   <= wr_ptr + 1'b1;
                                s_tdata  <= stim[wr_ptr+1'b1][AXIS_DATA_WIDTH-1:0];
                                s_tlast  <= stim[wr_ptr+1'b1][32];
                                s_tvalid <= 1'b1;
                            end
                        end
                    end
                end

                initial begin
                    $dumpfile("{tb_name}.vcd");
                    $dumpvars(0, {tb_name});

                    if (N_STIM > 0) $readmemh("{stim_file}", stim);
                    if (N_EXP > 0)  $readmemh("{exp_file}", expc);

                    ei = 0; pass_cnt = 0; fail_cnt = 0;
                    rst_n = 1'b0; m_tready = 1'b1;
                    repeat (4) @(posedge clk);
                    rst_n = 1'b1;

                    if (N_EXP == 0) begin
                        repeat (N_STIM + 20) @(posedge clk);
                        $display("{tb_name}: no test vectors embedded -- instruction-load-only smoke check, N_STIM=%0d beats sent", N_STIM);
                        $finish;
                    end
                end

                // capture outputs concurrently -- checks BOTH the predicted
                // class (tdata) and tlast (must fire on exactly the last of
                // every batch_width output beats), the same conformance bar
                // vanilla_gp_tiled's testbenches hold their protocol to.
                always @(posedge clk) begin
                    if (N_EXP > 0 && m_tvalid && m_tready) begin
                        if (m_tdata == expc[ei][AXIS_DATA_WIDTH-1:0] && m_tlast == expc[ei][32]) begin
                            pass_cnt = pass_cnt + 1;
                        end else begin
                            $display("FAIL out[%0d]: expected data=%0d last=%0d  got data=%0d last=%0d",
                                      ei, expc[ei][AXIS_DATA_WIDTH-1:0], expc[ei][32], m_tdata, m_tlast);
                            fail_cnt = fail_cnt + 1;
                        end
                        ei = ei + 1;
                        if (ei == N_EXP) begin
                            if (fail_cnt == 0)
                                $display("{tb_name}: ALL %0d TESTS PASSED", N_EXP);
                            else
                                $display("{tb_name}: FAILED %0d/%0d", fail_cnt, N_EXP);
                            $finish;
                        end
                    end
                end

                initial begin
                    #WATCHDOG;
                    $display("TIMEOUT: got %0d/%0d results", ei, N_EXP);
                    $finish;
                end

            endmodule
            """)

    def _gen_test_vectors_txt(self, tmir, config, E_words: list[int]) -> str:
        vecs = list(tmir.verification.test_vectors) if tmir.verification else []
        arch = tmir.architecture
        lines = [
            "Test vectors embedded in this bundle's model, from the source TMIR's",
            "verification.test_vectors. These are the EXACT vectors packed into",
            "model_stimulus.memh (as batch_width-sized Feature-stream packets) and",
            "checked against model_expected.memh -- tb_redress_model.v replays these",
            "to prove AcceleratorSpec.simulate(x) == RTL_simulation(x).",
            "",
            "HOW TO READ THIS AGAINST THE WAVEFORM (bash sim/waves.sh):",
            "  Watch m_axis_tdata every time m_axis_tvalid fires -- that's the",
            "  PREDICTED CLASS, in the low bits. There is no LOAD-ack beat (unlike",
            "  vanilla_gp_tiled) -- the first batch_width output beats are",
            "  vectors[0:batch_width]'s predictions in order, one output beat per",
            "  input lane, including any zero-padded lanes beyond the real vector",
            "  count in the final (possibly partial) batch.",
            "",
            f"model: {tmir.provenance.dataset_id if tmir.provenance else '(unknown)'}  "
            f"({arch.n_features} features, {arch.n_classes} classes, "
            f"{arch.n_clauses_per_class} clauses/class, {len(E_words)} encoded instruction words)",
            f"batch_width = {config.batch_width}",
            f"{len(vecs)} test vector(s):",
            "",
            f"{'idx':>4}  {'expected_class':>14}  input (bit 0 = feature 0, left-to-right)",
        ]
        for i, tv in enumerate(vecs):
            bits = "".join(str(int(b) & 1) for b in tv.input)
            lines.append(f"{i:>4}  {tv.expected_class:>14}  {bits}")
        if not vecs:
            lines.append("(none embedded -- this TMIR model has no verification.test_vectors,")
            lines.append(" so model_stimulus.memh only contains the Instruction-stream packet,")
            lines.append(" no Feature-stream packets)")
        return "\n".join(lines) + "\n"

    def _gen_tb_gtkw(self, tb_name: str) -> str:
        """Curated GTKWave layout for <tb_name>.vcd, grouping signals so the
        AXI-Stream output (where the predicted class appears) is immediately
        obvious rather than buried in an unsorted signal tree."""
        from matador.waves.gtkwave import _BIN, _DEC, _HEX, _blank, _group, _group_end, _header, _sig

        T = tb_name
        lines = [_header(f"{T}.vcd")]

        lines.append(_group("AXI-Stream Input (Instruction packet, then Feature-batch packets)"))
        lines.append(_sig(f"{T}.s_tvalid", _BIN))
        lines.append(_sig(f"{T}.s_tready", _BIN))
        lines.append(_sig(f"{T}.s_tdata", _HEX))
        lines.append(_sig(f"{T}.s_tlast", _BIN))
        lines.append(_group_end("AXI-Stream Input (Instruction packet, then Feature-batch packets)"))

        lines.append(_blank())
        lines.append(_group("AXI-Stream Output (PREDICTED CLASS appears in m_tdata here)"))
        lines.append(_sig(f"{T}.m_tvalid", _BIN))
        lines.append(_sig(f"{T}.m_tready", _BIN))
        lines.append(_sig(f"{T}.m_tdata", _HEX))
        lines.append(_sig(f"{T}.m_tlast", _BIN))
        lines.append(_group_end("AXI-Stream Output (PREDICTED CLASS appears in m_tdata here)"))

        lines.append(_blank())
        lines.append(_group("Status"))
        lines.append(_sig(f"{T}.dut.state", _HEX))
        lines.append(_sig(f"{T}.dut.current_class", _DEC))
        lines.append(_group_end("Status"))

        lines.append(_blank())
        lines.append(_group("Testbench bookkeeping"))
        lines.append(_sig(f"{T}.ei", _DEC))
        lines.append(_sig(f"{T}.pass_cnt", _DEC))
        lines.append(_sig(f"{T}.fail_cnt", _DEC))
        lines.append(_group_end("Testbench bookkeeping"))

        return "".join(lines)

    def _gen_provenance(self, tmir, config, E_words: list[int]) -> str:
        prov = tmir.provenance
        arch = tmir.architecture
        payload = {
            "matador_backend": self.name,
            "model_fingerprint": tmir.fingerprint(),
            "architecture": {
                "n_features": arch.n_features,
                "n_classes": arch.n_classes,
                "n_clauses_per_class": arch.n_clauses_per_class,
            },
            "mustang_encoding": {
                "n_instruction_words": len(E_words),
                "bits": len(E_words) * 16,
            },
            "synthesized_capacity": {
                "target_fpga": config.target_fpga,
                "max_includes": config.max_includes,
                "max_features": config.max_features,
                "max_classes": config.max_classes,
                "batch_width": config.batch_width,
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
                "tmir.provenance was not set on the source TMIR model -- "
                "training_provenance is unavailable, but model_fingerprint "
                "above still uniquely identifies this exact model."
            )
        return json.dumps(payload, indent=2) + "\n"

    def _gen_sim_scripts(self) -> dict[str, str]:
        srcs = " ".join(f"$SRC_DIR/{f}" for f in (*self._static_sources, f"{self._top_module}.v"))
        return {
            "run_iverilog.sh": textwrap.dedent(f"""\
                #!/usr/bin/env bash
                # Portable: paths are resolved relative to this script's own location
                # at runtime, not baked in at generation time.
                set -e
                SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
                SRC_DIR="$SCRIPT_DIR/../src"
                TB_DIR="$SCRIPT_DIR/../tb"
                cd "$SCRIPT_DIR"
                SRCS="{srcs}"
                iverilog -g2001 -Wall -Wno-timescale -o tb_redress_model $SRCS $TB_DIR/tb_redress_model.v && vvp tb_redress_model
                if [ -f "$TB_DIR/tb_reprogram_suite.v" ]; then
                    iverilog -g2001 -Wall -Wno-timescale -o tb_reprogram_suite $SRCS $TB_DIR/tb_reprogram_suite.v && vvp tb_reprogram_suite
                fi
                """),
            "waves.sh": textwrap.dedent("""\
                #!/usr/bin/env bash
                # Usage: bash waves.sh [tb_name|verilator]
                #   tb_name    — an iverilog testbench's .vcd (default: tb_redress_model)
                #   verilator  — the Verilator harness's .fst (sim/verilator/tb_top.fst),
                #                produced by `make -C verilator run`
                # With no argument, prefers the Verilator trace if one exists (usually
                # the most recent/most relevant run), falling back to the default
                # iverilog testbench's .vcd otherwise.
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
                    WF="$SCRIPT_DIR/tb_redress_model.vcd"
                fi

                if [ ! -f "$WF" ]; then
                    echo "No waveform at $WF. Run simulation first:"
                    echo "  bash run_iverilog.sh          (produces .vcd)"
                    echo "  make -C verilator run          (produces verilator/tb_top.fst)"
                    exit 1
                fi

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

    def _gen_readme(self, config, tmir, E_words: list[int]) -> str:
        from matador.backends.fpga_budget import bram_budget_bits
        from matador.backends.redress.config import REDRESS_BRAM_FRACTION

        arch = tmir.architecture
        budget_bits = bram_budget_bits(config.target_fpga, config.bram_bits_budget, REDRESS_BRAM_FRACTION)
        pct = 100.0 * config.required_bram_bits / budget_bits if budget_bits else 0.0
        return textwrap.dedent(f"""\
            # redress — MUSTANG Single-Core accelerator

            A hardware translation of the REDRESS 16-bit Include-Encoding scheme
            (source thesis Chapter 4.3, Algorithms 11/12) -- MUSTANG's Single-Core
            configuration (Chapter 5.3, Figure 5.13). Unlike vanilla_tiled/
            vanilla_hardwired, this core is **not** regenerated per model: it is
            synthesized once at the capacity in the reference table below, then
            reprogrammed at runtime over `s_axis` with an Instruction-stream packet
            (see `src/{self._top_module}.v`'s header comment for the full AXI-Stream
            protocol). Unlike vanilla_gp_tiled's dense tile memory, the loaded
            representation is SPARSE: one 16-bit word per included literal, nothing
            for excludes or empty clauses -- see `matador.backends.redress.codec`'s
            module docstring for the full algorithm.

            ## 0. Orientation

              src/{self._top_module}.v          Top module -- read its header comment
                                          for the full AXI-Stream protocol and
                                          instruction word layout.
              src/redress_bram.v           Generic BRAM (Instruction Memory, Feature FIFO).
              src/score_acc_rt.v           Per-class vote accumulator (one per lane).
              src/argmax_rt.v              Per-lane argmax (one per lane).
              tb/tb_redress_model.v        Testbench built for *this* model (below).
              sim/test_vectors.txt         <- THE TEST VECTORS, human-readable.
              sim/model_stimulus.memh      Machine-readable stimulus -- what the RTL
              sim/model_expected.memh      testbench actually reads.
              sim/verilator/               Verilator build (step 1).
              provenance.json               Which trained model this bundle encodes.

            If you only read one file before diving in, read `sim/test_vectors.txt`.

            ## 1. Compile & run

            Two independent simulators are shipped; either is enough to check this
            bundle. iverilog is the reference path (works everywhere); Verilator is
            faster and produces `.fst` waveforms.

            ```
            bash sim/run_iverilog.sh          # iverilog
            make -C sim/verilator run          # Verilator: replays model_stimulus.memh
            ```

            Both check the SAME correctness contract: `tb_redress_model` (iverilog)
            and the Verilator harness both load this model via one Instruction-stream
            packet (MUSTANG encoding of {len(E_words):,} words), then replay its
            embedded verification vectors in batch_width={config.batch_width}-sized
            Feature-stream packets, checking every predicted class (and `m_axis_tlast`)
            against `codec.mustang_decode_and_infer()` -- the `AcceleratorSpec == RTL`
            leg of Matador's three-layer contract.

            The Verilator harness (`sim/verilator/tb_top.cpp`) is generic: it replays
            whatever `--stim`/`--exp` `.memh` pair it's given (default: this model's
            own vectors), so the same compiled binary also replays a reprogram-suite
            run (§ 3) -- no rebuild needed, only a re-run:

            ```
            sim/verilator/obj_dir/V{self._top_module} --stim my_stim.memh --exp my_expected.memh
            ```

            `matador simulate --backend {self.name} --config ...` runs both paths
            automatically and reports PASS/FAIL per testbench.

            ## 2. View waveforms

            ```
            bash sim/waves.sh                    # prefers the Verilator .fst if present, else the iverilog .vcd
            ```

            Opens `tb_redress_model.vcd` with a curated signal layout
            (`tb_redress_model.gtkw`): AXI-Stream Input, AXI-Stream Output (the
            predicted class appears in `m_tdata` every time `m_tvalid` pulses),
            Status (FSM state, current class being processed), and Testbench
            bookkeeping (`ei`, the beat index -- cross-reference against
            `sim/test_vectors.txt` row `ei / batch_width`, lane `ei % batch_width`).

            ## 3. Prove reprogrammability with your own models

            ```
            matador reprogram-suite --backend {self.name} \\
                --config redress.yaml \\
                --reprogram-config reprogram_config.yaml
            ```

            where `reprogram_config.yaml` lists an ordered `steps:` sequence, each a
            trained TMIR model plus (optionally) a dataset or vector file to draw
            test vectors from -- see `examples/reprogram_config.yaml` in the matador
            repo for the schema (shared with vanilla_gp_tiled -- the shape has no
            gp_tiled-specific fields). Writes `tb/tb_reprogram_suite.v` +
            `sim/reprogram_stimulus.memh`/`reprogram_expected.memh`/
            `reprogram_manifest.txt` into this bundle alongside the single-model
            artifacts above; each step re-loads a fresh Instruction-stream packet
            before its own Feature-batches, one continuous run, no resynthesis.

            ## Reference: capacity & model (this synthesis)

              target_fpga          = {config.target_fpga}
              required BRAM bits   = {config.required_bram_bits:,} / {budget_bits:,} budget ({pct:.1f}%)
              max_includes         = {config.max_includes:,}  (Instruction Memory depth, 16-bit words)
              max_features         = {config.max_features:,}
              max_classes          = {config.max_classes}
              batch_width          = {config.batch_width}

              model loaded for simulation:
              n_features           = {arch.n_features}
              n_classes            = {arch.n_classes}
              n_clauses_per_class  = {arch.n_clauses_per_class}
              encoded instructions = {len(E_words):,} words ({len(E_words) * 16:,} bits)
            """)
