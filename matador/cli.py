import logging
import sys
from pathlib import Path

import click
import yaml

from matador import __version__

_DEFAULT_CONFIG = Path("/work/training_config.yaml")

_LOGGER = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_eager=True, is_flag=True, default=False, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Matador — automated RTL Accelerator Generator for Tsetlin Machines."""
    if ctx.invoked_subcommand is None:
        from matador.splash import show_if_interactive
        if not show_if_interactive():
            click.echo(ctx.get_help())
        return
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


@main.command("faena")
def faena() -> None:
    """Interactive guide: toolchain overview + next-step wizard."""
    _print_toolchain()

    has_model = click.confirm(
        "\n  Do you already have a trained Tsetlin Machine model (TMIR file)?",
        default=False,
    )

    if has_model:
        click.echo("")
        click.echo("  Skip straight to RTL generation:")
        click.echo("    Copy and edit:  cp examples/generate_config.yaml /work/generate_config.yaml")
        click.echo("    Then run:       matador generate --config /work/generate_config.yaml")
        return

    click.echo("")
    click.echo("  Step 1 — edit the training config:")
    click.echo("    cp examples/training_config.yaml /work/training_config.yaml")
    click.echo("")
    click.echo("  Step 2 — train:")
    click.echo("    matador train --config /work/training_config.yaml")
    click.echo("    Outputs: /work/TMIR/<model>.npz  +  /work/TMIR/validation_config.yaml")
    click.echo("             /work/TMIR/rom_inference.py  (standalone provenance script)")
    click.echo("")

    if click.confirm("  Start training now using /work/training_config.yaml?", default=True):
        _run_training(_DEFAULT_CONFIG)


def _print_toolchain() -> None:
    """Print the Matador toolchain stages to stdout."""
    # Right column is 40 chars wide
    C = 40
    stages = [
        ("train",
         "Train a TM model on Boolean feature data",
         "training_config.yaml → TMIR (.npz/.yaml)"),
        ("validate",
         "Check model accuracy on the held-out test",
         "TMIR + test data → accuracy report"),
        ("generate",
         "Synthesise Verilog RTL for all backends",
         "TMIR + generate_config.yaml → RTL/"),
        ("emulate",
         "Cycle-accurate software emulator",
         "TMIR + config → InferenceTrace per sample"),
        ("simulate",
         "Compile and run RTL testbenches",
         "generated RTL → PASS/FAIL per testbench"),
        ("provenance",
         "ROM-based inference report",
         "TMIR + test data → JSON (fingerprint+acc)"),
    ]

    # Column widths (content only, excluding │ and padding):
    #   left  = 22  →  cell = 2+22+2 = 26 chars between │
    #   right = 42  →  cell = 2+42+2 = 46 chars between │
    # Total line = 2(indent) + 1(│) + 26 + 1(│) + 46 + 1(│) = 77 chars
    LW, RW = 22, 42
    top  = "  ┌" + "─"*(LW+4) + "┬" + "─"*(RW+4) + "┐"
    mid  = "  ├" + "─"*(LW+4) + "┼" + "─"*(RW+4) + "┤"
    sep  = "  ├" + "─"*(LW+4) + "┼" + "─"*(RW+4) + "┤"
    bot  = "  └" + "─"*(LW+4) + "┴" + "─"*(RW+4) + "┘"
    blk  = f"  │  {'':^{LW}}  │  {'':^{RW}}  │"
    title = "The Matador Toolchain"
    hdr  = f"  │  {title:^{LW+RW+6}}  │"

    click.echo("")
    click.echo(top)
    click.echo(hdr)
    click.echo("  ├" + "─"*(LW+4) + "┬" + "─"*(RW+4) + "┤")
    for cmd, summary, detail in stages:
        c1 = f"matador {cmd}"
        click.echo(f"  │  {c1:<{LW}}  │  {summary[:RW]:<{RW}}  │")
        click.echo(f"  │  {'':^{LW}}  │  {detail[:RW]:<{RW}}  │")
        click.echo(blk)
    click.echo(bot)



@main.command("train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CONFIG,
    show_default=True,
    help="Path to training_config.yaml.",
)
def train(config_path: Path) -> None:
    """Train a Tsetlin Machine model from a YAML config file."""
    _run_training(config_path)


def _run_training(config_path: Path) -> None:
    from matador.config.schema import TrainingConfig
    from matador.models.trainer import export_tmir, load_data, train_model

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Create one based on examples/training_config.yaml and mount your\n"
            "  work directory: make shell WORK_DIR=/path/to/your/data"
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
        config = TrainingConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Config loaded from {config_path}")
    click.echo(f"  TM type   : {config.tm_type.value}")
    click.echo(f"  Clauses   : {config.clauses}")
    click.echo(f"  Classes   : {config.classes}")
    click.echo(f"  Features  : {config.features}")
    click.echo(f"  s / T     : {config.s} / {config.T}")
    click.echo(f"  Epochs    : {config.epochs}")
    click.echo("")

    click.echo("Loading data…")
    try:
        data = load_data(config)
    except Exception as exc:
        raise click.ClickException(f"Failed to load data: {exc}") from exc

    click.echo(
        f"  train samples: {len(data['x_train'])}  "
        f"test samples: {len(data['x_test'])}"
    )
    click.echo("")

    click.echo(f"Training for {config.epochs} epoch(s)…")
    tm = train_model(config, data)

    click.echo("")
    click.echo("Exporting TMIR…")
    yaml_out, npz_out, val_cfg_out = export_tmir(tm, config)
    tmir_dir = yaml_out.parent
    click.echo(f"Done. TMIR written to:\n  {yaml_out}\n  {npz_out}")
    click.echo(f"\nProvenance script (numpy-only, shareable):")
    click.echo(f"  python3 {tmir_dir}/rom_inference.py --test-data <path>")
    click.echo(f"\nTo validate the trained model run:")
    click.echo(f"  matador validate --config {val_cfg_out}")


_DEFAULT_VALIDATION_CONFIG = Path("/work/validation_config.yaml")


@main.command("validate")
@click.option(
    "--backend", "backend_name",
    default="tiled",
    show_default=True,
    type=click.Choice(["tiled", "hardwired"]),
    help="Backend (used with --mode rtl to locate the generated RTL directory).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_VALIDATION_CONFIG,
    show_default=True,
    help="Path to validation_config.yaml.",
)
@click.option(
    "--mode",
    type=click.Choice(["model", "rtl"]),
    default="model",
    show_default=True,
    help="model: software validation against TMIR; rtl: validate synthesised RTL via simulation.",
)
def validate(backend_name: str, config_path: Path, mode: str) -> None:
    """Validate a TMIR model or generated RTL."""
    from matador.models.validator import validate_model, validate_rtl

    if not config_path.exists():
        if mode == "rtl":
            raise click.ClickException(
                f"Config file not found: {config_path}\n"
                "  Pass your accelerator_config.yaml:\n"
                "    matador validate --mode rtl --config /path/to/accelerator_config.yaml"
            )
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Create a validation_config.yaml with at minimum:\n"
            "    model_path: /path/to/model.yaml\n"
            "  Optionally add:\n"
            "    test_data:  /path/to/test.txt"
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        raise click.ClickException(f"Failed to read config: {exc}") from exc

    if mode == "rtl":
        from matador.backends.registry import get as get_backend
        backend = get_backend(backend_name)()
        try:
            config = backend.config_class.model_validate(raw)
        except Exception as exc:
            raise click.ClickException(f"Invalid accelerator config: {exc}") from exc

        # RTL lives under <output_dir>/<backend>/
        config = config.model_copy(update={"output_dir": config.output_dir / backend_name})
        click.echo(f"Backend       : {backend_name}")
        click.echo(f"RTL directory : {config.output_dir / 'RTL'}")
        click.echo("")

        try:
            report = validate_rtl(config)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        for r in report.results:
            label = click.style("PASS", fg="green") if r.passed else click.style("FAIL", fg="red")
            click.echo(f"  {r.name:<20}  {label}   {r.summary}")
            for line in r.failures:
                click.echo(f"      {line}")

        click.echo("")
        click.echo(f"{report.n_passed} passed, {report.n_failed} failed")
        click.echo("")
        if report.all_passed:
            click.echo(click.style("RTL validation passed.", fg="green"))
        else:
            click.echo(click.style("RTL validation FAILED.", fg="red"))
            raise SystemExit(1)
        return

    from matador.config.schema import ValidationConfig
    try:
        config = ValidationConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Validating model: {config.model_path}")
    click.echo("")

    try:
        report = validate_model(config)
    except Exception as exc:
        raise click.ClickException(f"Failed to load model: {exc}") from exc

    # ── Schema ────────────────────────────────────────────────────────────────
    schema_label = click.style("PASS", fg="green") if report.schema_ok else click.style("FAIL", fg="red")
    click.echo(f"  Schema validation   {schema_label}")
    if not report.schema_ok:
        click.echo(f"    {report.schema_error}")

    # ── Test vectors ──────────────────────────────────────────────────────────
    if report.vectors_ran:
        v_ok = report.vectors_passed == report.vectors_total
        v_label = click.style("PASS", fg="green") if v_ok else click.style("FAIL", fg="red")
        click.echo(f"  Test vectors        {v_label}  ({report.vectors_passed}/{report.vectors_total})")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    if report.accuracy_pct is not None:
        click.echo(
            f"  Test accuracy       {report.accuracy_pct:.2f}%"
            f"  ({report.accuracy_correct}/{report.accuracy_total} correct)"
        )

    click.echo("")
    if report.all_passed:
        click.echo(click.style("Validation passed.", fg="green"))
    else:
        click.echo(click.style("Validation FAILED.", fg="red"))
        raise SystemExit(1)


_DEFAULT_ACCELERATOR_CONFIG = Path("/work/accelerator_config.yaml")


@main.command("generate")
@click.option(
    "--backend", "backend_name",
    default=None,
    type=click.Choice(["tiled", "hardwired"]),
    help="Backend to generate. Omit to generate all registered backends.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to config YAML (see examples/generate_config.yaml).",
)
def generate(backend_name: str | None, config_path: Path) -> None:
    """Generate RTL for a TM accelerator.

    By default all registered backends are generated, each into its own
    subdirectory:  <output_dir>/<backend>/RTL/

    Use --backend to generate a single backend only.

    \b
    Backends:
      tiled      Sequential FSM + tile ROM.  Knobs: feat_slice, clause_slice.
      hardwired  Combinational AND-gate unrolling + adder tree.  Knobs: pipeline_stages.
    """
    from matador.backends.registry import get as get_backend, list_backends
    from matador.ir.tm_ir import TMIR

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Create one based on examples/generate_config.yaml."
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        raise click.ClickException(f"Failed to read config: {exc}") from exc

    # Determine which backends to run.
    # When --backend is not specified, only generate backends whose
    # discriminating fields are present in the config.  This lets users
    # have a tiled-only config without accidentally triggering the
    # hardwired backend (and vice versa).
    _DISCRIMINATORS = {
        "tiled":     {"feat_slice", "clause_slice"},
        "hardwired": {"pipeline_stages"},
    }

    if backend_name:
        targets = [backend_name]
    else:
        all_names = list_backends()
        raw_keys  = set(raw.keys()) if isinstance(raw, dict) else set()
        configured = [n for n in all_names
                      if _DISCRIMINATORS.get(n, set()) & raw_keys]
        # Fallback: if no discriminating fields found, generate everything
        targets = configured if configured else all_names

    # Validate config against the first target that accepts it to load the model
    # (model_path and output_dir are common to all backends)
    base_config = None
    for name in targets:
        try:
            base_config = get_backend(name)().config_class.model_validate(raw)
            break
        except Exception:
            pass
    if base_config is None:
        raise click.ClickException(
            f"Config is not valid for any of the requested backends: {targets}\n"
            "  See examples/generate_config.yaml for all supported fields."
        )

    click.echo(f"Loading model: {base_config.model_path}")
    try:
        suffix = base_config.model_path.suffix.lower()
        tmir   = TMIR.from_yaml(base_config.model_path) if suffix in {".yaml", ".yml"} \
                 else TMIR.from_npz(base_config.model_path)
    except Exception as exc:
        raise click.ClickException(f"Failed to load TMIR: {exc}") from exc

    click.echo(
        f"  {tmir.variant} TM  |  "
        f"{tmir.architecture.n_features} features  |  "
        f"{tmir.architecture.n_classes} classes  |  "
        f"{tmir.architecture.n_clauses_total} clauses"
    )
    click.echo("")

    succeeded, failed = [], []

    for name in targets:
        backend = get_backend(name)()
        try:
            config = backend.config_class.model_validate(raw)
        except Exception as exc:
            click.echo(f"  [{name}] skipped — config invalid: {exc}")
            failed.append(name)
            continue

        # Each backend generates into its own subdirectory
        namespaced_config = config.model_copy(
            update={"output_dir": config.output_dir / name}
        )

        click.echo(f"  [{name}] generating…")
        try:
            artifacts = backend.generate(tmir, namespaced_config)
            click.echo(f"  [{name}] RTL written to: {artifacts.rtl_dir}")
            succeeded.append((name, artifacts))
        except NotImplementedError as exc:
            click.echo(f"  [{name}] skipped — {exc}")
            failed.append(name)
        except Exception as exc:
            click.echo(f"  [{name}] FAILED — {exc}")
            failed.append(name)

    click.echo("")
    if not succeeded:
        raise click.ClickException("No backends generated successfully.")

    click.echo(f"Generated: {', '.join(n for n, _ in succeeded)}")
    click.echo("")
    click.echo("Next steps:")
    for name, artifacts in succeeded:
        click.echo(f"  matador simulate --backend {name} --config {config_path}")
    click.echo(f"  matador emulate  --backend <name>  --config {config_path}")


@main.command("simulate")
@click.option(
    "--backend", "backend_name",
    default="tiled",
    show_default=True,
    type=click.Choice(["tiled", "hardwired"]),
    help="Backend that was used to generate the RTL.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to backend config YAML.",
)
@click.option(
    "--sim",
    type=click.Choice(["verilator", "iverilog", "auto"]),
    default="auto",
    show_default=True,
    help="Simulator to use. auto: Verilator if available, else iverilog.",
)
@click.option("--tb", default=None, help="Run only this testbench (e.g. tb_system).")
@click.option("--waves", "open_waves", is_flag=True, default=False,
              help="Open GTKWave after simulation.")
def simulate(backend_name: str, config_path: Path, sim: str, tb: str, open_waves: bool) -> None:
    """Compile and run RTL testbenches; optionally open GTKWave."""
    from matador.backends.registry import get as get_backend
    from matador.models.validator import validate_rtl

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  Run 'matador generate --backend {backend_name}' first."
        )

    backend = get_backend(backend_name)()
    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    # RTL lives under <output_dir>/<backend>/
    config = config.model_copy(update={"output_dir": config.output_dir / backend_name})

    click.echo("Running RTL simulation…")
    click.echo("")

    try:
        report = validate_rtl(config)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    for r in report.results:
        if tb and r.name != tb and r.name != f"tb_{tb}":
            continue
        label = click.style("PASS", fg="green") if r.passed else click.style("FAIL", fg="red")
        click.echo(f"  {r.name:<25}  {label}   {r.summary}")
        for line in r.failures:
            click.echo(f"      {line}")

    click.echo("")
    click.echo(f"{report.n_passed} passed, {report.n_failed} failed")
    click.echo("")

    if open_waves:
        import subprocess
        sim_dir  = config.output_dir / "RTL" / "sim"
        waves_sh = sim_dir / "waves.sh"
        tb_name  = tb or "tb_system"
        if waves_sh.exists():
            click.echo(f"Opening GTKWave for {tb_name}…")
            subprocess.Popen(["bash", str(waves_sh), tb_name])
        else:
            click.echo(f"waves.sh not found at {waves_sh} — re-run 'matador generate'")

    if not report.all_passed:
        raise SystemExit(1)


@main.command("waves")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to accelerator_config.yaml.",
)
@click.option("--tb", default="tb_system", show_default=True,
              help="Which testbench waveform to open.")
def waves(config_path: Path, tb: str) -> None:
    """Open GTKWave for a simulated testbench waveform."""
    import subprocess
    from matador.config.schema import TMAcceleratorConfig

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Run 'matador generate' first."
        )

    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = TMAcceleratorConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    sim_dir  = config.output_dir / "RTL" / "sim"
    waves_sh = sim_dir / "waves.sh"

    if not waves_sh.exists():
        raise click.ClickException(
            f"waves.sh not found at {waves_sh}\n"
            "  Run 'matador generate' first."
        )

    click.echo(f"Opening GTKWave for {tb}…")
    result = subprocess.run(["bash", str(waves_sh), tb])
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@main.command("emulate")
@click.option(
    "--backend", "backend_name",
    default="tiled",
    show_default=True,
    type=click.Choice(["tiled", "hardwired"]),
    help="Backend that was used to generate the RTL.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to backend config YAML.",
)
@click.option("--trace", "trace_path", type=click.Path(path_type=Path),
              default=None, help="Write InferenceTrace JSON to this path.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print per-vector trace summary to stdout.")
@click.option("--verify", is_flag=True, default=False,
              help="Also cross-check emulator against reference inference.")
def emulate(backend_name: str, config_path: Path, trace_path: Path, verbose: bool, verify: bool) -> None:
    """Run the software emulator on embedded TMIR test vectors."""
    import json
    from matador.backends.registry import get as get_backend
    from matador.emulator.accelerator import TMAcceleratorEmulator
    from matador.ir.tm_ir import TMIR

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  Run 'matador generate --backend {backend_name}' first or point to a config YAML."
        )

    backend = get_backend(backend_name)()
    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    # Model lives in the namespaced output dir written by generate
    config = config.model_copy(update={"output_dir": config.output_dir / backend_name})

    click.echo(f"Loading model: {config.model_path}")
    try:
        suffix = config.model_path.suffix.lower()
        tmir   = TMIR.from_yaml(config.model_path) if suffix in {".yaml", ".yml"} else TMIR.from_npz(config.model_path)
    except Exception as exc:
        raise click.ClickException(f"Failed to load TMIR: {exc}") from exc

    if not (tmir.verification and tmir.verification.test_vectors):
        raise click.ClickException(
            "No test vectors embedded in TMIR.\n"
            "  Re-train with 'matador train' — test vectors are embedded automatically."
        )

    vectors = tmir.verification.test_vectors
    click.echo(f"Running emulator on {len(vectors)} test vector(s)…")
    click.echo("")

    emul = TMAcceleratorEmulator(tmir, config)
    all_traces = []
    fail_cnt   = 0
    pass_cnt   = 0

    for idx, vec in enumerate(vectors):
        trace = emul.run(vec.input)
        predicted = trace.predicted_class
        expected  = vec.expected_class
        correct   = (predicted == expected)

        if correct:
            label = click.style("PASS", fg="green")
            pass_cnt += 1
        else:
            label = click.style("FAIL", fg="red")
            fail_cnt += 1

        scores_str = ", ".join(str(s) for s in (trace.argmax_event.scores if trace.argmax_event else []))
        click.echo(f"  [{idx:3d}] {label}  predicted={predicted}  expected={expected}  scores=[{scores_str}]")

        if verbose:
            click.echo(f"         FSM transitions: {len(trace.fsm_events)}")
            click.echo(f"         ROM reads:        {len(trace.rom_events)}")
            click.echo(f"         Clause evals:     {len(trace.clause_partial_events)}")
            click.echo(f"         Score votes:      {len(trace.score_vote_events)}")

        all_traces.append(trace)

    click.echo("")
    click.echo(f"{pass_cnt} passed, {fail_cnt} failed")

    if verify:
        click.echo("")
        click.echo("Cross-checking emulator against reference inference…")
        from matador.verification.compare import compare_emulator_to_reference
        report = compare_emulator_to_reference(tmir, config)
        if report.all_passed:
            click.echo(click.style(
                f"  Emulator matches reference on all {report.n_vectors} vector(s).", fg="green"
            ))
        else:
            click.echo(click.style(
                f"  Emulator/reference MISMATCH on {report.n_failed}/{report.n_vectors} vector(s).",
                fg="red",
            ))
            first = report.first_failure()
            if first:
                click.echo(f"  First mismatch: vector {first.index}")
                click.echo(f"    emulator  class={first.emulator_class}  scores={first.emulator_scores}")
                click.echo(f"    reference class={first.reference_class} scores={first.reference_scores}")
            fail_cnt += report.n_failed

    if trace_path is not None:
        import dataclasses
        def _serialise(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if isinstance(obj, list):
                return [_serialise(x) for x in obj]
            return obj

        records = [_serialise(t) for t in all_traces]
        trace_path.write_text(json.dumps(records, indent=2))
        click.echo(f"Trace written to: {trace_path}")

    click.echo("")
    if fail_cnt == 0:
        click.echo(click.style("Emulation passed.", fg="green"))
    else:
        click.echo(click.style("Emulation FAILED.", fg="red"))
        raise SystemExit(1)


@main.command("provenance")
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to TMIR model file (.yaml or .npz).",
)
@click.option(
    "--test-data",
    "test_data_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to space-separated Boolean test dataset (last column = label).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write JSON report to this path (default: <model_dir>/provenance_report.json).",
)
def provenance(model_path: Path, test_data_path: Path, output_path: Path) -> None:
    """Run full-dataset ROM-based inference and write a provenance report.

    Loads the Include-action matrix from the TMIR file (identical to what
    the RTL tile ROM stores) and runs inference through the pure-NumPy
    reference engine.  No tmu C extension required.

    The JSON report includes: model fingerprint, dataset SHA-256, accuracy,
    per-class accuracy, and confusion matrix — suitable for sharing with
    hardware collaborators as a golden reference.
    """
    from matador.inference.provenance import generate

    if not model_path.exists():
        raise click.ClickException(f"Model file not found: {model_path}")
    if not test_data_path.exists():
        raise click.ClickException(f"Test dataset not found: {test_data_path}")

    if output_path is None:
        output_path = model_path.parent / "provenance_report.json"

    click.echo(f"Model      : {model_path}")
    click.echo(f"Test data  : {test_data_path}")
    click.echo(f"Inference  : ROM-equivalent (pure NumPy, matador.inference.reference)")
    click.echo("")

    try:
        report = generate(model_path, test_data_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"  Samples  : {report.n_samples}")
    click.echo(f"  Features : {report.n_features}")
    click.echo(f"  Classes  : {report.n_classes}")
    click.echo(f"  Clauses  : {report.n_clauses_total}")
    click.echo("")
    click.echo(f"  Accuracy : {report.accuracy_pct:.2f}%"
               f"  ({report.accuracy_correct}/{report.accuracy_total})")
    click.echo("")
    click.echo("  Per-class accuracy:")
    for cls, res in report.per_class.items():
        click.echo(f"    class {cls:>2}  {res['accuracy_pct']:6.2f}%"
                   f"  ({res['correct']}/{res['total']})")
    click.echo("")
    click.echo(f"  Model fingerprint : {report.model_fingerprint}")
    click.echo(f"  Dataset SHA-256   : {report.test_data_sha256}")
    click.echo("")

    report.write(output_path)
    click.echo(f"Provenance report written to:\n  {output_path}")


@main.command()
def version() -> None:
    """Print the Matador version."""
    click.echo(f"matador {__version__}")


@main.command("list-backends")
def list_backends() -> None:
    """List available RTL accelerator backends."""
    from matador.backends.registry import list_backends as _list
    click.echo("Available accelerator backends (--backend flag):")
    descriptions = {
        "tiled":     "Sequential FSM + tile ROM.  Knobs: feat_slice, clause_slice.",
        "hardwired": "Combinational AND-gate unrolling + adder tree.  Knobs: pipeline_stages.  [Phase 2]",
    }
    for name in _list():
        click.echo(f"  {name:<12} {descriptions.get(name, '')}")
    click.echo("")
    click.echo("Synthesis backends (Vivado):")
    click.echo("  vivado      Xilinx Vivado (synthesis + implementation)")


@main.command("list-simulators")
def list_simulators() -> None:
    """List available RTL simulators."""
    click.echo("Available simulators:")
    click.echo("  verilator   Open-source SystemVerilog simulator")
    click.echo("  iverilog    Icarus Verilog simulator")
