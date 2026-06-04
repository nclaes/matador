import logging
from pathlib import Path

import click
import yaml

from matador import __version__

_DEFAULT_CONFIG = Path("/work/training_config.yaml")

_LOGGER = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", is_eager=True, is_flag=True, default=False, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """Matador — automated RTL Accelerator Generator for Tsetlin Machines."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


@main.command("faena")
def faena() -> None:
    """Begin the work: train a model or generate RTL from an existing one."""
    click.echo("")
    click.echo("  Matador — Tsetlin Machine RTL Accelerator Generator")
    click.echo("  " + "─" * 50)
    click.echo("")

    has_model = click.confirm(
        "  Do you already have a trained Tsetlin Machine model (TMIR file)?",
        default=False,
    )

    if has_model:
        click.echo("")
        click.echo("  RTL generation from an existing TMIR model is not yet implemented.")
        click.echo("  Run 'matador generate --help' when available.")
        return

    click.echo("")
    click.echo("  Training flow:")
    click.echo("    1. Create a training_config.yaml in your work directory (/work).")
    click.echo("       See examples/training_config.yaml for a template.")
    click.echo("    2. Run:  matador train --config /work/training_config.yaml")
    click.echo("       The model is saved as TMIR (.yaml + .npz) under <output_dir>/TMIR/.")
    click.echo("    3. After training, run:  matador generate  (coming soon)")
    click.echo("")

    if click.confirm("  Start training now using /work/training_config.yaml?", default=True):
        _run_training(_DEFAULT_CONFIG)



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
def validate(config_path: Path, mode: str) -> None:
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
        from matador.config.schema import TMAcceleratorConfig
        try:
            config = TMAcceleratorConfig.model_validate(raw)
        except Exception as exc:
            raise click.ClickException(f"Invalid accelerator config: {exc}") from exc

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
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to accelerator_config.yaml.",
)
def generate(config_path: Path) -> None:
    """Generate RTL for a vanilla TM accelerator from an accelerator config file."""
    from matador.config.schema import TMAcceleratorConfig
    from matador.ir.tm_ir import TMIR
    from matador.rtl.accelerator import TMAccelerator

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Create one based on examples/accelerator_config.yaml."
        )

    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = TMAcceleratorConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Loading model: {config.model_path}")
    try:
        suffix = config.model_path.suffix.lower()
        tmir   = TMIR.from_yaml(config.model_path) if suffix in {".yaml", ".yml"} else TMIR.from_npz(config.model_path)
    except Exception as exc:
        raise click.ClickException(f"Failed to load TMIR: {exc}") from exc

    click.echo(
        f"  {tmir.variant} TM  |  "
        f"{tmir.architecture.n_features} features  |  "
        f"{tmir.architecture.n_classes} classes  |  "
        f"{tmir.architecture.n_clauses_total} clauses"
    )
    click.echo(f"  AXI-Stream width : {config.axis_data_width} bits")
    click.echo(f"  FIFO depth       : {config.fifo_depth}")
    click.echo(f"  Tile dimensions  : feat_slice={config.feat_slice}  clause_slice={config.clause_slice}")
    click.echo("")

    try:
        accel   = TMAccelerator(tmir, config)
        rtl_dir = accel.generate()
    except Exception as exc:
        raise click.ClickException(f"RTL generation failed: {exc}") from exc

    click.echo(f"RTL written to: {rtl_dir}")
    click.echo("")
    click.echo("Sources    : RTL/src/tm_accelerator.v  (+axis_fifo, clause_eval, score_acc, argmax)")
    click.echo("Testbenches: RTL/tb/tb_system.v  (+tb_axis_fifo, tb_clause_eval, tb_score_acc, tb_argmax)")
    click.echo("Waveforms  : RTL/sim/tb_system.gtkw  (+tb_axis_fifo, tb_clause_eval, tb_score_acc, tb_argmax)")
    click.echo("")
    click.echo("To run the software emulator (no simulator required):")
    click.echo(f"  matador emulate --config {config_path} --verify")
    click.echo("To simulate (iverilog — produces .vcd):")
    click.echo(f"  matador simulate --config {config_path}")
    click.echo("To simulate with Verilator (produces FST waveforms):")
    click.echo(f"  make -C {rtl_dir}/sim/verilator run")
    click.echo("To open GTKWave:")
    click.echo(f"  matador waves --config {config_path} [--tb tb_system]")
    click.echo("To lint with Verilator:")
    click.echo(f"  bash {rtl_dir}/sim/lint_verilator.sh")


@main.command("simulate")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to accelerator_config.yaml.",
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
def simulate(config_path: Path, sim: str, tb: str, open_waves: bool) -> None:
    """Compile and run RTL testbenches; optionally open GTKWave."""
    from matador.config.schema import TMAcceleratorConfig
    from matador.models.validator import validate_rtl

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
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ACCELERATOR_CONFIG,
    show_default=True,
    help="Path to accelerator_config.yaml.",
)
@click.option("--trace", "trace_path", type=click.Path(path_type=Path),
              default=None, help="Write InferenceTrace JSON to this path.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print per-vector trace summary to stdout.")
@click.option("--verify", is_flag=True, default=False,
              help="Also cross-check emulator against reference inference.")
def emulate(config_path: Path, trace_path: Path, verbose: bool, verify: bool) -> None:
    """Run the software emulator on embedded TMIR test vectors."""
    import json
    from matador.config.schema import TMAcceleratorConfig
    from matador.emulator.accelerator import TMAcceleratorEmulator
    from matador.ir.tm_ir import TMIR

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Run 'matador generate' first or point to an accelerator_config.yaml."
        )

    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = TMAcceleratorConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

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
    """List available synthesis and implementation backends."""
    click.echo("Available backends:")
    click.echo("  vivado      Xilinx Vivado (synthesis + implementation)")


@main.command("list-simulators")
def list_simulators() -> None:
    """List available RTL simulators."""
    click.echo("Available simulators:")
    click.echo("  verilator   Open-source SystemVerilog simulator")
    click.echo("  iverilog    Icarus Verilog simulator")
