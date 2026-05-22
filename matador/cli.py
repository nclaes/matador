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
        "  Do you already have a trained Tsetlin Machine model (TA states file)?",
        default=False,
    )

    if has_model:
        click.echo("")
        click.echo("  RTL generation from an existing model is not yet implemented.")
        click.echo("  Run 'matador generate --help' when available.")
        return

    click.echo("")
    click.echo("  Training flow:")
    click.echo("    1. Create a training_config.yaml in your work directory (/work).")
    click.echo("       See examples/training_config.yaml for a template.")
    click.echo("    2. Run:  matador train --config /work/training_config.yaml")
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
    from matador.models.trainer import export_ta_states, load_data, train_model

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
    click.echo("Exporting TA states…")
    out = export_ta_states(tm, config)
    click.echo(f"Done. TA states written to:\n  {out}")


@main.command()
def simulate() -> None:
    """Run a Verilator simulation of the generated RTL."""
    click.echo("simulate: not yet implemented")


@main.command()
def waves() -> None:
    """Open GTKWave for simulation waveform results."""
    click.echo("waves: not yet implemented")


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
