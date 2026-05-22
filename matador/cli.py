import click

from matador import __version__


@click.group()
def main() -> None:
    """Matador — automated FPGA accelerator design for Tsetlin Machines."""


@main.command()
def build() -> None:
    """Build an FPGA accelerator from a trained TM model."""
    click.echo("build: not yet implemented")


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
