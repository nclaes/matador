from click.testing import CliRunner

from matador.cli import main


def test_version_contains_version_string() -> None:
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_list_backends_does_not_crash() -> None:
    result = CliRunner().invoke(main, ["list-backends"])
    assert result.exit_code == 0


def test_list_simulators_does_not_crash() -> None:
    result = CliRunner().invoke(main, ["list-simulators"])
    assert result.exit_code == 0
