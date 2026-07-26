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


def test_faena_from_scratch_walks_ingest_booleanize_train_chain() -> None:
    """No model, no Boolean data, no raw data yet -- the wizard must walk
    through all three preprocessing steps before training, not jump
    straight to "edit training_config.yaml" the way it used to."""
    result = CliRunner().invoke(main, ["faena"], input="n\nn\nn\n")
    assert result.exit_code == 0
    assert "matador ingest --config" in result.output
    assert "matador booleanize --config" in result.output
    assert "matador train --config" in result.output
    # ordering: ingest guidance must appear before booleanize, which must
    # appear before train
    assert result.output.index("matador ingest") < result.output.index("matador booleanize")
    assert result.output.index("matador booleanize") < result.output.index("matador train")


def test_faena_with_raw_data_skips_ingest_step() -> None:
    # "matador ingest" (no flags) still appears once, in the always-printed
    # static toolchain reference table at the top -- check the wizard's
    # actual step instructions (which include --config) aren't repeated.
    result = CliRunner().invoke(main, ["faena"], input="n\nn\ny\n")
    assert result.exit_code == 0
    assert "matador ingest --config" not in result.output
    assert "matador booleanize --config" in result.output


def test_faena_with_boolean_data_skips_ingest_and_booleanize() -> None:
    result = CliRunner().invoke(main, ["faena"], input="n\ny\nn\n")  # trailing n: don't start training now
    assert result.exit_code == 0
    assert "matador ingest --config" not in result.output
    assert "matador booleanize --config" not in result.output
    assert "matador train --config" in result.output


def test_faena_with_existing_model_skips_everything_else() -> None:
    result = CliRunner().invoke(main, ["faena"], input="y\n")
    assert result.exit_code == 0
    assert "matador generate --config" in result.output
    assert "matador ingest --config" not in result.output
    assert "matador train --config" not in result.output
