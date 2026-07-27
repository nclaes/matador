"""Tests for `matador clean` — a destructive command, so these exercise
real filesystem state (not mocks) to verify exactly what does and doesn't
get removed under each mode.
"""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from matador.cli import main


def _make_work(tmp: Path) -> None:
    (tmp / "TMIR").mkdir()
    (tmp / "TMIR" / "digits").mkdir()
    (tmp / "TMIR" / "digits" / "model.yaml").write_text("x")
    (tmp / "raw").mkdir()
    (tmp / "raw" / "digits.npz").write_bytes(b"")
    (tmp / "booleanised").mkdir()
    (tmp / "_cache").mkdir()
    (tmp / "_extracted").mkdir()
    (tmp / "vanilla_gp_tiled").mkdir()
    (tmp / "vanilla_gp_tiled" / "RTL").mkdir()
    (tmp / "training_config.yaml").write_text("tm_type: vanilla\n")
    (tmp / "vanilla_gp_tiled.yaml").write_text("model_path: x\n")
    (tmp / "data_source_config.yaml").write_text("key: digits\n")
    (tmp / "my_own_notes.txt").write_text("unrelated file the user put here")


def test_clean_nonexistent_work_dir_is_a_graceful_noop(tmp_path):
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path / "does_not_exist")])
    assert result.exit_code == 0
    assert "does not exist" in result.output


def test_clean_dry_run_deletes_nothing(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0
    assert "dry run" in result.output
    assert (tmp_path / "TMIR").exists()
    assert (tmp_path / "training_config.yaml").exists()
    assert (tmp_path / "my_own_notes.txt").exists()


def test_clean_declining_confirmation_deletes_nothing(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path)], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert (tmp_path / "TMIR").exists()


def test_clean_default_scope_removes_generated_output_keeps_configs_and_unrelated_files(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path)], input="y\n")
    assert result.exit_code == 0

    for removed in ("TMIR", "raw", "booleanised", "_cache", "_extracted", "vanilla_gp_tiled"):
        assert not (tmp_path / removed).exists(), removed

    for kept in ("training_config.yaml", "vanilla_gp_tiled.yaml", "data_source_config.yaml", "my_own_notes.txt"):
        assert (tmp_path / kept).exists(), kept


def test_clean_yes_flag_skips_confirmation_prompt(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path), "-y"])
    assert result.exit_code == 0
    assert "[y/N]" not in result.output
    assert not (tmp_path / "TMIR").exists()


def test_clean_configs_flag_also_removes_authored_configs_not_unrelated_files(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path), "--configs", "-y"])
    assert result.exit_code == 0

    for removed in ("TMIR", "training_config.yaml", "vanilla_gp_tiled.yaml", "data_source_config.yaml"):
        assert not (tmp_path / removed).exists(), removed
    assert (tmp_path / "my_own_notes.txt").exists()


def test_clean_all_flag_removes_literally_everything(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path), "--all", "-y"])
    assert result.exit_code == 0
    assert list(tmp_path.iterdir()) == []


def test_clean_all_prompt_warns_about_unrecognized_files(tmp_path):
    _make_work(tmp_path)
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path), "--all"], input="n\n")
    assert "including anything matador didn't create" in result.output
    assert (tmp_path / "my_own_notes.txt").exists()  # declined -> untouched


def test_clean_refuses_root_path():
    result = CliRunner().invoke(main, ["clean", "--work-dir", "/"])
    assert result.exit_code != 0
    assert "Refusing to clean" in result.output


def test_clean_refuses_home_directory():
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(Path.home())])
    assert result.exit_code != 0
    assert "Refusing to clean" in result.output


def test_clean_refuses_matador_repo_root():
    import matador

    repo_root = str(Path(matador.__file__).resolve().parent.parent)
    result = CliRunner().invoke(main, ["clean", "--work-dir", repo_root])
    assert result.exit_code != 0
    assert "Refusing to clean" in result.output


def test_clean_nothing_to_clean_when_work_dir_has_no_recognized_artifacts(tmp_path):
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\n")
    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Nothing to clean" in result.output
    assert (tmp_path / "training_config.yaml").exists()


def test_clean_covers_every_registered_backend_dynamically(tmp_path):
    """Not hardcoded to a fixed backend list -- must recognize whichever
    backends matador.backends.registry.list_backends() currently returns."""
    from matador.backends.registry import list_backends

    for bk in list_backends():
        (tmp_path / bk).mkdir()

    result = CliRunner().invoke(main, ["clean", "--work-dir", str(tmp_path), "-y"])
    assert result.exit_code == 0
    for bk in list_backends():
        assert not (tmp_path / bk).exists(), bk
