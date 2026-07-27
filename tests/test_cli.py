from click.testing import CliRunner

from matador.cli import main


def test_version_contains_version_string() -> None:
    result = CliRunner().invoke(main, ["version"])
    assert result.exit_code == 0
    assert "0.2.0" in result.output


def test_list_backends_does_not_crash() -> None:
    result = CliRunner().invoke(main, ["list-backends"])
    assert result.exit_code == 0


def test_registry_shows_both_backends_and_datasets() -> None:
    """matador registry is a thin aggregator over list-backends + list-datasets
    -- both sections' content must appear, and each individually-invoked
    command's own output must be an exact subset of it (they share the same
    print helpers, not a separately-maintained copy)."""
    from matador.backends.registry import list_backends
    from matador.preprocessing.registry import list_datasets

    combined = CliRunner().invoke(main, ["registry"])
    assert combined.exit_code == 0
    for name in list_backends():
        assert name in combined.output
    for name in list_datasets():
        assert name in combined.output

    backends_only = CliRunner().invoke(main, ["list-backends"])
    assert backends_only.exit_code == 0
    assert backends_only.output in combined.output

    datasets_only = CliRunner().invoke(main, ["list-datasets"])
    assert datasets_only.exit_code == 0
    assert datasets_only.output in combined.output


def test_list_simulators_does_not_crash() -> None:
    result = CliRunner().invoke(main, ["list-simulators"])
    assert result.exit_code == 0


def test_faena_from_scratch_walks_ingest_booleanize_train_chain() -> None:
    """No model, no Boolean data, no raw data yet -- the wizard must walk
    through all three preprocessing steps before training, not jump
    straight to "edit training_config.yaml" the way it used to."""
    result = CliRunner().invoke(main, ["faena"], input="n\nn\nn\n")
    assert result.exit_code == 0
    assert "matador list-datasets" in result.output
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
    # Regression: this used to print "matador generate --config ..." with no
    # --backend, which is a required option -- the exact command faena told
    # users to run failed immediately. list-backends must be mentioned so a
    # brand-new user has a way to pick one.
    assert "matador list-backends" in result.output
    assert "matador generate --backend" in result.output
    assert "matador ingest --config" not in result.output
    assert "matador train --config" not in result.output


def test_faena_generate_hint_is_actually_runnable_once_backend_is_filled_in() -> None:
    """The exact command faena prints (with a real --backend substituted
    for the <backend> placeholder) must not fail on option parsing."""
    result = CliRunner().invoke(main, ["faena"], input="y\n")
    assert "matador generate --backend <backend> --config /work/<backend>.yaml" in result.output

    probe = CliRunner().invoke(main, ["generate", "--backend", "vanilla_tiled", "--config", "/nonexistent.yaml"])
    # Must fail because the config file doesn't exist (ClickException),
    # NOT because --backend is missing (UsageError/exit code 2 from click's
    # own option parser) -- proves the printed command shape is valid.
    assert "Missing option" not in probe.output
    assert "Config file not found" in probe.output


def test_faena_step5_output_paths_reflect_model_name_namespacing() -> None:
    """Regression: matador.models.trainer.export_tmir() namespaces its
    output under TMIR/<model_name>/ (fixes a real collision bug), but
    faena's own inline description of that output kept the old flat
    /work/TMIR/<model>.npz path after the change -- verify it's current."""
    result = CliRunner().invoke(main, ["faena"], input="n\ny\nn\n")
    assert "/work/TMIR/<model_name>/<model>.npz" in result.output
    assert "/work/TMIR/<model_name>/rom_inference.py" in result.output


def test_examples_backend_configs_have_no_stale_backend_flag_values() -> None:
    """Regression: examples/generate_config.yaml, accelerator_config.yaml,
    tiled_config.yaml, and hardwired_config.yaml all pre-date the
    'vanilla_*' backend-naming convention and referenced --backend
    tiled/hardwired in their own usage comments, which have never been
    valid --backend choices since the rename. Guards against the same rot
    creeping back in, or spreading to new example files."""
    import re
    from pathlib import Path

    from matador.backends.registry import list_backends

    valid = set(list_backends())
    examples_dir = Path(__file__).resolve().parents[1] / "examples"
    for path in examples_dir.glob("*.yaml"):
        for line in path.read_text().splitlines():
            stripped = line.strip().lstrip("#").strip()
            if not stripped.startswith("matador "):
                continue   # skip prose mentioning "--backend" outside an actual invocation line
            m = re.search(r"--backend\s+(\S+)", stripped)
            if not m:
                continue
            flag_value = m.group(1)
            assert flag_value in valid, (
                f"{path.name}: {line.strip()!r} references '--backend {flag_value}', which is "
                f"not a registered backend ({sorted(valid)}) -- stale example content."
            )


def test_waves_requires_backend_flag() -> None:
    """Regression: `matador waves` used to take no --backend at all, hardcode
    the pre-plugin-split TMAcceleratorConfig, and look for waveforms at
    <output_dir>/RTL/sim/ instead of the real <output_dir>/<backend>/RTL/sim/
    -- meaning it could never find a waveform for ANY backend's actual
    output. It must now require --backend like generate/simulate/emulate."""
    result = CliRunner().invoke(main, ["waves", "--config", "/nonexistent.yaml"])
    assert result.exit_code == 2
    assert "Missing option '--backend'" in result.output


def test_waves_resolves_backend_namespaced_rtl_path(tmp_path) -> None:
    """Real end-to-end: generate RTL for a backend, then confirm `matador
    waves` finds waves.sh under the actual <output_dir>/<backend>/RTL/sim/
    path it was written to -- not the old flat <output_dir>/RTL/sim/."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent / "backends"))
    from test_gp_tiled import _make_tmir  # noqa: E402 (path hack above)

    import yaml as _yaml

    tmir = _make_tmir(n_features=8, n_classes=2, n_clauses_pc=4, threshold=4)
    tmir_path = tmp_path / "model.yaml"
    tmir.to_yaml(tmir_path)

    accel_cfg = tmp_path / "vanilla_gp_tiled.yaml"
    accel_cfg.write_text(_yaml.safe_dump({
        "model_path": str(tmir_path), "output_dir": str(tmp_path / "out"),
        "fifo_depth": 16, "target_fpga": "xc7z020", "max_features": 512, "max_clauses_total": 256,
    }))

    runner = CliRunner()
    gen = runner.invoke(main, ["generate", "--backend", "vanilla_gp_tiled", "--config", str(accel_cfg)])
    assert gen.exit_code == 0, gen.output

    real_waves_sh = tmp_path / "out" / "vanilla_gp_tiled" / "RTL" / "sim" / "waves.sh"
    assert real_waves_sh.exists()

    result = runner.invoke(main, ["waves", "--backend", "vanilla_gp_tiled", "--config", str(accel_cfg)])
    # Must fail because no simulation has been run yet (waves.sh's own
    # "no waveform" message), NOT because it looked in the wrong directory.
    assert "waves.sh not found" not in result.output


def test_faena_mentions_show_recipe_for_booleanize_step() -> None:
    """faena's Step 2 (booleanize) should point users at --show-recipe as
    the way to inspect (and start editing) a recipe, whether or not the
    dataset they picked has a verified default."""
    result = CliRunner().invoke(main, ["faena"], input="n\nn\nn\n")
    assert "--show-recipe" in result.output
