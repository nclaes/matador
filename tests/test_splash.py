"""Tests for matador.splash — workspace-state detection for the interactive
splash screen / `matador status`.

Covers: RTL detection dynamic against the backend registry (not a hardcoded
pair), and the multi-item workspace dashboard -- raw/boolean/TMIR state are
all list-based (every dataset/model actually in the workspace, not just the
newest), with per-item next-action rendering directly under each item that
still needs one, rather than one bundled block at the end.
"""

from __future__ import annotations

import time

import matador.splash as splash
from matador.backends.registry import list_backends


def _make_workspace(tmp_path, rtl_backend: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\n")
    model_dir = tmp_path / "TMIR" / "testmodel"
    model_dir.mkdir(parents=True)
    (model_dir / "TM_TMIR_test.yaml").write_text("placeholder")
    if rtl_backend:
        (tmp_path / rtl_backend / "RTL").mkdir(parents=True)
    return tmp_path


def test_workspace_state_detects_rtl_for_every_registered_backend(tmp_path, monkeypatch):
    for backend_name in list_backends():
        work = _make_workspace(tmp_path / backend_name, rtl_backend=backend_name)
        monkeypatch.setattr(splash, "_WORK", work)

        state = splash._workspace_state()

        assert backend_name in state["rtl_backends"], (
            f"{backend_name} RTL not detected — _workspace_state() must check "
            f"every registry.list_backends() entry, not a hardcoded subset"
        )


def test_workspace_state_reports_no_rtl_before_generation(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend=None)
    monkeypatch.setattr(splash, "_WORK", work)

    state = splash._workspace_state()

    assert state["rtl_backends"] == []


def test_dm_shows_rtl_generated_not_generate_rtl_prompt_after_generation(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend="vanilla_gp_tiled")
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "RTL generated" in lines
    assert "vanilla_gp_tiled" in lines
    # Must not still be telling the user to generate RTL they already have.
    assert "not yet generated" not in lines.split("vanilla_gp_tiled")[1].split("\n")[0]


def test_dm_still_prompts_to_generate_rtl_before_any_backend_ran(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend=None)
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "not yet generated" in lines
    assert "matador generate --backend" in lines
    assert "RTL generated" not in lines


# ---------------------------------------------------------------------------
# Preprocessing pipeline (raw data -> booleanized data) workspace detection
# ---------------------------------------------------------------------------

def test_workspace_state_detects_data_source_config(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data_source_config.yaml").write_text("key: digits\n")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    assert state["data_source_config"] == tmp_path / "data_source_config.yaml"
    assert state["raw_datasets"] == []


def test_workspace_state_detects_raw_ingest_output_excluding_tmir(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "digits.npz").write_bytes(b"")
    # a TMIR model .npz must NOT be mistaken for raw ingest output
    (tmp_path / "TMIR").mkdir()
    (tmp_path / "TMIR" / "model.npz").write_bytes(b"")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    assert state["raw_datasets"] == [tmp_path / "raw" / "digits.npz"]


def test_workspace_state_lists_every_raw_dataset_not_just_newest(tmp_path, monkeypatch):
    """Multiple datasets ingested into the same /work (e.g. digits + sports
    for a reprogrammable backend's multi-model stream) must all show up,
    not just the most recently ingested one."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "digits.npz").write_bytes(b"")
    time.sleep(0.01)
    (tmp_path / "raw" / "sports.npz").write_bytes(b"")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    names = {p.stem for p in state["raw_datasets"]}
    assert names == {"digits", "sports"}


def test_workspace_state_detects_boolean_data_via_report_json(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_train.txt").write_text("0 1 0\n")
    (out / "digits_test.txt").write_text("1 0 1\n")
    (out / "digits_report.json").write_text("{}")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    assert state["boolean_datasets"] == [out / "digits_report.json"]


def test_workspace_state_lists_every_boolean_dataset_not_just_newest(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_report.json").write_text("{}")
    time.sleep(0.01)
    (out / "sports_report.json").write_text("{}")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    names = {p.name.removesuffix("_report.json") for p in state["boolean_datasets"]}
    assert names == {"digits", "sports"}


def test_dm_full_chain_from_nothing_mentions_both_starting_points(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "Nothing here yet" in lines
    assert "data_source_config.yaml" in lines
    assert "training_config.yaml" in lines


def test_dm_suggests_ingest_when_only_data_source_config_present(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data_source_config.yaml").write_text("key: digits\n")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "configured, not yet run" in lines
    assert "matador ingest --config" in lines


def test_dm_suggests_booleanize_dataset_shortcut_for_registered_raw_data(tmp_path, monkeypatch):
    """A raw dataset whose name matches a registered catalog key gets the
    concise --dataset shortcut, not the generic --config template flow."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "digits.npz").write_bytes(b"")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "matador booleanize --dataset digits --raw-dir" in lines


def test_dm_suggests_generic_booleanize_config_for_unregistered_raw_data(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "my_custom_data.npz").write_bytes(b"")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "matador booleanize --dataset my_custom_data" not in lines
    assert "cp examples/booleanisation_config.yaml" in lines
    assert "matador booleanize --config /work/booleanisation_config.yaml" in lines


def test_dm_does_not_suggest_booleanizing_a_raw_dataset_already_booleanized(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "digits.npz").write_bytes(b"")
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_report.json").write_text("{}")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    # the digits raw-data row is ticked and has no action line under it
    raw_section = lines.split("Raw data")[1].split("Boolean data")[0]
    assert "digits" in raw_section
    assert "booleanize" not in raw_section.lower()


def test_dm_suggests_training_config_with_exact_paths_when_booleanized(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_train.txt").write_text("0 1 0\n")
    (out / "digits_test.txt").write_text("1 0 1\n")
    (out / "digits_report.json").write_text("{}")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert str(out / "digits_train.txt") in lines
    assert str(out / "digits_test.txt") in lines


def test_dm_does_not_repeat_train_action_for_a_dataset_already_trained(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_report.json").write_text("{}")
    model_dir = tmp_path / "TMIR" / "digits"
    model_dir.mkdir(parents=True)
    (model_dir / "TM_TMIR_test.yaml").write_text("placeholder")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    bool_section = lines.split("Boolean data")[1].split("Training config")[0]
    assert "digits" in bool_section
    assert "matador train" not in bool_section


def test_dm_training_config_target_suppresses_duplicate_train_prompt(tmp_path, monkeypatch):
    """When training_config.yaml's train_data already points at a Boolean
    dataset that's covered by its own per-item action above, the standalone
    Training config block shouldn't repeat the same matador train command."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_report.json").write_text("{}")
    (tmp_path / "training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {out / 'digits_train.txt'}\ntest_data: {out / 'digits_test.txt'}\n"
    )
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    cfg_section = lines.split("Training config")[1].split("Trained models")[0]
    assert "matador train" not in cfg_section
    bool_section = lines.split("Boolean data")[1].split("Training config")[0]
    assert "matador train --config" in bool_section


def test_dm_training_config_prompts_standalone_for_externally_provided_data(tmp_path, monkeypatch):
    """A training_config.yaml pointed at Boolean data the user made
    themselves (no matador-booleanize report.json at all) must still get a
    train prompt -- the per-Boolean-dataset view can't see it."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_dir = tmp_path / "mydata"
    data_dir.mkdir()
    (tmp_path / "training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {data_dir / 'own_train.txt'}\ntest_data: {data_dir / 'own_test.txt'}\n"
    )
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    cfg_section = lines.split("Training config")[1].split("Trained models")[0]
    assert "matador train --config" in cfg_section


def test_workspace_state_lists_every_training_config_not_just_one(tmp_path, monkeypatch):
    """A workspace can legitimately hold several training configs, one per
    model being trained -- matches matador train's own per-model
    TMIR/<model_name>/ output namespacing. Used to report only a single
    config regardless of how many actually existed."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\ntrain_data: digits_train.txt\n")
    (tmp_path / "sports_training_config.yaml").write_text("tm_type: vanilla\ntrain_data: sports_train.txt\n")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    names = {p.name for p in state["training_configs"]}
    assert names == {"training_config.yaml", "sports_training_config.yaml"}


def test_dm_boolean_dataset_suggests_non_colliding_config_when_another_already_exists(tmp_path, monkeypatch):
    """Regression: a second Boolean dataset with no training config of its
    own yet used to always be told to cp to the same canonical
    training_config.yaml as any other dataset -- silently inviting the user
    to overwrite whichever model's config was already there. It must
    instead suggest a dataset-specific filename once another config exists."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {tmp_path / 'digits_train.txt'}\n"
    )
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "sports_report.json").write_text("{}")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    bool_section = lines.split("Boolean data")[1].split("Training config")[0]
    assert f"cp examples/training_config.yaml {tmp_path / 'sports_training_config.yaml'}" in bool_section
    assert f"matador train --config {tmp_path / 'sports_training_config.yaml'}" in bool_section
    # must not suggest reusing/overwriting the existing (digits) config
    assert f"matador train --config {tmp_path / 'training_config.yaml'}" not in bool_section


def test_dm_boolean_dataset_reuses_existing_config_that_already_targets_it(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_report.json").write_text("{}")
    (tmp_path / "digits_training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {out / 'digits_train.txt'}\n"
    )
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    bool_section = lines.split("Boolean data")[1].split("Training config")[0]
    assert f"matador train --config {tmp_path / 'digits_training_config.yaml'}" in bool_section
    assert "cp examples/training_config.yaml" not in bool_section


def test_dm_lists_each_training_config_with_its_target(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {tmp_path / 'digits_train.txt'}\n"
    )
    (tmp_path / "sports_training_config.yaml").write_text(
        f"tm_type: vanilla\ntrain_data: {tmp_path / 'sports_train.txt'}\n"
    )
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    cfg_section = lines.split("Training config")[1].split("Trained models")[0]
    assert "training_config.yaml" in cfg_section
    assert "sports_training_config.yaml" in cfg_section
    assert "targets: digits" in cfg_section
    assert "targets: sports" in cfg_section


def test_dm_two_same_dataset_models_with_distinct_model_name_are_tracked_separately(tmp_path, monkeypatch):
    """The actual scenario this whole naming/target-matching exists for:
    two DIFFERENTLY SIZED models trained from the SAME dataset. Each config
    sets its own explicit model_name:, so _training_config_target must
    return that override, not the shared dataset name derived from
    train_data -- otherwise both configs would look like "the" config for
    "digits" and the dashboard couldn't tell a small model already exists
    from a large one still pending."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_report.json").write_text("{}")

    (tmp_path / "digits_small_training_config.yaml").write_text(
        f"tm_type: vanilla\nmodel_name: digits_small\ntrain_data: {out / 'digits_train.txt'}\nclauses: 100\n"
    )
    (tmp_path / "digits_large_training_config.yaml").write_text(
        f"tm_type: vanilla\nmodel_name: digits_large\ntrain_data: {out / 'digits_train.txt'}\nclauses: 2000\n"
    )

    # digits_small has already been trained; digits_large hasn't.
    model_dir = tmp_path / "TMIR" / "digits_small"
    model_dir.mkdir(parents=True)
    (model_dir / "TM_TMIR_test.yaml").write_text("placeholder")

    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    assert splash._training_config_target(tmp_path / "digits_small_training_config.yaml") == "digits_small"
    assert splash._training_config_target(tmp_path / "digits_large_training_config.yaml") == "digits_large"

    lines = "\n".join(splash._dm(state))
    cfg_section = lines.split("Training config")[1].split("Trained models")[0]
    assert "targets: digits_small" in cfg_section
    assert "targets: digits_large" in cfg_section
    # digits_large hasn't been trained yet -- must still get a train prompt
    assert f"matador train --config {tmp_path / 'digits_large_training_config.yaml'}" in cfg_section
    # digits_small has -- must NOT get a redundant train prompt
    assert f"matador train --config {tmp_path / 'digits_small_training_config.yaml'}" not in cfg_section


def test_dm_suggests_reprogram_suite_only_for_reprogrammable_backend_with_rtl(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend="vanilla_gp_tiled")
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "reprogram-suite" in lines
    assert "reprogram-suite (multi-model)" in lines


def test_dm_does_not_suggest_reprogram_suite_for_non_reprogrammable_backend(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend="vanilla_tiled")
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "reprogram-suite" not in lines


# ---------------------------------------------------------------------------
# show_if_interactive() no longer inline-prints the full registry -- that
# moved to `matador registry` (and list-backends/list-datasets, kept as
# aliases); the splash just points at it.
# ---------------------------------------------------------------------------

def test_splash_points_at_registry_command_instead_of_inline_listing(tmp_path, monkeypatch):
    import io

    monkeypatch.setattr(splash, "_WORK", tmp_path)

    class _FakeStdout(io.StringIO):
        def isatty(self):
            return True

    fake_stdout = _FakeStdout()
    monkeypatch.setattr(splash.sys, "stdout", fake_stdout)

    shown = splash.show_if_interactive()

    assert shown is True
    output = fake_stdout.getvalue()
    assert "matador registry" in output
    # the old inline dump is gone -- no per-backend/per-dataset detail lines
    assert "recipe available" not in output
    assert "no default recipe" not in output


def test_splash_footer_points_at_developer_docs_for_extending(tmp_path, monkeypatch):
    import io

    work = _make_workspace(tmp_path, rtl_backend=None)
    monkeypatch.setattr(splash, "_WORK", work)

    class _FakeStdout(io.StringIO):
        def isatty(self):
            return True

    fake_stdout = _FakeStdout()
    monkeypatch.setattr(splash.sys, "stdout", fake_stdout)

    splash.show_if_interactive()

    output = fake_stdout.getvalue()
    assert "docs/Developer.md" in output
    # comes after the "Start over" block, alongside the other one-line footer pointers
    assert output.index("Start over") < output.index("docs/Developer.md")


# ---------------------------------------------------------------------------
# Pre-made (already-booleanised) datasets — real data.zip-derived files
# already sitting in data/<Name>/*.txt for some catalog entries. Distinct
# from "raw data ingested"/"data booleanized" (which are about the user's
# OWN /work workspace) — this is catalog-wide, existence-checked info that
# lets a user skip ingest/booleanize entirely for datasets that already
# have it done.
# ---------------------------------------------------------------------------

def test_dm_empty_workspace_lists_premade_datasets_with_paths(tmp_path, monkeypatch):
    from matador.preprocessing import registry

    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "ready-made" in lines
    premade = [n for n in registry.list_datasets() if registry.booleanised_files_exist(n)]
    assert premade, "expected at least one dataset with pre-extracted booleanised files"
    for n in premade:
        assert n in lines
        files = registry.resolve_booleanised_files(n)
        assert str(files[0]) in lines
        assert str(files[1]) in lines
    # a dataset whose archive isn't pre-extracted must NOT be listed as premade
    not_premade = [n for n in registry.list_datasets() if not registry.booleanised_files_exist(n)]
    assert not_premade, "expected at least one dataset without pre-extracted files (sanity check)"


def test_dm_premade_datasets_disappear_once_pipeline_started(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend=None)
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "pre-made" not in lines
    assert "ready-made" not in lines


def test_dm_shows_premade_status_row_and_action_when_relevant(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data_source_config.yaml").write_text("key: digits\n")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "pre-made datasets" in lines
    assert "Skip ingest/booleanize entirely" in lines
    assert "cp examples/training_config.yaml" in lines


# ---------------------------------------------------------------------------
# Multi-model workspaces: matador train now namespaces output under
# TMIR/<model_name>/ (matador/models/trainer.py::export_tmir) so multiple
# models can coexist in one /work. The dashboard is fully multi-model-aware:
# every discovered model directory is listed, not just the newest.
# ---------------------------------------------------------------------------

def test_workspace_state_detects_tmir_under_nested_model_subdirectory(tmp_path, monkeypatch):
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\n")
    model_dir = tmp_path / "TMIR" / "digits"
    model_dir.mkdir(parents=True)
    (model_dir / "TM_TMIR_test.yaml").write_text("placeholder")
    (model_dir / "TM_TMIR_test.npz").write_bytes(b"")
    (model_dir / "validation_config.yaml").write_text("model_path: x\n")

    state = splash._workspace_state()

    assert len(state["tmir_models"]) == 1
    model = state["tmir_models"][0]
    assert model["name"] == "digits"
    assert model["yaml"] == model_dir / "TM_TMIR_test.yaml"
    assert model["npz"] == model_dir / "TM_TMIR_test.npz"
    assert model["val_config"] == model_dir / "validation_config.yaml"

    lines = "\n".join(splash._dm(state))
    assert "digits" in lines
    assert "matador validate --config" in lines


def test_workspace_state_lists_every_model_when_multiple_coexist(tmp_path, monkeypatch):
    """Regression: this used to report only the newest of several coexisting
    models (documented, accepted limitation) -- namespacing under
    TMIR/<model_name>/ was originally added just to stop them overwriting
    each other's files; the dashboard itself stayed single-model-aware. It
    is now fully multi-model-aware: every model directory is discovered."""
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\n")

    older = tmp_path / "TMIR" / "digits"
    older.mkdir(parents=True)
    (older / "TM_TMIR_test.yaml").write_text("placeholder")

    time.sleep(0.01)

    newer = tmp_path / "TMIR" / "sports"
    newer.mkdir(parents=True)
    (newer / "TM_TMIR_test.yaml").write_text("placeholder")

    state = splash._workspace_state()

    names = {m["name"] for m in state["tmir_models"]}
    assert names == {"digits", "sports"}

    lines = "\n".join(splash._dm(state))
    assert "digits" in lines
    assert "sports" in lines


def test_dm_model_shows_provenance_action_only_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\n")
    model_dir = tmp_path / "TMIR" / "digits"
    model_dir.mkdir(parents=True)
    (model_dir / "TM_TMIR_test.yaml").write_text("placeholder")
    (model_dir / "TM_TMIR_test.npz").write_bytes(b"")

    state = splash._workspace_state()
    lines = "\n".join(splash._dm(state))
    assert "matador provenance --model" in lines

    (model_dir / "provenance_report.json").write_text("{}")
    state = splash._workspace_state()
    lines = "\n".join(splash._dm(state))
    assert "matador provenance --model" not in lines


def test_dm_mentions_clean_dry_run_as_a_start_over_option(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend=None)
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "matador clean --dry-run" in lines
