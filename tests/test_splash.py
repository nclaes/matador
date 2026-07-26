"""Tests for matador.splash — workspace-state detection for the interactive
splash screen / `matador status`.

Regression coverage for a real bug: _workspace_state() used to hardcode RTL
detection for exactly ["vanilla_tiled", "vanilla_hardwired"], so any other
registered backend (e.g. vanilla_gp_tiled) never showed as "RTL generated"
even after a successful `matador generate`, and the splash kept telling the
user to generate RTL they'd already generated. Detection must be dynamic
against matador.backends.registry.list_backends(), not a fixed pair.
"""

from __future__ import annotations

import matador.splash as splash
from matador.backends.registry import list_backends


def _make_workspace(tmp_path, rtl_backend: str | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_config.yaml").write_text("tm_type: vanilla\n")
    tmir_dir = tmp_path / "TMIR"
    tmir_dir.mkdir()
    (tmir_dir / "TM_TMIR_test.yaml").write_text("placeholder")
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
    assert "Generate RTL" not in lines
    assert "Simulate RTL" in lines


def test_dm_still_prompts_to_generate_rtl_before_any_backend_ran(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend=None)
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "Generate RTL" in lines
    assert "Simulate RTL" not in lines


# ---------------------------------------------------------------------------
# Preprocessing pipeline (raw data -> booleanized data) workspace detection
# ---------------------------------------------------------------------------

def test_workspace_state_detects_data_source_config(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "data_source_config.yaml").write_text("key: digits\n")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    assert state["data_source_config"] == tmp_path / "data_source_config.yaml"
    assert state["raw_data"] is None


def test_workspace_state_detects_raw_ingest_output_excluding_tmir(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "digits.npz").write_bytes(b"")
    # a TMIR model .npz must NOT be mistaken for raw ingest output
    (tmp_path / "TMIR").mkdir()
    (tmp_path / "TMIR" / "model.npz").write_bytes(b"")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    assert state["raw_data"] == tmp_path / "raw" / "digits.npz"


def test_workspace_state_detects_boolean_data_via_report_json(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "booleanised"
    out.mkdir()
    (out / "digits_train.txt").write_text("0 1 0\n")
    (out / "digits_test.txt").write_text("1 0 1\n")
    (out / "digits_report.json").write_text("{}")
    monkeypatch.setattr(splash, "_WORK", tmp_path)

    state = splash._workspace_state()

    assert state["boolean_report"] == out / "digits_report.json"


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

    assert "raw data ingested" in lines
    assert "matador ingest --config" in lines


def test_dm_suggests_booleanize_when_raw_data_present(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "digits.npz").write_bytes(b"")
    monkeypatch.setattr(splash, "_WORK", tmp_path)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "Booleanize the ingested data" in lines
    assert "matador booleanize" in lines


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


def test_dm_suggests_reprogram_suite_only_for_reprogrammable_backend_with_rtl(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend="vanilla_gp_tiled")
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "reprogram-suite" in lines
    assert "Prove multi-model reprogramming" in lines


def test_dm_does_not_suggest_reprogram_suite_for_non_reprogrammable_backend(tmp_path, monkeypatch):
    work = _make_workspace(tmp_path, rtl_backend="vanilla_tiled")
    monkeypatch.setattr(splash, "_WORK", work)
    state = splash._workspace_state()

    lines = "\n".join(splash._dm(state))

    assert "reprogram-suite" not in lines
