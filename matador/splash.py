"""Matador splash screen and workspace-aware DM guidance.

Shown only when `matador` is invoked with no arguments on an interactive
terminal.  In scripts, pipes, and CI the splash is suppressed automatically.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from matador import __version__

# ---------------------------------------------------------------------------
# Workspace state detection
# ---------------------------------------------------------------------------

_WORK = Path("/work")


def _find_newest(pattern: str) -> Path | None:
    matches = sorted(_WORK.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _find_training_configs() -> list[Path]:
    """Return every training config found in /work root, regardless of
    filename -- the common canonical name(s), plus any other YAML containing
    the 'tm_type' field (unique to training configs). A workspace can
    legitimately hold several, one per model being trained (e.g.
    training_config.yaml + sports_training_config.yaml) -- matches matador
    train's own per-model TMIR/<model_name>/ output namespacing, rather than
    assuming there's only ever one config in play."""
    found: dict[str, Path] = {}

    # Common explicit names
    for name in ("training_config.yaml", "training_config.yml",
                 "training.yaml", "training.yml"):
        p = _WORK / name
        if p.exists():
            found[p.name] = p

    # Any other root-level YAML with a tm_type key
    for p in _WORK.glob("*.yaml"):
        if p.name in found:
            continue
        try:
            if "tm_type" in p.read_text():
                found[p.name] = p
        except Exception:
            pass

    return sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _training_config_target(cfg_path: "Path | None") -> "str | None":
    """The model_name a training config actually trains -- an explicit
    model_name: if set, else derived from train_data's filename. Mirrors
    matador.models.trainer.export_tmir()'s own precedence EXACTLY
    (`config.model_name or _derive_model_name(config.train_data)`) so the
    dashboard/CLI guidance agree with what a real `matador train` run would
    actually name the output as. This is what makes two differently-sized
    models trained from the SAME dataset distinguishable: with an explicit
    model_name: (e.g. digits_large vs. plain digits), each config's target
    is its own model_name, not the shared dataset name -- ignoring
    model_name: here would make both configs look like "the" config for
    "digits" and the dashboard couldn't tell them apart."""
    if not cfg_path:
        return None
    try:
        import yaml as _yaml
        data = _yaml.safe_load(cfg_path.read_text()) or {}
        explicit = data.get("model_name")
        if explicit:
            return explicit
        train_data = data.get("train_data")
        if not train_data:
            return None
        from matador.models.trainer import _derive_model_name
        return _derive_model_name(Path(train_data))
    except Exception:
        return None


def _read_model_meta(model_dir: Path) -> dict:
    meta_path = model_dir / "model_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _discover_models() -> list[dict]:
    """One entry per discovered model directory. matador train namespaces
    output under TMIR/<model_name>/ (matador/models/trainer.py::export_tmir)
    so multiple models can coexist in one /work -- grouped by parent
    directory (npz OR yaml, since a workspace can have either without the
    other) rather than picking just the newest, so the dashboard is aware of
    every model actually sitting in the workspace, not just the last one
    trained."""
    npz_dirs = {p.parent for p in _WORK.glob("TMIR/**/*.npz")}
    yaml_dirs = {p.parent for p in _WORK.glob("TMIR/**/TM_TMIR_*.yaml")}
    model_dirs = sorted(npz_dirs | yaml_dirs, key=lambda d: d.stat().st_mtime, reverse=True)

    models = []
    for d in model_dirs:
        npz = next(iter(d.glob("*.npz")), None)
        yaml_path = next(iter(d.glob("TM_TMIR_*.yaml")), None)
        val_config = d / "validation_config.yaml"
        provenance = d / "provenance_report.json"
        models.append({
            "name": d.name,
            "dir": d,
            "npz": npz,
            "yaml": yaml_path,
            "val_config": val_config if val_config.exists() else None,
            "provenance": provenance if provenance.exists() else None,
            "meta": _read_model_meta(d),
        })
    return models


def _workspace_state() -> dict:
    """Inspect /work and return a dict of what has been produced."""
    state: dict = {}

    if not _WORK.exists():
        return state

    # ── Preprocessing pipeline (raw data -> booleanized data), upstream of
    #    training_config below ──────────────────────────────────────────────
    state["data_source_config"] = (_WORK / "data_source_config.yaml") \
        if (_WORK / "data_source_config.yaml").exists() else None

    # matador ingest's own npz output (default output dir is /work/raw, but
    # export.path in the config can point anywhere under /work) -- excludes
    # TMIR/ so a trained model's own .npz isn't mistaken for raw data. Every
    # ingested dataset is kept (not just the newest), since one /work can
    # legitimately hold several -- that's exactly the scenario matador
    # train's model_name namespacing exists to support.
    state["raw_datasets"] = sorted(
        (p for p in _WORK.glob("**/*.npz") if "TMIR" not in p.relative_to(_WORK).parts),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    state["booleanisation_config"] = (_WORK / "booleanisation_config.yaml") \
        if (_WORK / "booleanisation_config.yaml").exists() else None

    # matador booleanize always writes a "<name>_report.json" alongside its
    # <name>_train.txt/<name>_test.txt -- a much more specific signal than
    # guessing a directory name, since export path is user-configurable.
    # Every booleanized dataset is kept, same reasoning as raw_datasets above.
    state["boolean_datasets"] = sorted(
        _WORK.glob("**/*_report.json"), key=lambda p: p.stat().st_mtime, reverse=True,
    )

    state["training_configs"] = _find_training_configs()

    # Every discovered model, not just the newest -- see _discover_models().
    state["tmir_models"] = _discover_models()

    # RTL backends may sit directly under /work or under /work/TMIR depending
    # on the output_dir used during generation. Checked against every
    # REGISTERED backend (not a hardcoded pair) so newly added backends show
    # up here automatically instead of silently going undetected.
    try:
        from matador.backends.registry import list_backends as _lb
        _backend_names = _lb()
    except Exception:
        _backend_names = []
    state["rtl_backends"] = [
        bk for bk in _backend_names
        if any((_WORK / p).exists() for p in (f"{bk}/RTL", f"TMIR/{bk}/RTL"))
    ]

    return state


# ---------------------------------------------------------------------------
# Guidance — precise, goal-oriented, matador-specific
# ---------------------------------------------------------------------------

_BOLD  = "\033[1m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"
_GOLD  = "\033[33m"
_RED   = "\033[31m"
_DIM   = "\033[2m"
_RESET = "\033[0m"


def _model_summary(model: dict) -> str:
    meta = model["meta"]
    if meta:
        n_c = meta.get("n_clauses_total", "?")
        n_cls = meta.get("n_classes", "?")
        n_f = meta.get("n_features", "?")
        fp = meta.get("model_fingerprint", "")
        fp_s = f"  …{fp[-12:]}" if fp else ""
        return f"{n_c} clauses · {n_cls} classes · {n_f} features{fp_s}"
    return str(model["dir"])


def _dm(state: dict) -> list[str]:
    """Return workspace-aware guidance: per-stage status with the next
    action shown directly under each item that needs one, not bundled into
    one block at the end."""
    lines: list[str] = []

    # ── No /work mounted ────────────────────────────────────────────────────
    if not state:
        lines += [
            f"{_GOLD}  No work directory mounted.{_RESET}",
            "",
            f"  The workflow cannot begin without one:",
            f"    {_CYAN}make shell WORK_DIR=/path/to/your/data{_RESET}",
        ]
        return lines

    raw_datasets = state.get("raw_datasets", [])
    boolean_datasets = state.get("boolean_datasets", [])
    tmir_models = state.get("tmir_models", [])
    rtl_backends = state.get("rtl_backends", [])
    has_ds_cfg   = bool(state.get("data_source_config"))
    has_bool_cfg = bool(state.get("booleanisation_config"))
    training_configs = state.get("training_configs", [])
    has_cfg = bool(training_configs)
    # Which dataset/model each existing training config's train_data already
    # points at, so guidance can offer an existing config to reuse or -- if
    # every existing one already belongs to a different model -- suggest a
    # new, non-colliding filename instead of overwriting one in use.
    target_to_cfg: dict[str, Path] = {}
    for c in training_configs:
        t = _training_config_target(c)
        if t and t not in target_to_cfg:
            target_to_cfg[t] = c

    # Pre-made Boolean datasets — already-committed train/test files (e.g.
    # from data.zip) for a registered catalog entry, requiring no `matador
    # ingest`/`matador booleanize` at all. Not every registered dataset's
    # archive is pre-extracted in every checkout, so this is a real
    # existence check (registry.booleanised_files_exist), not just "is it
    # in the catalog."
    try:
        from matador.preprocessing import registry as _dataset_registry
        all_registered = set(_dataset_registry.list_datasets())
        premade = [n for n in _dataset_registry.list_datasets() if _dataset_registry.booleanised_files_exist(n)]
    except Exception:
        all_registered = set()
        premade = []

    # ── Nothing here at all ─────────────────────────────────────────────────
    if not any((has_ds_cfg, raw_datasets, has_bool_cfg, boolean_datasets, has_cfg, tmir_models)):
        lines += [
            f"{_GOLD}  Nothing here yet.{_RESET}",
            "",
        ]
        if premade:
            lines += [
                f"  Fastest start — these registered datasets already have ready-made",
                f"  Boolean files, no ingest/booleanize needed:",
                f"    {_CYAN}matador registry{_RESET}",
            ]
            for n in premade:
                files = _dataset_registry.resolve_booleanised_files(n)
                lines.append(f"    {_DIM}{n}: train_data={files[0]}  test_data={files[1]}{_RESET}")
            lines.append("")
        lines += [
            f"  Already have your own Boolean (0/1) training data? Set up training directly:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
            "",
            f"  Starting from raw (non-Boolean) or external data instead? Check if it's",
            f"  already registered first:",
            f"    {_CYAN}matador registry{_RESET}",
            f"    {_CYAN}matador ingest --dataset <name> --output-dir /work/raw{_RESET}",
            f"  Not registered? Describe it yourself:",
            f"    {_CYAN}cp examples/data_source_config.yaml /work/data_source_config.yaml{_RESET}",
            "",
            f"  Then, either way:",
            f"    {_CYAN}matador faena{_RESET}",
        ]
        return lines

    tick = f"{_GREEN}✓{_RESET}"
    dash = f"{_DIM}–{_RESET}"

    boolean_names = {p.name.removesuffix("_report.json") for p in boolean_datasets}
    model_names = {m["name"] for m in tmir_models}

    lines.append(f"  {'Status & actions'}")
    lines.append(f"  {'─' * 56}")
    lines.append("")

    # ── Pre-made datasets shortcut (only while nothing's been started) ──────
    if premade and not any((raw_datasets, boolean_datasets, has_cfg, tmir_models)):
        lines.append(
            f"  {tick} pre-made datasets  {_DIM}{len(premade)} available, no ingest/booleanize "
            f"needed — matador registry{_RESET}"
        )
        lines.append(f"    Skip ingest/booleanize entirely — use one directly:")
        for n in premade:
            files = _dataset_registry.resolve_booleanised_files(n)
            lines.append(f"    {_DIM}{n}: train_data={files[0]}  test_data={files[1]}{_RESET}")
        lines.append(f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}")
        lines.append(f"    {_DIM}  (set train_data/test_data to one of the paths above){_RESET}")
        lines.append("")

    # ── Raw data ─────────────────────────────────────────────────────────────
    lines.append(f"  {_BOLD}Raw data{_RESET}")
    if raw_datasets:
        for p in raw_datasets:
            name = p.stem
            done = name in boolean_names
            lines.append(f"  {tick if done else dash} {name:<16} {_DIM}{p}{_RESET}")
            if not done:
                if name in all_registered:
                    lines.append(f"    → {_CYAN}matador booleanize --dataset {name} --raw-dir {p.parent}{_RESET}")
                else:
                    lines.append(f"    → {_CYAN}cp examples/booleanisation_config.yaml /work/booleanisation_config.yaml{_RESET}")
                    lines.append(f"    {_DIM}  point raw_npz at: {p}{_RESET}")
                    lines.append(f"    → {_CYAN}matador booleanize --config /work/booleanisation_config.yaml{_RESET}")
    elif has_ds_cfg:
        lines.append(f"  {dash} configured, not yet run  {_DIM}{state['data_source_config']}{_RESET}")
        lines.append(f"    → {_CYAN}matador ingest --config {state['data_source_config']}{_RESET}")
    else:
        lines.append(f"  {dash} none yet")
        lines.append(f"    → {_CYAN}matador registry{_RESET}  {_DIM}(check if it's already registered){_RESET}")
        lines.append(f"    → {_CYAN}matador ingest --dataset <name> --output-dir /work/raw{_RESET}")
    lines.append("")

    # ── Boolean data ─────────────────────────────────────────────────────────
    lines.append(f"  {_BOLD}Boolean data{_RESET}")
    if boolean_datasets:
        for p in boolean_datasets:
            name = p.name.removesuffix("_report.json")
            train_txt = p.parent / f"{name}_train.txt"
            test_txt = p.parent / f"{name}_test.txt"
            done = name in model_names
            lines.append(f"  {tick if done else dash} {name:<16} {_DIM}{p.parent}{_RESET}")
            if not done:
                existing = target_to_cfg.get(name)
                if existing:
                    lines.append(f"    → {_CYAN}matador train --config {existing}{_RESET}")
                else:
                    # No config points at this dataset yet. If other configs
                    # already exist (for other models), suggest a
                    # non-colliding name (matching matador booleanize's own
                    # <name>_train.txt/<name>_report.json convention) rather
                    # than one that would silently repurpose someone else's
                    # in-progress config. If you want a SECOND, differently
                    # sized model from this same dataset later, give it its
                    # own explicit model_name: and name its config to match
                    # (<model_name>_training_config.yaml) -- see
                    # examples/training_config.yaml.
                    suggested = _WORK / "training_config.yaml" if not training_configs \
                        else _WORK / f"{name}_training_config.yaml"
                    lines.append(f"    → {_CYAN}cp examples/training_config.yaml {suggested}{_RESET}")
                    lines.append(f"    {_DIM}  set train_data: {train_txt}{_RESET}")
                    lines.append(f"    {_DIM}  set test_data:  {test_txt}{_RESET}")
                    lines.append(f"    → {_CYAN}matador train --config {suggested}{_RESET}")
    elif has_bool_cfg:
        lines.append(f"  {dash} configured, not yet run  {_DIM}{state['booleanisation_config']}{_RESET}")
        lines.append(f"    → {_CYAN}matador booleanize --config {state['booleanisation_config']}{_RESET}")
    else:
        lines.append(f"  {dash} none yet")
    lines.append("")

    # ── Training config(s) -- a workspace can hold several, one per model
    #     (matches matador train's own per-model TMIR/<model_name>/ output
    #     namespacing). Also covers users who bring their own Boolean data
    #     with no matador-booleanize report.json at all, which the
    #     per-dataset Boolean-data view above can't see. ─────────────────────
    lines.append(f"  {_BOLD}Training config{_RESET}")
    if training_configs:
        for c in training_configs:
            target = _training_config_target(c)
            suffix = f"  {_DIM}(targets: {target}){_RESET}" if target else ""
            lines.append(f"  {tick} {_DIM}{c}{_RESET}{suffix}")
            # Only prompt here if its target isn't already covered by a
            # Boolean-data action above (or already trained) -- otherwise
            # this would just repeat the same "matador train" line twice.
            if target not in boolean_names and target not in model_names:
                lines.append(f"    → {_CYAN}matador train --config {c}{_RESET}")
    else:
        lines.append(f"  {dash} none yet")
        lines.append(f"    {_DIM}(needed if you have your own Boolean (0/1) data, not produced via matador booleanize){_RESET}")
        lines.append(f"    → {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}")
    lines.append("")

    # ── Trained models (TMIR) ────────────────────────────────────────────────
    lines.append(f"  {_BOLD}Trained models{_RESET}")
    if tmir_models:
        for m in tmir_models:
            lines.append(f"  {tick} {m['name']:<16} {_DIM}{_model_summary(m)}{_RESET}")
            if m["val_config"]:
                lines.append(f"    → validate:    {_CYAN}matador validate --config {m['val_config']}{_RESET}")
            if m["npz"] and not m["provenance"]:
                lines.append(f"    → provenance:  {_CYAN}matador provenance --model {m['npz']} --test-data /work/data/test.txt{_RESET}")
    else:
        lines.append(f"  {dash} none yet")
    lines.append("")

    # ── RTL / backends ───────────────────────────────────────────────────────
    lines.append(f"  {_BOLD}RTL / backends{_RESET}")
    if not tmir_models:
        lines.append(f"  {dash} train a model first")
    else:
        from matador.backends.registry import get as _get_backend, list_backends as _lb
        all_backends = _lb()
        for bk in all_backends:
            if bk in rtl_backends:
                lines.append(f"  {tick} {bk:<16} {_DIM}RTL generated{_RESET}")
                lines.append(f"    → simulate:  {_CYAN}matador simulate --backend {bk} --config /work/{bk}.yaml{_RESET}")
                try:
                    supports_reprogramming = _get_backend(bk)().supports_reprogramming
                except Exception:
                    supports_reprogramming = False
                if supports_reprogramming:
                    lines.append(f"    → reprogram-suite (multi-model):")
                    lines.append(f"      {_CYAN}cp examples/reprogram_config.yaml /work/reprogram_config.yaml{_RESET}")
                    lines.append(
                        f"      {_CYAN}matador reprogram-suite --backend {bk} --config /work/{bk}.yaml "
                        f"--reprogram-config /work/reprogram_config.yaml{_RESET}"
                    )
            else:
                lines.append(f"  {dash} {bk:<16} {_DIM}not yet generated{_RESET}")
                lines.append(f"    → {_CYAN}cp examples/{bk}.yaml /work/{bk}.yaml{_RESET}")
                lines.append(f"    → {_CYAN}matador generate --backend {bk} --config /work/{bk}.yaml{_RESET}")
        if rtl_backends:
            lines.append(
                f"  {_DIM}Emulate (software, no simulator needed):  "
                f"matador emulate --backend {rtl_backends[0]} --config /work/{rtl_backends[0]}.yaml --verify{_RESET}"
            )
    lines.append("")

    # ── Start over ────────────────────────────────────────────────────────────
    lines += [
        f"  Start over  {_DIM}(preview first, nothing is removed without confirmation){_RESET}:",
        f"    {_CYAN}matador clean --dry-run{_RESET}",
        "",
    ]

    return lines


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def show_if_interactive() -> bool:
    """Print splash + DM guidance if stdout is a TTY.

    Returns True if the splash was shown (caller should then exit 0),
    False if not a TTY (caller should fall through to Click help).
    """
    if not sys.stdout.isatty():
        return False

    state = _workspace_state()

    # ── Header ────────────────────────────────────────────────────────────
    print()
    print(f"  {_BOLD}{_GOLD}MATADOR{_RESET}")
    print(f"  {_DIM}Trains Tsetlin Machines and takes them all the way to{_RESET}")
    print(f"  {_DIM}synthesisable Verilog RTL: ingest → booleanize → train →{_RESET}")
    print(f"  {_DIM}generate → simulate/emulate, with multi-model runtime{_RESET}")
    print(f"  {_DIM}reprogramming and reproducibility tooling built in.{_RESET}")
    print(f"  {_DIM}Microsystems Group · Newcastle University{_RESET}")
    print(f"  {'─' * 56}")
    print()
    print(f"  {_DIM}Registered datasets & backends:  matador registry{_RESET}")
    print()

    # ── DM guidance ───────────────────────────────────────────────────────
    for line in _dm(state):
        print(line)

    print()
    print(f"  {_DIM}All commands:  matador --help{_RESET}")
    print(f"  {_DIM}Guided flow:   matador faena{_RESET}")
    print(f"  {_DIM}Add your own dataset/technique/backend:  see docs/Developer.md{_RESET}")
    print()

    return True
