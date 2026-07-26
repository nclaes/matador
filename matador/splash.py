"""Matador splash screen and workspace-aware DM guidance.

Shown only when `matador` is invoked with no arguments on an interactive
terminal.  In scripts, pipes, and CI the splash is suppressed automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matador import __version__

# ---------------------------------------------------------------------------
# ASCII bull
# ---------------------------------------------------------------------------

_BULL = r""" 
                                                                                                                                                                                                            
                                $$@$@$$@                              
                            >$$$$@@$@$$$$$@                           
                        $$$$@$$@@$@$$@$$$@$.                        
                        @$$@$$$$$$@@$@$$$$$$@@$$a                      
                    $$@@$@$$$$$$@@$@$$$$$$@@$$@$$$@$$$$$%            
                \$$$$@@$@$$$$$$@@$@$$$$$$@@$$$$*$@      '           
                $$$$$$$@@$$$$$$$$@@$@$$$$$$@@$@$$$$ %@Bx              
    @        @$$$@$$@$$$@@$$$$$$$$@@$@$$$@$$@@$@$$$                    
    p$$$$@a  $$@$$$$$$$@$$$$$$$$@@$@$$@   @@$@$$                     
            @@@$@$$$$@$@@$@$$$$$$@@$@$$$    @$@$.                     
            $$@@$@$$$$$$B       Q@@$$@$$$$                             
        @$$@@$@$@                 B$$$$$$.                          
        $@$@$@                       @$$$                           
        $$$$                     W$$$$@$                             
        ]$$                      $$$@$                                
        @                                                             
                                                                                                
    MATADOR                                                                                                                           
    Automated RTL Accelerator Generator for Tsetlin Machines

    Microsystems Group
    Newcastle University
    ----------------------------------------------------------     

"""

# ---------------------------------------------------------------------------
# Workspace state detection
# ---------------------------------------------------------------------------

_WORK = Path("/work")


def _find_newest(pattern: str) -> Path | None:
    matches = sorted(_WORK.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _find_training_config() -> Path | None:
    """Return the training config file if one exists, regardless of filename.

    Checks common names first, then falls back to any YAML in /work/ root
    that contains the 'tm_type' field (unique to training configs).
    """
    # Common explicit names
    for name in ("training_config.yaml", "training_config.yml",
                 "training.yaml", "training.yml"):
        p = _WORK / name
        if p.exists():
            return p

    # Last resort: scan for any root-level YAML with a tm_type key
    for p in _WORK.glob("*.yaml"):
        try:
            if "tm_type" in p.read_text():
                return p
        except Exception:
            pass

    return None


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
    # TMIR/ so a trained model's own .npz isn't mistaken for raw data.
    _npz_candidates = sorted(
        (p for p in _WORK.glob("**/*.npz") if "TMIR" not in p.relative_to(_WORK).parts),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    state["raw_data"] = _npz_candidates[0] if _npz_candidates else None

    state["booleanisation_config"] = (_WORK / "booleanisation_config.yaml") \
        if (_WORK / "booleanisation_config.yaml").exists() else None

    # matador booleanize always writes a "<name>_report.json" alongside its
    # <name>_train.txt/<name>_test.txt -- a much more specific signal than
    # guessing a directory name, since export path is user-configurable.
    state["boolean_report"] = _find_newest("**/*_report.json")

    state["reprogram_config"] = (_WORK / "reprogram_config.yaml") \
        if (_WORK / "reprogram_config.yaml").exists() else None
    state["reprogram_suite"] = _find_newest("**/tb_reprogram_suite.v")

    state["training_config"] = _find_training_config()

    # Use ** recursive glob so files are found regardless of whether output_dir
    # was /work (→ /work/TMIR/*.npz) or /work/TMIR (→ /work/TMIR/TMIR/*.npz).
    # NPZ files are always model files; YAML must match TM_TMIR_* to avoid
    # picking up validation_config.yaml or accelerator_config.yaml.
    state["tmir_npz"]  = _find_newest("TMIR/**/*.npz")
    state["tmir_yaml"] = _find_newest("TMIR/**/TM_TMIR_*.yaml")

    state["val_config"] = (
        _find_newest("TMIR/**/validation_config.yaml") or
        (_WORK / "validation_config.yaml").exists()
    )

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

    state["provenance"] = (
        _find_newest("TMIR/**/provenance_report.json") or
        _find_newest("provenance_report.json")
    )

    return state


def _read_model_meta(state: dict) -> dict:
    """Read model_metadata.json from the TMIR directory if available."""
    import json
    model_path = state.get("tmir_npz") or state.get("tmir_yaml")
    if not model_path:
        return {}
    meta = model_path.parent / "model_metadata.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text())
        except Exception:
            pass
    return {}


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


def _dm(state: dict) -> list[str]:
    """Return workspace-aware guidance: full status summary + all available actions."""
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

    has_ds_cfg   = bool(state.get("data_source_config"))
    has_raw      = bool(state.get("raw_data"))
    has_bool_cfg = bool(state.get("booleanisation_config"))
    has_bool     = bool(state.get("boolean_report"))
    has_cfg  = bool(state.get("training_config"))
    has_tmir = bool(state.get("tmir_npz") or state.get("tmir_yaml"))
    has_rtl  = bool(state.get("rtl_backends"))
    has_prov = bool(state.get("provenance"))
    val_cfg  = state.get("val_config")
    model    = state.get("tmir_npz") or state.get("tmir_yaml")
    meta     = _read_model_meta(state) if has_tmir else {}

    cfg = state.get("training_config")

    # ── Nothing here at all ─────────────────────────────────────────────────
    if not any((has_ds_cfg, has_raw, has_bool_cfg, has_bool, has_cfg, has_tmir)):
        lines += [
            f"{_GOLD}  Nothing here yet.{_RESET}",
            "",
            f"  Already have Boolean (0/1) training data? Set up training directly:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
            "",
            f"  Starting from raw (non-Boolean) or external data instead? Begin upstream:",
            f"    {_CYAN}cp examples/data_source_config.yaml /work/data_source_config.yaml{_RESET}",
            "",
            f"  Then, either way:",
            f"    {_CYAN}matador faena{_RESET}",
        ]
        return lines

    # ── Status summary ───────────────────────────────────────────────────────
    tick = f"{_GREEN}✓{_RESET}"
    dash = f"{_DIM}–{_RESET}"

    lines.append(f"  {'Status':}")
    lines.append(f"  {'─' * 56}")

    # Raw data ingestion (optional — a user with their own Boolean data
    # skips straight to training config below)
    if has_raw:
        lines.append(f"  {tick} raw data ingested  {_DIM}{state['raw_data']}{_RESET}")
    elif has_ds_cfg:
        lines.append(f"  {dash} raw data ingested  {_DIM}configured, not yet run{_RESET}")

    # Booleanization (optional — same caveat)
    if has_bool:
        report = state["boolean_report"]
        lines.append(f"  {tick} data booleanized   {_DIM}{report.parent}{_RESET}")
    elif has_bool_cfg:
        lines.append(f"  {dash} data booleanized   {_DIM}configured, not yet run{_RESET}")

    # Training config
    if has_cfg:
        lines.append(f"  {tick} training config  {_DIM}{cfg}{_RESET}")
    else:
        lines.append(f"  {dash} training config  {_DIM}not found{_RESET}")

    # Trained model
    if has_tmir and meta:
        n_c  = meta.get("n_clauses_total", "?")
        n_cls = meta.get("n_classes", "?")
        n_f  = meta.get("n_features", "?")
        fp   = meta.get("model_fingerprint", "")
        fp_s = f"  …{fp[-12:]}" if fp else ""
        lines.append(f"  {tick} model trained    {_DIM}{n_c} clauses · {n_cls} classes · {n_f} features{fp_s}{_RESET}")
    elif has_tmir:
        lines.append(f"  {tick} model trained    {_DIM}{model.name if model else 'TMIR/'}{_RESET}")
    else:
        lines.append(f"  {dash} model trained    {_DIM}not yet{_RESET}")

    # RTL
    backends = state.get("rtl_backends", [])
    if backends:
        lines.append(f"  {tick} RTL generated    {_DIM}{' + '.join(backends)}{_RESET}")
    else:
        lines.append(f"  {dash} RTL generated    {_DIM}not yet{_RESET}")

    # Provenance
    if has_prov:
        lines.append(f"  {tick} provenance       {_DIM}report filed{_RESET}")
    else:
        lines.append(f"  {dash} provenance       {_DIM}not yet{_RESET}")

    lines.append("")

    # ── Next actions (show everything applicable) ────────────────────────────
    lines.append(f"  {'Available actions':}")
    lines.append(f"  {'─' * 56}")

    if has_ds_cfg and not has_raw:
        lines += [
            f"  Fetch/materialize raw data:",
            f"    {_CYAN}matador ingest --config {state['data_source_config']}{_RESET}",
            "",
        ]

    if has_raw and not has_bool:
        if not has_bool_cfg:
            lines += [
                f"  Booleanize the ingested data  {_DIM}(copy a config template first){_RESET}:",
                f"    {_CYAN}cp examples/booleanisation_config.yaml /work/booleanisation_config.yaml{_RESET}",
                f"    {_DIM}  point raw_npz at: {state['raw_data']}{_RESET}",
                f"    {_CYAN}matador booleanize --config /work/booleanisation_config.yaml{_RESET}",
                "",
            ]
        else:
            lines += [
                f"  Booleanize the ingested data:",
                f"    {_CYAN}matador booleanize --config {state['booleanisation_config']}{_RESET}",
                "",
            ]

    if has_bool and not has_cfg:
        report = state["boolean_report"]
        name = report.name.removesuffix("_report.json")
        train_txt = report.parent / f"{name}_train.txt"
        test_txt = report.parent / f"{name}_test.txt"
        lines += [
            f"  Train on the booleanized data  {_DIM}(copy a config template first){_RESET}:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
            f"    {_DIM}  set train_data: {train_txt}{_RESET}",
            f"    {_DIM}  set test_data:  {test_txt}{_RESET}",
            f"    {_CYAN}matador train --config /work/training_config.yaml{_RESET}",
            "",
        ]

    if has_cfg and not has_tmir:
        lines += [
            f"  Train the model:",
            f"    {_CYAN}matador train --config {cfg}{_RESET}",
            "",
        ]

    if has_tmir:
        if val_cfg and isinstance(val_cfg, Path):
            lines += [
                f"  Validate model accuracy:",
                f"    {_CYAN}matador validate --config {val_cfg}{_RESET}",
                "",
            ]

        if not has_rtl:
            from matador.backends.registry import list_backends as _lb, describe as _desc
            lines.append(f"  Generate RTL  {_DIM}(copy a config template first){_RESET}:")
            for bk in _lb():
                lines.append(f"    {_CYAN}matador generate --backend {bk} --config /work/{bk}.yaml{_RESET}")
                lines.append(f"    {_DIM}  cp examples/{bk}.yaml /work/{bk}.yaml{_RESET}")
            lines.append("")
        else:
            from matador.backends.registry import get as _get_backend

            for bk in backends:
                lines += [
                    f"  Simulate RTL ({bk}):",
                    f"    {_CYAN}matador simulate --backend {bk} --config /work/{bk}.yaml{_RESET}",
                    "",
                ]
                try:
                    supports_reprogramming = _get_backend(bk)().supports_reprogramming
                except Exception:
                    supports_reprogramming = False
                if supports_reprogramming:
                    lines += [
                        f"  Prove multi-model reprogramming ({bk})  {_DIM}(copy a config template first){_RESET}:",
                        f"    {_CYAN}cp examples/reprogram_config.yaml /work/reprogram_config.yaml{_RESET}",
                        f"    {_CYAN}matador reprogram-suite --backend {bk} --config /work/{bk}.yaml "
                        f"--reprogram-config /work/reprogram_config.yaml{_RESET}",
                        "",
                    ]
            lines += [
                f"  Emulate (software, no simulator needed):",
                f"    {_CYAN}matador emulate --backend {backends[0]} --config /work/{backends[0]}.yaml --verify{_RESET}",
                "",
            ]

        if model and not has_prov:
            lines += [
                f"  Provenance report  {_DIM}(ROM-based inference over full test set){_RESET}:",
                f"    {_CYAN}matador provenance --model {model} --test-data /work/data/test.txt{_RESET}",
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

    # ── Bull ──────────────────────────────────────────────────────────────
    for line in _BULL.splitlines():
        print(f"{_GOLD}{line}{_RESET}")

    # ── Registered backends (from registry — dynamic) ──────────────────────
    try:
        from matador.backends.registry import list_backends as _lb, describe as _desc
        print(f"  {_DIM}Registered backends:{_RESET}")
        for bk in _lb():
            print(f"    {_CYAN}{bk:<22}{_RESET}  {_DIM}{_desc(bk)}{_RESET}")
        print()
    except Exception:
        pass

    # ── DM guidance ───────────────────────────────────────────────────────
    for line in _dm(state):
        print(line)

    print()
    print(f"  {_DIM}All commands:  matador --help{_RESET}")
    print(f"  {_DIM}Guided flow:   matador faena{_RESET}")
    print()

    return True
