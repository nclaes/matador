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
    # on the output_dir used during generation.
    state["rtl_tiled"]     = any((_WORK / p).exists() for p in
                                  ["tiled/RTL", "TMIR/tiled/RTL"])
    state["rtl_hardwired"] = any((_WORK / p).exists() for p in
                                  ["hardwired/RTL", "TMIR/hardwired/RTL"])

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

    has_cfg  = bool(state.get("training_config"))
    has_tmir = bool(state.get("tmir_npz") or state.get("tmir_yaml"))
    has_rtl  = bool(state.get("rtl_tiled") or state.get("rtl_hardwired"))
    has_prov = bool(state.get("provenance"))
    val_cfg  = state.get("val_config")
    model    = state.get("tmir_npz") or state.get("tmir_yaml")
    meta     = _read_model_meta(state) if has_tmir else {}

    cfg = state.get("training_config")

    # ── Nothing here at all ─────────────────────────────────────────────────
    if not has_cfg and not has_tmir:
        lines += [
            f"{_GOLD}  Nothing here yet.{_RESET}",
            "",
            f"  Set up your training configuration to begin:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
            "",
            f"  Edit it for your dataset and hyperparameters, then:",
            f"    {_CYAN}matador faena{_RESET}",
        ]
        return lines

    # ── Status summary ───────────────────────────────────────────────────────
    tick = f"{_GREEN}✓{_RESET}"
    dash = f"{_DIM}–{_RESET}"

    lines.append(f"  {'Status':}")
    lines.append(f"  {'─' * 56}")

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
    backends = []
    if state.get("rtl_tiled"):     backends.append("tiled")
    if state.get("rtl_hardwired"): backends.append("hardwired")
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
            lines += [
                f"  Generate RTL  {_DIM}(create /work/generate_config.yaml first){_RESET}:",
                f"    {_CYAN}matador generate --config /work/generate_config.yaml{_RESET}",
                "",
            ]
        else:
            for bk in backends:
                lines += [
                    f"  Simulate RTL ({bk}):",
                    f"    {_CYAN}matador simulate --backend {bk} --config /work/generate_config.yaml{_RESET}",
                    "",
                ]
            lines += [
                f"  Emulate (software, no simulator needed):",
                f"    {_CYAN}matador emulate --backend {backends[0]} --config /work/generate_config.yaml --verify{_RESET}",
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

    # ── DM guidance ───────────────────────────────────────────────────────
    for line in _dm(state):
        print(line)

    print()
    print(f"  {_DIM}All commands:  matador --help{_RESET}")
    print(f"  {_DIM}Guided flow:   matador faena{_RESET}")
    print()

    return True
