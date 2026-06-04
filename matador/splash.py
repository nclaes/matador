"""Matador splash screen and workspace-aware DM guidance.

Shown only when `matador` is invoked with no arguments on an interactive
terminal.  In scripts, pipes, and CI the splash is suppressed automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

from matador import __version__

# ---------------------------------------------------------------------------
# ASCII art
# ---------------------------------------------------------------------------

_LOGO = r"""
     __  __       _       _____      _       ____      ___      ____
    |  \/  |     / \     |_   _|    / \     |  _ \    / _ \    |  _ \
    | \  / |    / _ \      | |     / _ \    | | | |  | | | |   | |_) |
    | |\/| |   / ___ \     | |    / ___ \   | |_| |  | |_| |   |  _ <
    |_|  |_|  /_/   \_\    |_|   /_/   \_\  |____/    \___/    |_| \_\

"""

# ---------------------------------------------------------------------------
# ASCII bull
# ---------------------------------------------------------------------------

_BULL = r""" 
                                            ////____\\\
                                        ///////      \\\\\\
        ///\\\                        /|//               \\\
    /_/////\\\\                      //|/                  \\\\\\
    /||||||    ||||                  /|//                       \\\\\\
    ||||||||   |||                   //||                             \\\\
    |||\//    ||||               //////                                 \\\
            |||              //_////                                    ||||       ||
        |||||     //_//___//                                      ////|||\    /|||
        |||||||/////_/                                         ||//     \\\__|||||
            \||||/                                               ||             |//
            |||                                                   \|  ||||||/_//
            |||                                                       ||| ||
            ||||                                             |||       |__ ||
            ||                               \\         |||  |||           ||
            ||             /|\               ||\        ||  //|\\\         ||
        ////|            ||||_____\||\/____|||||\      ||||||   \||||\   /||||
        |////         /_//||||/      |||//______/|\\     ||| |||   || |\\_/////
        ||    //_///_//_// ||     |////_            \\\  |||||||  /||
    |||  ////_//        ||\\   |||                \\\|  ||   |//
    /|| |//               \\\\\||||                  \\\  |||///
    /// |||                    ||||||\                  \\|/||||
    |||  |||                     ||    ||\                |||   ||\
    ||\\//||\___________________/||\__/|||\______________/||\__/||\\_

    MATADOR: Automated RTL Accelerator Generator for Tsetlin Machines

"""

# ---------------------------------------------------------------------------
# Workspace state detection
# ---------------------------------------------------------------------------

_WORK = Path("/work")


def _find_newest(pattern: str) -> Path | None:
    matches = sorted(_WORK.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _workspace_state() -> dict:
    """Inspect /work and return a dict of what has been produced."""
    state: dict = {}

    if not _WORK.exists():
        return state

    state["training_config"] = (_WORK / "training_config.yaml").exists()
    state["tmir_npz"]        = _find_newest("TMIR/*.npz")
    state["tmir_yaml"]       = _find_newest("TMIR/*.yaml")
    state["val_config"]      = _find_newest("TMIR/validation_config.yaml") or \
                               (_WORK / "validation_config.yaml").exists()
    state["rtl_tiled"]       = (_WORK / "tiled" / "RTL").exists()
    state["rtl_hardwired"]   = (_WORK / "hardwired" / "RTL").exists()
    state["provenance"]      = _find_newest("TMIR/provenance_report.json") or \
                               _find_newest("provenance_report.json")

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
    """Return workspace-aware guidance lines. Goal: validated RTL."""
    lines: list[str] = []

    has_tmir     = bool(state.get("tmir_npz") or state.get("tmir_yaml"))
    has_rtl      = bool(state.get("rtl_tiled") or state.get("rtl_hardwired"))
    has_prov     = bool(state.get("provenance"))

    # ── No /work mounted ────────────────────────────────────────────────────
    if not state:
        lines += [
            f"{_GOLD}  No work directory mounted.{_RESET}",
            "",
            f"  The corrida cannot begin without one:",
            f"    {_CYAN}make shell WORK_DIR=/path/to/your/data{_RESET}",
        ]
        return lines

    # ── No training config and no model ─────────────────────────────────────
    if not state.get("training_config") and not has_tmir:
        lines += [
            f"{_GOLD}  Nothing here yet.{_RESET}",
            "",
            f"  Set up your training configuration to begin the corrida:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
            "",
            f"  Edit it for your dataset and hyperparameters, then:",
            f"    {_CYAN}matador faena{_RESET}",
            "",
            f"  {_DIM}Goal: validated RTL under /work/tiled/RTL/{_RESET}",
        ]
        return lines

    # ── Config ready, no model trained yet ──────────────────────────────────
    if state.get("training_config") and not has_tmir:
        lines += [
            f"{_GOLD}  Training configuration found.{_RESET}",
            f"  {_DIM}/work/training_config.yaml{_RESET}",
            "",
            f"  Train the Tsetlin Machine:",
            f"    {_CYAN}matador train --config /work/training_config.yaml{_RESET}",
            "",
            f"  {_DIM}Produces TMIR model + validation config + provenance script under /work/TMIR/{_RESET}",
        ]
        return lines

    # ── Model trained, no RTL yet ────────────────────────────────────────────
    if has_tmir and not has_rtl:
        meta  = _read_model_meta(state)
        model = state.get("tmir_npz") or state.get("tmir_yaml")

        if meta:
            fp    = meta.get("model_fingerprint", "")
            fp_id = f"…{fp[-16:]}" if fp else ""
            n_c   = meta.get("n_clauses_total", "?")
            n_cls = meta.get("n_classes", "?")
            n_f   = meta.get("n_features", "?")
            lines += [
                f"{_GOLD}  Model trained.{_RESET}",
                f"  {_DIM}{n_c} clauses · {n_cls} classes · {n_f} features · {fp_id}{_RESET}",
            ]
        else:
            lines += [
                f"{_GOLD}  Model trained.{_RESET}",
                f"  {_DIM}{model}{_RESET}",
            ]

        val_cfg = state.get("val_config")
        if val_cfg and isinstance(val_cfg, Path):
            lines += [
                "",
                f"  Validate accuracy before generating RTL:",
                f"    {_CYAN}matador validate --config {val_cfg}{_RESET}",
            ]

        lines += [
            "",
            f"  Generate RTL for all configured backends:",
            f"    {_CYAN}matador generate --config /work/generate_config.yaml{_RESET}",
            "",
            f"  {_DIM}Writes to /work/tiled/RTL/ and /work/hardwired/RTL/{_RESET}",
            f"  {_DIM}Goal: RTL testbenches passing under both backends{_RESET}",
        ]
        return lines

    # ── RTL generated ────────────────────────────────────────────────────────
    if has_tmir and has_rtl:
        model  = state.get("tmir_npz") or state.get("tmir_yaml")
        meta   = _read_model_meta(state)
        backends = []
        if state.get("rtl_tiled"):     backends.append("tiled")
        if state.get("rtl_hardwired"): backends.append("hardwired")
        bk_str = " + ".join(backends)

        fp_id = ""
        if meta:
            fp = meta.get("model_fingerprint", "")
            fp_id = f"  {_DIM}…{fp[-16:]}{_RESET}" if fp else ""

        lines += [
            f"{_GOLD}  RTL generated: {bk_str}.{_RESET}{fp_id}",
            "",
            f"  Run RTL testbenches to validate:",
        ]
        for bk in backends:
            lines.append(
                f"    {_CYAN}matador simulate --backend {bk} --config /work/generate_config.yaml{_RESET}"
            )

        if not has_prov and model:
            lines += [
                "",
                f"  Verify inference parity with the ROM:",
                f"    {_CYAN}matador provenance --model {model} --test-data /work/data/test.txt{_RESET}",
            ]

        if has_prov:
            lines += [
                "",
                f"  {_GREEN}Provenance report written.{_RESET}",
                f"  {_DIM}The corrida is complete when all testbenches pass and provenance is filed.{_RESET}",
            ]
        else:
            lines += [
                "",
                f"  {_DIM}The corrida is complete when testbenches pass and provenance is filed.{_RESET}",
            ]
        return lines

    return [f"  {_DIM}Type 'matador --help' to see all commands.{_RESET}"]


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
        print(f"{_RED}{line}{_RESET}")

    # ── DM guidance ───────────────────────────────────────────────────────
    for line in _dm(state):
        print(line)

    print()
    print(f"  {_DIM}All commands:  matador --help{_RESET}")
    print(f"  {_DIM}Guided flow:   matador faena{_RESET}")
    print()

    return True
