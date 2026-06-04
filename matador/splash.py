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
  ================================================================
   __  __       _        _____      _       ____      ___      ____
  |  \/  |     / \      |_   _|    / \     |  _ \    / _ \    |  _ \
  | \  / |    / _ \       | |     / _ \    | | | |  | | | |   | |_) |
  | |\/| |   / ___ \      | |    / ___ \   | |_| |  | |_| |   |  _ <
  |_|  |_|  /_/   \_\    _|_|   /_/   \_\  |____/    \___/    |_| \_\
  ================================================================
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
    state["rtl_dir"]         = (_WORK / "RTL").exists()
    state["provenance"]      = _find_newest("TMIR/provenance_report.json") or \
                               _find_newest("provenance_report.json")

    return state


# ---------------------------------------------------------------------------
# DM guidance
# ---------------------------------------------------------------------------

_BOLD  = "\033[1m"
_CYAN  = "\033[36m"
_GREEN = "\033[32m"
_GOLD  = "\033[33m"
_RED   = "\033[31m"
_DIM   = "\033[2m"
_RESET = "\033[0m"


def _dm(state: dict) -> list[str]:
    """Return contextual guidance lines based on workspace state."""
    lines: list[str] = []

    has_tmir = bool(state.get("tmir_npz") or state.get("tmir_yaml"))
    has_rtl  = bool(state.get("rtl_dir"))

    if not state:
        # /work not mounted
        lines += [
            f"{_GOLD}  The arena is empty.{_RESET}",
            "",
            "  Mount your work directory first:",
            f"    {_CYAN}make shell WORK_DIR=/path/to/your/data{_RESET}",
            "",
            "  Then copy the training template:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
        ]
        return lines

    if not state.get("training_config") and not has_tmir:
        lines += [
            f"{_GOLD}  Step 1 — Prepare your training configuration.{_RESET}",
            "",
            "  Create a training config in your work directory:",
            f"    {_CYAN}cp examples/training_config.yaml /work/training_config.yaml{_RESET}",
            f"    {_DIM}# edit it for your dataset, clause count, and hyperparameters{_RESET}",
            "",
            "  Then run the full guided flow:",
            f"    {_CYAN}matador faena{_RESET}",
        ]
        return lines

    if state.get("training_config") and not has_tmir:
        lines += [
            f"{_GOLD}  Step 2 — Train your Tsetlin Machine.{_RESET}",
            "",
            f"  Config found:  {_DIM}/work/training_config.yaml{_RESET}",
            "",
            f"    {_CYAN}matador train --config /work/training_config.yaml{_RESET}",
            "",
            f"  {_DIM}This produces TMIR files under /work/TMIR/{_RESET}",
        ]
        return lines

    if has_tmir and not has_rtl:
        model = state.get("tmir_npz") or state.get("tmir_yaml")
        lines += [
            f"{_GOLD}  Step 3 — Generate RTL for your accelerator.{_RESET}",
            "",
            f"  Trained model:  {_DIM}{model}{_RESET}",
            "",
            "  Choose your architecture:",
            f"    {_CYAN}matador generate --backend tiled     --config /work/accelerator_config.yaml{_RESET}",
            f"    {_DIM}  Sequential FSM + tile ROM.  Tune feat_slice / clause_slice.{_RESET}",
            "",
            f"    {_CYAN}matador generate --backend hardwired --config /work/hardwired_config.yaml{_RESET}",
            f"    {_DIM}  Combinational AND gates + adder tree.  Tune pipeline_stages.{_RESET}",
            "",
            "  Validate the model first (optional but recommended):",
            f"    {_CYAN}matador validate --config /work/TMIR/validation_config.yaml{_RESET}",
        ]
        return lines

    if has_tmir and has_rtl:
        model = state.get("tmir_npz") or state.get("tmir_yaml")
        lines += [
            f"{_GOLD}  RTL generated.  Choose your next move:{_RESET}",
            "",
            f"  {_GREEN}▶{_RESET} Simulate the RTL:",
            f"    {_CYAN}matador simulate --config /work/accelerator_config.yaml{_RESET}",
            "",
            f"  {_GREEN}▶{_RESET} Run the software emulator:",
            f"    {_CYAN}matador emulate  --config /work/accelerator_config.yaml --verify{_RESET}",
            "",
            f"  {_GREEN}▶{_RESET} Generate provenance report (ROM-based inference):",
            f"    {_CYAN}matador provenance --model {model} --test-data /work/data/test.txt{_RESET}",
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

    # ── Logo ──────────────────────────────────────────────────────────────
    for line in _LOGO.splitlines():
        print(f"{_GOLD}{line}{_RESET}")

    # ── Subtitle ──────────────────────────────────────────────────────────
    print(f"  {_DIM}Automated RTL Accelerator Generator for Tsetlin Machines  |  v{__version__}{_RESET}")
    print()
    print(f"  {_DIM}─────────────────────────────────────────────────────────{_RESET}")
    print()

    # ── DM guidance ───────────────────────────────────────────────────────
    for line in _dm(state):
        print(line)

    print()
    print(f"  {_DIM}All commands:  matador --help{_RESET}")
    print(f"  {_DIM}Guided flow:   matador faena{_RESET}")
    print()

    return True
