"""GtkWaveGenerator — generates .gtkw save files for each RTL module.

Each file defines named signal groups, default radix, and a sensible zoom
level so engineers can open GTKWave and immediately see meaningful structure
without manual reorganisation.

GTKWave save-file format reference:
  https://github.com/gtkwave/gtkwave/blob/master/gtkwave4/doc/gtkwave3.pdf §C
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Signal attribute codes (GTKWave format)
# ---------------------------------------------------------------------------
# @28  = hex display, standard wire
# @22  = binary, standard wire
# @200 = group separator
# @800200 = group header (named group)
# @8   = decimal signed

_HEX    = "@28"
_BIN    = "@22"
_DEC    = "@8"
_SDEC   = "@408"    # signed decimal
_GROUP  = "@800200"  # group open/close
_BLANK  = "@200"     # blank separator


def _header(dumpfile: str, timescale_ns: int = 10) -> str:
    return (
        f"[dumpfile] \"{dumpfile}\"\n"
        f"[timestart] 0\n"
        f"[size] 1920 1080\n"
        f"[zoom] -22\n"
        f"[pos] 0 0\n"
        f"*-22 0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1\n"
    )


def _group(name: str) -> str:
    return f"{_GROUP}\n-{name}\n"


def _group_end(name: str) -> str:
    return f"{_GROUP}\n-END_{name}\n"


def _sig(path: str, fmt: str = _HEX) -> str:
    return f"{fmt}\n{path}\n"


def _blank(label: str = "") -> str:
    if label:
        return f"{_BLANK}\n-{label}\n"
    return f"{_BLANK}\n"


# ---------------------------------------------------------------------------
# GtkWaveGenerator
# ---------------------------------------------------------------------------

class GtkWaveGenerator:
    """Generates per-module .gtkw save files for the TM accelerator RTL.

    Parameterised from a TMAccelerator instance so signal widths and counts
    are always consistent with the generated Verilog.
    """

    def __init__(self, accel) -> None:
        """
        Args:
            accel: TMAccelerator instance (already derived params).
        """
        self.accel = accel

    def generate_all(self, sim_dir: Path) -> dict:
        """Write all .gtkw files into sim_dir.

        Returns:
            dict mapping tb_name → Path of written file.
        """
        writers = {
            "tb_system":      self._gtkw_system,
            "tb_axis_fifo":   self._gtkw_axis_fifo,
            "tb_clause_eval": self._gtkw_clause_eval,
            "tb_score_acc":   self._gtkw_score_acc,
            "tb_argmax":      self._gtkw_argmax,
        }
        result = {}
        for name, fn in writers.items():
            content = fn()
            path = sim_dir / f"{name}.gtkw"
            path.write_text(content)
            result[name] = path
        return result

    # ------------------------------------------------------------------
    # tb_system.gtkw — top-level end-to-end view
    # ------------------------------------------------------------------

    def _gtkw_system(self) -> str:
        a = self.accel
        C = a.n_classes
        CT = a.n_clauses_total
        CS = a.clause_slice
        lines = [_header("tb_system.vcd")]

        # FSM state
        lines.append(_group("FSM"))
        lines.append(_sig("tb_system.dut.state", _HEX))
        lines.append(_sig("tb_system.dut.busy",  _BIN))
        lines.append(_group_end("FSM"))

        # AXI-Stream input
        lines.append(_blank())
        lines.append(_group("AXI-Stream Input"))
        lines.append(_sig("tb_system.s_tvalid", _BIN))
        lines.append(_sig("tb_system.s_tready", _BIN))
        lines.append(_sig("tb_system.s_tdata",  _HEX))
        lines.append(_sig("tb_system.s_tlast",  _BIN))
        lines.append(_group_end("AXI-Stream Input"))

        # Tile counters
        lines.append(_blank())
        lines.append(_group("Tile Counters"))
        lines.append(_sig("tb_system.dut.feat_cnt",        _DEC))
        lines.append(_sig("tb_system.dut.clause_slice_cnt", _DEC))
        lines.append(_sig("tb_system.dut.score_cnt",       _DEC))
        lines.append(_sig("tb_system.dut.beat_cnt",        _DEC))
        lines.append(_group_end("Tile Counters"))

        # TA action signals (generate block — clause_slice wires)
        lines.append(_blank())
        lines.append(_group("TA Actions"))
        for k in range(min(CS, 8)):
            lines.append(_sig(f"tb_system.dut.ta_actions_k[{k}]", _HEX))
        lines.append(_sig("tb_system.dut.tile_has_actions", _BIN))
        lines.append(_group_end("TA Actions"))

        # Clause accumulators (show up to 16)
        lines.append(_blank())
        lines.append(_group("Clause Accumulators"))
        for g in range(min(CT, 16)):
            lines.append(_sig(f"tb_system.dut.clause_pass[{g}]",        _BIN))
            lines.append(_sig(f"tb_system.dut.clause_has_actions[{g}]", _BIN))
        lines.append(_group_end("Clause Accumulators"))

        # Score decode
        lines.append(_blank())
        lines.append(_group("Score Decode"))
        lines.append(_sig("tb_system.dut.score_class",    _DEC))
        lines.append(_sig("tb_system.dut.score_is_pos",   _BIN))
        lines.append(_sig("tb_system.dut.score_active",   _BIN))
        lines.append(_group_end("Score Decode"))

        # Scores (flat + per-class)
        lines.append(_blank())
        lines.append(_group("Scores"))
        lines.append(_sig("tb_system.dut.scores_flat", _HEX))
        for c in range(C):
            lines.append(_sig(f"tb_system.dut.scores[{c}]", _SDEC))
        lines.append(_group_end("Scores"))

        # AXI-Stream output
        lines.append(_blank())
        lines.append(_group("AXI-Stream Output"))
        lines.append(_sig("tb_system.m_tvalid", _BIN))
        lines.append(_sig("tb_system.m_tready", _BIN))
        lines.append(_sig("tb_system.m_tdata",  _HEX))
        lines.append(_sig("tb_system.m_tlast",  _BIN))
        lines.append(_group_end("AXI-Stream Output"))

        return "".join(lines)

    # ------------------------------------------------------------------
    # tb_axis_fifo.gtkw
    # ------------------------------------------------------------------

    def _gtkw_axis_fifo(self) -> str:
        lines = [_header("tb_axis_fifo.vcd")]

        lines.append(_group("S-Port (Input)"))
        lines.append(_sig("tb_axis_fifo.s_tvalid", _BIN))
        lines.append(_sig("tb_axis_fifo.s_tready", _BIN))
        lines.append(_sig("tb_axis_fifo.s_tdata",  _HEX))
        lines.append(_sig("tb_axis_fifo.s_tlast",  _BIN))
        lines.append(_group_end("S-Port (Input)"))

        lines.append(_blank())
        lines.append(_group("M-Port (Output)"))
        lines.append(_sig("tb_axis_fifo.m_tvalid", _BIN))
        lines.append(_sig("tb_axis_fifo.m_tready", _BIN))
        lines.append(_sig("tb_axis_fifo.m_tdata",  _HEX))
        lines.append(_sig("tb_axis_fifo.m_tlast",  _BIN))
        lines.append(_group_end("M-Port (Output)"))

        lines.append(_blank())
        lines.append(_group("FIFO Internals"))
        lines.append(_sig("tb_axis_fifo.dut.wptr", _DEC))
        lines.append(_sig("tb_axis_fifo.dut.rptr", _DEC))
        lines.append(_sig("tb_axis_fifo.dut.full",  _BIN))
        lines.append(_sig("tb_axis_fifo.dut.empty", _BIN))
        lines.append(_group_end("FIFO Internals"))

        return "".join(lines)

    # ------------------------------------------------------------------
    # tb_clause_eval.gtkw
    # ------------------------------------------------------------------

    def _gtkw_clause_eval(self) -> str:
        lines = [_header("tb_clause_eval.vcd")]

        lines.append(_group("Inputs"))
        lines.append(_sig("tb_clause_eval.literals",       _BIN))
        lines.append(_sig("tb_clause_eval.ta_action_mask", _BIN))
        lines.append(_group_end("Inputs"))

        lines.append(_blank())
        lines.append(_group("Internals"))
        lines.append(_sig("tb_clause_eval.dut.no_actions", _BIN))
        lines.append(_sig("tb_clause_eval.dut.all_active", _BIN))
        lines.append(_group_end("Internals"))

        lines.append(_blank())
        lines.append(_group("Output"))
        lines.append(_sig("tb_clause_eval.active", _BIN))
        lines.append(_group_end("Output"))

        return "".join(lines)

    # ------------------------------------------------------------------
    # tb_score_acc.gtkw
    # ------------------------------------------------------------------

    def _gtkw_score_acc(self) -> str:
        C = self.accel.n_classes
        SW = self.accel.score_width
        lines = [_header("tb_score_acc.vcd")]

        lines.append(_group("Control"))
        lines.append(_sig("tb_score_acc.valid",    _BIN))
        lines.append(_sig("tb_score_acc.cls",      _DEC))
        lines.append(_sig("tb_score_acc.polarity", _BIN))
        lines.append(_sig("tb_score_acc.active",   _BIN))
        lines.append(_sig("tb_score_acc.clear",    _BIN))
        lines.append(_group_end("Control"))

        lines.append(_blank())
        lines.append(_group("Scores"))
        lines.append(_sig("tb_score_acc.scores_flat", _HEX))
        for c in range(C):
            lines.append(_sig(f"tb_score_acc.dut.scores[{c}]", _SDEC))
        lines.append(_group_end("Scores"))

        return "".join(lines)

    # ------------------------------------------------------------------
    # tb_argmax.gtkw
    # ------------------------------------------------------------------

    def _gtkw_argmax(self) -> str:
        lines = [_header("tb_argmax.vcd")]

        lines.append(_group("Input"))
        lines.append(_sig("tb_argmax.scores_flat", _HEX))
        lines.append(_group_end("Input"))

        lines.append(_blank())
        lines.append(_group("Output"))
        lines.append(_sig("tb_argmax.pred_class", _DEC))
        lines.append(_group_end("Output"))

        return "".join(lines)
