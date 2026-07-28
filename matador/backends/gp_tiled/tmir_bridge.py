"""Bridge between Matador's TMIR schema and GP_TM_Inference_Accelerator's
TMModel representation.

This module implements the mapping that GP_TM_Inference_Accelerator's own
``py/tm_emulator.py::load_tmir()`` left as an explicit stub, pending access
to the TMIR schema:

    TMIR.representation (ta_states/includes) -> TMModel.include[g] bitmask

Bit convention (shared by both sides — see tm_ir.py's Architecture and
tm_emulator.py's TMModel docstring): for global clause g, bit f (f <
n_features) is the positive literal of feature f, bit n_features+f is the
negated literal of feature f.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from matador.backends.gp_tiled.vendor import tm_emulator
from matador.backends.gp_tiled.vendor.tm_emulator import TMModel

# NOTE: do not `from tm_emulator import FEAT_SLICE, CLAUSE_SLICE` here — those
# are now config-driven (sync_capacity() patches them per call), so a
# module-import-time binding would silently go stale. Always read
# config.feat_slice / config.clause_slice instead.

if TYPE_CHECKING:
    from matador.backends.gp_tiled.config import GPTiledAcceleratorConfig
    from matador.ir.tm_ir import TMIR


# Hard ceilings imposed by the AXI-Stream header's own wire format (see
# tm_accel_gp.v:22-30) — word1/word2 8-bit sub-fields (n_classes,
# clauses_per_class, threshold, n_beats, n_feat_slices, n_clause_slices) and
# word3 16-bit sub-fields (n_clauses_total, n_tiles). The vendored core's
# internal FSM registers were widened to match these exactly (see
# tm_accel_gp.v's capacity register table), so these are now the true
# ceilings — not an arbitrary internal-register limitation. Going beyond
# them requires changing the wire protocol itself (a breaking change to the
# header format), out of scope for this backend.
HARD_MAX_CLASSES = 255
HARD_MAX_CLAUSES_TOTAL = 65535
HARD_MAX_FEAT_SLICES = 255
HARD_MAX_CLAUSE_SLICES = 255
HARD_MAX_CLAUSES_PER_CLASS = 255
HARD_MAX_TILES = 65535
# threshold is VESTIGIAL (scores are unclamped, see score_acc_rt.v) --
# this is the header's 8-bit field width, not a hardware clamp bound.
HARD_MAX_THRESHOLD = 255


class GPCapacityError(ValueError):
    """Raised when a TMIR model's geometry exceeds the backend's configured
    (compile-time) capacity, or uses a feature GP_TM_Inference_Accelerator
    does not support."""


def sync_capacity(config: "GPTiledAcceleratorConfig") -> None:
    """Point the vendored tm_emulator's capacity constants at this backend's
    configured (not necessarily default) compile-time capacity.

    TMModel.validate()/pack_tiles()/encode_load_packet() read FEAT_SLICE,
    CLAUSE_SLICE, TILE_WIDTH, WORDS_PER_ROW, and the MAX_* ceilings as module
    globals (looked up dynamically at call time, not bound at import), so
    they must reflect the *configured* values — including feat_slice/
    clause_slice, which are no longer fixed at 32 — whenever the config
    overrides tm_emulator.py's own defaults.
    """
    tm_emulator.FEAT_SLICE = config.feat_slice
    tm_emulator.CLAUSE_SLICE = config.clause_slice
    tm_emulator.TILE_WIDTH = config.clause_slice * 2 * config.feat_slice
    tm_emulator.WORDS_PER_ROW = tm_emulator.TILE_WIDTH // tm_emulator.AXIS_W
    tm_emulator.MAX_CLASSES = config.max_classes
    tm_emulator.MAX_CLAUSES_TOTAL = config.max_clauses_total
    tm_emulator.MAX_FEAT_SLICES = config.max_feat_slices
    tm_emulator.MAX_CLAUSE_SLICES = config.max_clause_slices
    tm_emulator.MAX_TILES = config.max_feat_slices * config.max_clause_slices
    tm_emulator.MAX_THRESHOLD = HARD_MAX_THRESHOLD


def _row_to_bitmask(row: np.ndarray) -> int:
    """Pack a boolean literal row into GP's integer bitmask convention."""
    val = 0
    for l in np.flatnonzero(row):
        val |= 1 << int(l)
    return val


def tmir_to_tmmodel(tmir: "TMIR", name: str = "model") -> TMModel:
    """Convert a validated TMIR model into a GP TMModel.

    Requires ``architecture.clause_organization == "per_class"`` — GP's
    class/polarity decode (``g // cpc``, ``(g % cpc) < cpc // 2``) assumes
    this layout, the same constraint the hardwired backend enforces.
    """
    arch = tmir.architecture
    if arch.clause_organization != "per_class":
        raise GPCapacityError(
            "vanilla_gp_tiled only supports clause_organization='per_class', "
            f"got {arch.clause_organization!r}"
        )
    if arch.n_clauses_per_class is None or arch.n_clauses_per_class % 2 != 0:
        raise GPCapacityError(
            "vanilla_gp_tiled requires an even n_clauses_per_class "
            f"(GP splits clauses into positive/negative halves), got "
            f"{arch.n_clauses_per_class!r}"
        )

    inc = tmir._resolve_includes()
    if inc is None:
        raise GPCapacityError(
            "TMIR model has no ta_states or includes representation to derive "
            "GP Include bits from."
        )
    flat_inc = np.asarray(inc, dtype=bool).reshape(arch.n_clauses_total, arch.n_literals)

    include = [_row_to_bitmask(flat_inc[g]) for g in range(arch.n_clauses_total)]

    model = TMModel(
        n_features=arch.n_features,
        n_classes=arch.n_classes,
        clauses_per_class=arch.n_clauses_per_class,
        threshold=arch.threshold,
        include=include,
        name=name,
    )
    return model


def check_capacity(tmir: "TMIR", config: "GPTiledAcceleratorConfig") -> None:
    """Validate that a TMIR model's geometry fits within the backend's
    compile-time capacity maxima. Raises GPCapacityError otherwise.

    Also points the vendored tm_emulator's capacity globals at the
    configured (not necessarily default) values, so downstream calls to
    TMModel.validate()/encode_load_packet() apply the right ceilings.
    """
    sync_capacity(config)
    arch = tmir.architecture
    n_feat_slices = -(-arch.n_features // config.feat_slice)
    n_clause_slices = -(-arch.n_clauses_total // config.clause_slice)

    problems = []
    if arch.n_classes > config.max_classes:
        problems.append(
            f"n_classes={arch.n_classes} exceeds configured max_classes={config.max_classes}"
        )
    if arch.n_clauses_total > config.max_clauses_total:
        problems.append(
            f"n_clauses_total={arch.n_clauses_total} exceeds configured "
            f"max_clauses_total={config.max_clauses_total}"
        )
    if n_feat_slices > config.max_feat_slices:
        problems.append(
            f"model needs {n_feat_slices} feature slices ({arch.n_features} features / "
            f"feat_slice={config.feat_slice}), exceeds configured "
            f"max_feat_slices={config.max_feat_slices} (derived from "
            f"max_features={config.max_features}/feat_slice={config.feat_slice})"
        )
    if n_clause_slices > config.max_clause_slices:
        problems.append(
            f"model needs {n_clause_slices} clause slices ({arch.n_clauses_total} clauses / "
            f"clause_slice={config.clause_slice}), exceeds configured "
            f"max_clause_slices={config.max_clause_slices} (derived from "
            f"max_clauses_total={config.max_clauses_total}/clause_slice={config.clause_slice})"
        )
    if arch.threshold > HARD_MAX_THRESHOLD:
        problems.append(
            f"threshold={arch.threshold} exceeds the hard RTL limit of {HARD_MAX_THRESHOLD} "
            f"(the CMD_LOAD header's threshold field is 8 bits; threshold itself is "
            f"vestigial and has no effect on scoring, see score_acc_rt.v)"
        )
    if arch.n_clauses_per_class is not None and arch.n_clauses_per_class > HARD_MAX_CLAUSES_PER_CLASS:
        problems.append(
            f"n_clauses_per_class={arch.n_clauses_per_class} exceeds the protocol's hard "
            f"limit of {HARD_MAX_CLAUSES_PER_CLASS} (word1's clauses_per_class header field "
            f"is 8 bits)"
        )

    if problems:
        raise GPCapacityError(
            "TMIR model does not fit the vanilla_gp_tiled backend's configured capacity:\n  "
            + "\n  ".join(problems)
        )
