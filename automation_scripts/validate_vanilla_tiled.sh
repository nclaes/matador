#!/usr/bin/env bash
#
# validate_vanilla_tiled.sh -- backend-specific validation for
# vanilla_tiled: trains one model, generates fresh RTL parameterized to
# exactly that model, then compiles/runs it through BOTH real simulators
# and the software emulator.
#
# `matador simulate` itself already compiles and runs every unit
# testbench under iverilog AND the system testbench under Verilator (when
# installed) in one call -- see matador.models.validator.validate_rtl's
# own docstring -- so no manual per-simulator orchestration is needed here
# the way vanilla_gp_tiled's reprogram-suite testbench requires (see
# validate_vanilla_gp_tiled.sh).
#
# Real network fetch + real training + real RTL generation + real
# simulation + real emulation, nothing mocked.
#
# Usage:
#   ./automation_scripts/validate_vanilla_tiled.sh [WORK_DIR]
#   WORK_DIR=/work ./automation_scripts/validate_vanilla_tiled.sh
#
# Tunable via environment variables:
#   MODEL_DATASET   registered dataset to train the model from (default:
#                   digits -- small, fast, verified recipe)
#   CLAUSES         PER-CLASS clause count (default: 20) -- vanilla
#                   (per_class) TMs report n_clauses_total = clauses *
#                   n_classes; see validate_vanilla_gp_tiled.sh's own note.
#   EPOCHS          training epochs (default: 2)
#   T_THRESHOLD     voting threshold (default: 200) -- vanilla_tiled has
#                   no known hard ceiling on this the way vanilla_gp_tiled's
#                   vendored core does (SCORE_WIDTH=6, T<=31), so the
#                   standard template default is safe to use here.
#
# Output: $WORK_DIR/automation_report/vanilla_tiled/ (report.md, report.json, logs/)
#
# Note on `matador emulate --verify`: used here. Its cross-check
# (matador.verification.compare.compare_emulator_to_reference) hardcodes
# the vanilla_tiled emulator class specifically -- this is the ONE backend
# it actually supports (a known, documented limitation for every other
# backend -- see docs/Developer.md's "Known limitation" note under The
# three-layer contract, and validate_vanilla_hardwired.sh's own note on
# why it deliberately does NOT use --verify).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

WORK_DIR="${1:-${WORK_DIR:-/work}}"
MODEL_DATASET="${MODEL_DATASET:-digits}"
CLAUSES="${CLAUSES:-20}"
EPOCHS="${EPOCHS:-2}"
T_THRESHOLD="${T_THRESHOLD:-200}"
BACKEND="vanilla_tiled"

if ! command -v matador >/dev/null 2>&1; then
    echo "error: 'matador' not found on PATH -- run this inside the dev container" >&2
    exit 1
fi
if ! command -v iverilog >/dev/null 2>&1; then
    echo "error: 'iverilog' not found on PATH -- run this inside the dev container" >&2
    exit 1
fi

REPORT_DIR="$WORK_DIR/automation_report/$BACKEND"
LOG_DIR="$REPORT_DIR/logs"
RESULTS_JSONL="$REPORT_DIR/results.jsonl"
mkdir -p "$LOG_DIR"
: > "$RESULTS_JSONL"   # truncate/start fresh each run

echo "Matador $BACKEND validation (generate + simulate + emulate)"
echo "  work dir : $WORK_DIR"
echo "  report   : $REPORT_DIR"
echo "  model dataset=$MODEL_DATASET  clauses=$CLAUSES  epochs=$EPOCHS  T=$T_THRESHOLD"
echo ""

RUN_START=$(date +%s)

# ── Step 1: train the model ─────────────────────────────────────────────────

echo "=== $MODEL_DATASET ==="
if ! train_one_model "$MODEL_DATASET" "$WORK_DIR" "$LOG_DIR" "$CLAUSES" "$EPOCHS" "$T_THRESHOLD"; then
    echo "training failed for $MODEL_DATASET -- cannot validate $BACKEND without a model."
    RUN_END=$(date +%s)
    generate_report "$RESULTS_JSONL" "$REPORT_DIR/report.json" "$REPORT_DIR/report.md" \
        "$WORK_DIR" "$RUN_START" "$RUN_END" "Matador $BACKEND validation report"
    exit $?
fi
read -r TMIR_PATH N_FEATURES N_CLASSES < "$WORK_DIR/.trained_${MODEL_DATASET}.txt"
echo "  trained: $TMIR_PATH ($N_FEATURES features, $N_CLASSES classes)"
echo ""

# ── Step 2: generate + simulate + emulate ────────────────────────────────────

echo "=== $BACKEND ==="

accel_cfg="$WORK_DIR/${BACKEND}.yaml"
cat > "$accel_cfg" <<EOF
model_path: $TMIR_PATH
output_dir: $WORK_DIR
axis_data_width: 32
fifo_depth: 16
EOF

if run_step "$BACKEND" "generate" "$LOG_DIR/generate.log" \
        matador generate --backend "$BACKEND" --config "$accel_cfg"; then

    # Runs every unit testbench under iverilog AND the system testbench
    # under Verilator (when installed) in this one call.
    run_step "$BACKEND" "simulate" "$LOG_DIR/simulate.log" \
        matador simulate --backend "$BACKEND" --config "$accel_cfg"

    run_step "$BACKEND" "emulate" "$LOG_DIR/emulate.log" \
        matador emulate --backend "$BACKEND" --config "$accel_cfg" --verify
else
    echo "  generate failed -- skipping simulate/emulate"
fi
echo ""

RUN_END=$(date +%s)

# ── Report ────────────────────────────────────────────────────────────────

generate_report "$RESULTS_JSONL" "$REPORT_DIR/report.json" "$REPORT_DIR/report.md" \
    "$WORK_DIR" "$RUN_START" "$RUN_END" "Matador $BACKEND validation report"
STATUS=$?

echo ""
echo "Full report written to:"
echo "  $REPORT_DIR/report.md    (human-readable -- share this one)"
echo "  $REPORT_DIR/report.json  (structured)"
echo "  $LOG_DIR/                (full logs per step)"

exit "$STATUS"
