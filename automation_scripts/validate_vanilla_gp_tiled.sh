#!/usr/bin/env bash
#
# validate_vanilla_gp_tiled.sh -- backend-specific validation for
# vanilla_gp_tiled, the one runtime-reprogrammable backend. Proves it
# really can be loaded with MORE THAN ONE trained model and tested
# correctly, both:
#   1. In RTL simulation, via `matador reprogram-suite` -- one synthesized
#      bundle, reprogrammed across every model in one continuous
#      LOAD -> INFER -> LOAD -> INFER run, with EVERY replayed vector
#      individually checked (not a spot check -- see
#      examples/reprogram_config.yaml's own docs for how the testbench
#      does this) -- compiled and run through BOTH iverilog and Verilator
#      against the identical vectors.
#   2. In software emulation, via `matador emulate` -- one real run per
#      model against that model's own embedded test vectors, proving the
#      emulator (not just the RTL) is also correct for each model that
#      will be loaded.
#
# See validate_vanilla_tiled.sh / validate_vanilla_hardwired.sh for the
# other two (non-reprogrammable) backends -- each backend gets its own
# standalone script rather than one script looping over several, since
# what's actually worth proving differs per backend (this one's whole
# point is multi-model reprogramming; the other two are single-model
# generate/simulate/emulate).
#
# Real network fetches, real training, real iverilog simulation -- nothing
# mocked. If this passes, multi-model reprogramming genuinely works, for
# both verification paths matador offers.
#
# Usage:
#   ./automation_scripts/validate_vanilla_gp_tiled.sh [WORK_DIR]
#   WORK_DIR=/work ./automation_scripts/validate_vanilla_gp_tiled.sh
#
# Tunable via environment variables:
#   MODELS        space-separated registered dataset keys to use as the
#                 reprogramming set (default: "digits statlog mammographic"
#                 -- small, fast, verified-recipe datasets). Needs >= 2;
#                 each becomes one model loaded into the SAME synthesized
#                 accelerator.
#   CLAUSES       PER-CLASS clause count for every model (default: 20) --
#                 vanilla (per_class) TMs report n_clauses_total =
#                 clauses * n_classes, which is what actually has to fit
#                 the synthesized capacity (computed automatically below as
#                 CLAUSES * the largest class count among MODELS).
#                 Deliberately identical and small across models, both for
#                 speed and so they all cleanly fit one modest capacity.
#   EPOCHS        training epochs per model (default: 2)
#   N_SAMPLES     vectors checked per model in the RTL reprogram-suite
#                 simulation (default: 15) -- exhaustive over this many,
#                 not a sample of them: every one is individually verified.
#   TARGET_FPGA   target_fpga for the synthesized capacity (default: xc7z020)
#   T_THRESHOLD   voting threshold for every model (default: 15) -- the
#                 vendored gp_tiled core has a hard, non-configurable
#                 SCORE_WIDTH=6 limit of 31 on this value, regardless of
#                 the capacity fields above; unrelated to CLAUSES/classes.
#
# Output: $WORK_DIR/automation_report/reprogramming/ (report.md, report.json,
# logs/) -- kept separate from validate_all_datasets.sh's own report so the
# two scripts don't clobber each other if run back to back.
#
# Note on `matador emulate --verify`: intentionally NOT used here. Its
# cross-check (matador.verification.compare.compare_emulator_to_reference)
# hardcodes the vanilla_tiled emulator class and is a known, documented,
# not-yet-fixed limitation for every other backend including
# vanilla_gp_tiled (see docs/Developer.md's "Known limitation" note under
# The three-layer contract). Plain `matador emulate` (no --verify) still
# gives a real, meaningful check: each vector's emulator prediction is
# compared against that vector's OWN embedded expected_class (produced by
# the reference engine at training time), and the CLI exits non-zero on
# any mismatch -- exactly what run_step() below checks.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

WORK_DIR="${1:-${WORK_DIR:-/work}}"
read -r -a MODELS <<< "${MODELS:-digits statlog mammographic}"
CLAUSES="${CLAUSES:-20}"
EPOCHS="${EPOCHS:-2}"
N_SAMPLES="${N_SAMPLES:-15}"
TARGET_FPGA="${TARGET_FPGA:-xc7z020}"
# The vendored gp_tiled core's score accumulator is fixed at SCORE_WIDTH=6
# bits regardless of any config field -- T (voting threshold) must stay
# <=31 for ANY model destined for this backend, not just a capacity choice.
T_THRESHOLD="${T_THRESHOLD:-15}"
BACKEND="vanilla_gp_tiled"
FEAT_SLICE=32

if [ "${#MODELS[@]}" -lt 2 ]; then
    echo "error: need at least 2 MODELS to prove multi-model reprogramming (got: ${MODELS[*]})" >&2
    exit 1
fi

if ! command -v matador >/dev/null 2>&1; then
    echo "error: 'matador' not found on PATH -- run this inside the dev container" >&2
    exit 1
fi
if ! command -v iverilog >/dev/null 2>&1; then
    echo "error: 'iverilog' not found on PATH -- run this inside the dev container" >&2
    exit 1
fi

REPORT_DIR="$WORK_DIR/automation_report/reprogramming"
LOG_DIR="$REPORT_DIR/logs"
RESULTS_JSONL="$REPORT_DIR/results.jsonl"
mkdir -p "$LOG_DIR"
: > "$RESULTS_JSONL"   # truncate/start fresh each run

echo "Matador multi-model reprogramming validation"
echo "  work dir : $WORK_DIR"
echo "  report   : $REPORT_DIR"
echo "  models   : ${MODELS[*]}"
echo "  clauses=$CLAUSES  epochs=$EPOCHS  n_samples=$N_SAMPLES  target_fpga=$TARGET_FPGA"
echo ""

# has_verified_recipe()/train_one_model() come from _lib.sh (shared with
# validate_vanilla_tiled.sh/validate_vanilla_hardwired.sh)

# ── Step 1: train one model per dataset in MODELS ───────────────────────────

RUN_START=$(date +%s)

TRAINED_NAMES=()
TRAINED_TMIR=()
MAX_FEATURES=0
MAX_CLASSES=0

for ds in "${MODELS[@]}"; do
    echo "=== $ds ==="
    if train_one_model "$ds" "$WORK_DIR" "$LOG_DIR" "$CLAUSES" "$EPOCHS" "$T_THRESHOLD"; then
        read -r tmir_path n_features n_classes < "$WORK_DIR/.trained_${ds}.txt"
        TRAINED_NAMES+=("$ds")
        TRAINED_TMIR+=("$tmir_path")
        [ "$n_features" -gt "$MAX_FEATURES" ] && MAX_FEATURES=$n_features
        [ "$n_classes" -gt "$MAX_CLASSES" ] && MAX_CLASSES=$n_classes
        echo "  trained: $tmir_path ($n_features features, $n_classes classes)"
    else
        echo "  training failed for $ds -- excluded from the reprogram set"
    fi
    echo ""
done

if [ "${#TRAINED_NAMES[@]}" -lt 2 ]; then
    echo "error: fewer than 2 models trained successfully (${#TRAINED_NAMES[@]}/${#MODELS[@]}) --"
    echo "       cannot prove multi-model reprogramming. See logs in $LOG_DIR."
    record "suite" "reprogram-suite" "FAIL" "fewer than 2 models trained successfully" "0"
    RUN_END=$(date +%s)
    generate_report "$RESULTS_JSONL" "$REPORT_DIR/report.json" "$REPORT_DIR/report.md" \
        "$WORK_DIR" "$RUN_START" "$RUN_END" "Matador multi-model reprogramming validation report"
    exit 1
fi

echo "Reprogram set (${#TRAINED_NAMES[@]}): ${TRAINED_NAMES[*]}"
echo ""

# ── Step 2: size and generate the synthesized accelerator ───────────────────
# Round up to a full feat_slice so n_feat_slices divides evenly. `clauses:`
# in each training config is PER CLASS for a vanilla (per_class) TM -- the
# TMIR's actual n_clauses_total is clauses * n_classes -- so the worst case
# across the reprogram set is CLAUSES * MAX_CLASSES (the model with the
# most classes), not CLAUSES alone. clause_slice stays equal to CLAUSES so
# every model occupies a whole number of clause slices.

CAP_FEATURES=$(( ((MAX_FEATURES + FEAT_SLICE - 1) / FEAT_SLICE) * FEAT_SLICE ))
CAP_CLASSES=$(( MAX_CLASSES + 2 ))
CAP_CLAUSES_TOTAL=$(( CLAUSES * MAX_CLASSES ))

ACCEL_CFG="$WORK_DIR/${BACKEND}.yaml"
cat > "$ACCEL_CFG" <<EOF
model_path: ${TRAINED_TMIR[0]}
output_dir: $WORK_DIR
axis_data_width: 32
fifo_depth: 16
target_fpga: $TARGET_FPGA
feat_slice: $FEAT_SLICE
clause_slice: $CLAUSES
max_features: $CAP_FEATURES
max_clauses_total: $CAP_CLAUSES_TOTAL
max_classes: $CAP_CLASSES
EOF

echo "Capacity: max_features=$CAP_FEATURES max_clauses_total=$CAP_CLAUSES_TOTAL max_classes=$CAP_CLASSES (target_fpga=$TARGET_FPGA)"
echo ""

if ! run_step "suite" "generate" "$LOG_DIR/suite_generate.log" \
        matador generate --backend "$BACKEND" --config "$ACCEL_CFG"; then
    echo "generate failed -- cannot proceed to reprogram-suite or emulation."
    RUN_END=$(date +%s)
    generate_report "$RESULTS_JSONL" "$REPORT_DIR/report.json" "$REPORT_DIR/report.md" \
        "$WORK_DIR" "$RUN_START" "$RUN_END" "Matador multi-model reprogramming validation report"
    exit $?
fi

# ── Step 3: reprogram_config.yaml listing every trained model ───────────────

REPROGRAM_CFG="$WORK_DIR/reprogram_config.yaml"
{
    echo "steps:"
    for i in "${!TRAINED_NAMES[@]}"; do
        ds="${TRAINED_NAMES[$i]}"
        echo "  - model: ${TRAINED_TMIR[$i]}"
        echo "    dataset: $WORK_DIR/booleanised/${ds}_test.txt"
        echo "    n_samples: $N_SAMPLES"
        echo "    seed: 0"
        echo "    name: $ds"
    done
} > "$REPROGRAM_CFG"

# ── Step 4: build the reprogram-suite testbench + stimulus ──────────────────

if ! run_step "suite" "reprogram-suite-build" "$LOG_DIR/suite_reprogram_build.log" \
        matador reprogram-suite --backend "$BACKEND" --config "$ACCEL_CFG" --reprogram-config "$REPROGRAM_CFG"; then
    echo "reprogram-suite build failed -- see $LOG_DIR/suite_reprogram_build.log"
else
    # ── Step 5a: actually compile + run the RTL simulation via iverilog ─────
    RTL_DIR="$WORK_DIR/$BACKEND/RTL"
    SRC_DIR="$RTL_DIR/src"
    SIM_DIR="$RTL_DIR/sim"
    TB_PATH="$RTL_DIR/tb/tb_reprogram_suite.v"
    OUT_BIN="$SIM_DIR/tb_reprogram_suite"
    SRCS=(
        "$SRC_DIR/axis_fifo.v" "$SRC_DIR/clause_eval.v" "$SRC_DIR/tile_mem.v"
        "$SRC_DIR/score_acc_rt.v" "$SRC_DIR/argmax_rt.v" "$SRC_DIR/tm_accel_gp.v"
    )

    SIM_LOG="$LOG_DIR/suite_reprogram_iverilog.log"
    start=$(date +%s)
    if iverilog -g2001 -Wall -Wno-timescale -o "$OUT_BIN" "${SRCS[@]}" "$TB_PATH" > "$SIM_LOG" 2>&1 \
            && (cd "$SIM_DIR" && vvp "$(basename "$OUT_BIN")") >> "$SIM_LOG" 2>&1; then
        sim_status="PASS"
    else
        sim_status="FAIL"
    fi
    end=$(date +%s)
    if ! grep -q "ALL TESTS PASSED" "$SIM_LOG"; then
        sim_status="FAIL"   # vvp itself always exits 0 -- the pass/fail verdict is in its output, not its exit code
    fi
    sim_detail=$(grep -E "PASSED|FAILED|TIMEOUT" "$SIM_LOG" | tail -n 3 | tr '\n' ' ' | tr -s ' ')
    record "suite" "reprogram-suite-simulate-iverilog" "$sim_status" "$sim_detail" "$((end - start))"
    echo "  [$sim_status] reprogram-suite-simulate-iverilog (suite_reprogram_iverilog.log)"

    # ── Step 5b: also compile + run the SAME vectors through Verilator ──────
    # (matador generate copies RTL/sim/verilator/{Makefile,tb_top.cpp} into
    # the bundle -- that harness is generic, replaying any --stim/--exp
    # pair, so it can check the identical reprogram_stimulus.memh/
    # reprogram_expected.memh iverilog just checked above. Two independent
    # simulators against the same vectors catches a simulator-specific bug
    # that either alone could miss. Skipped, not failed, if verilator isn't
    # installed -- it's a heavier dependency than iverilog.)
    if command -v verilator >/dev/null 2>&1; then
        VERILATOR_DIR="$SIM_DIR/verilator"
        VLOG="$LOG_DIR/suite_reprogram_verilator.log"
        start=$(date +%s)
        if make -C "$VERILATOR_DIR" \
                ARGS="--stim ../reprogram_stimulus.memh --exp ../reprogram_expected.memh" \
                run > "$VLOG" 2>&1; then
            vsim_status="PASS"
        else
            vsim_status="FAIL"
        fi
        end=$(date +%s)
        if ! grep -q "ALL TESTS PASSED" "$VLOG"; then
            vsim_status="FAIL"
        fi
        vsim_detail=$(grep -E "PASSED|FAILED|TIMEOUT|[Ee]rror" "$VLOG" | tail -n 3 | tr '\n' ' ' | tr -s ' ')
        record "suite" "reprogram-suite-simulate-verilator" "$vsim_status" "$vsim_detail" "$((end - start))"
        echo "  [$vsim_status] reprogram-suite-simulate-verilator (suite_reprogram_verilator.log)"
    else
        record "suite" "reprogram-suite-simulate-verilator" "SKIPPED" "verilator not found on PATH" "0"
        echo "  SKIPPED reprogram-suite-simulate-verilator (verilator not on PATH)"
    fi
fi
echo ""

# ── Step 6: emulate each model individually against the SAME synthesized
#     bundle's config, proving the software emulator is correct per model ───

for i in "${!TRAINED_NAMES[@]}"; do
    ds="${TRAINED_NAMES[$i]}"
    per_model_cfg="$WORK_DIR/${BACKEND}_${ds}.yaml"
    sed "s#^model_path:.*#model_path: ${TRAINED_TMIR[$i]}#" "$ACCEL_CFG" > "$per_model_cfg"
    run_step "$ds" "emulate" "$LOG_DIR/${ds}_emulate.log" \
        matador emulate --backend "$BACKEND" --config "$per_model_cfg"
done
echo ""

RUN_END=$(date +%s)

# ── Report ────────────────────────────────────────────────────────────────

generate_report "$RESULTS_JSONL" "$REPORT_DIR/report.json" "$REPORT_DIR/report.md" \
    "$WORK_DIR" "$RUN_START" "$RUN_END" "Matador multi-model reprogramming validation report"
STATUS=$?

echo ""
echo "Full report written to:"
echo "  $REPORT_DIR/report.md    (human-readable -- share this one)"
echo "  $REPORT_DIR/report.json  (structured)"
echo "  $LOG_DIR/                (full logs per step)"
echo ""
echo "Also written (inspect directly if useful):"
echo "  $WORK_DIR/reprogram_config.yaml                 (the multi-model step list used)"
echo "  $WORK_DIR/$BACKEND/RTL/sim/reprogram_manifest.txt  (which beat belongs to which model/vector)"

exit "$STATUS"
