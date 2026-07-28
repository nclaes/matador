#!/usr/bin/env bash
#
# validate_all_datasets.sh -- end-to-end functional validation of the whole
# ingest -> booleanize -> train -> validate pipeline, for EVERY dataset
# registered in data/Raw_Data_Bank.yaml, training TWO differently-sized
# models per dataset (same dataset, distinct model_name:) to also exercise
# the multi-model-per-dataset machinery.
#
# This is a CORRECTNESS smoke test, not a benchmark: clause counts and
# epochs default deliberately small so the whole sweep finishes in a
# reasonable time. Real UCI network fetches + real training, nothing
# mocked -- if it passes, the pipeline actually works end to end.
#
# Usage (run inside the matador dev container, `make shell WORK_DIR=...`):
#   ./automation_scripts/validate_all_datasets.sh [WORK_DIR]
#   WORK_DIR=/work ./automation_scripts/validate_all_datasets.sh
#
# WORK_DIR defaults to /work (the container's usual mount point) if neither
# the argument nor the WORK_DIR env var is given.
#
# Tunable via environment variables:
#   EPOCHS          training epochs per model       (default: 2)
#   SMALL_CLAUSES   clause count for the "small" model of each pair (default: 20)
#   LARGE_CLAUSES   clause count for the "large" model of each pair (default: 100)
#   SKIP_DATASETS   comma-separated dataset keys to skip, e.g. "emg,sports"
#                   (emg has ~4.2M raw rows with no verified windowed
#                   recipe, and sports' unverified skeleton recipe produces
#                   a very wide 45000-bit encoding -- both are real,
#                   correct runs, just slow; skip them for a quick pass)
#
# Output (all under $WORK_DIR/automation_report/):
#   logs/*.log      full stdout+stderr of every command run
#   results.jsonl   one JSON object per step, appended as the run progresses
#   report.json     the same data, structured + summarised
#   report.md       human-readable table -- share this one back for review
#
# Exit code is 0 only if every step of every dataset passed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$SCRIPT_DIR/_lib.sh"

# ── Setup ─────────────────────────────────────────────────────────────────

WORK_DIR="${1:-${WORK_DIR:-/work}}"
EPOCHS="${EPOCHS:-2}"
SMALL_CLAUSES="${SMALL_CLAUSES:-20}"
LARGE_CLAUSES="${LARGE_CLAUSES:-100}"
SKIP_DATASETS="${SKIP_DATASETS:-}"

if ! command -v matador >/dev/null 2>&1; then
    echo "error: 'matador' not found on PATH -- run this inside the dev container" >&2
    exit 1
fi

REPORT_DIR="$WORK_DIR/automation_report"
LOG_DIR="$REPORT_DIR/logs"
RESULTS_JSONL="$REPORT_DIR/results.jsonl"
mkdir -p "$LOG_DIR"
: > "$RESULTS_JSONL"   # truncate/start fresh each run

echo "Matador multi-dataset validation"
echo "  work dir : $WORK_DIR"
echo "  report   : $REPORT_DIR"
echo "  epochs=$EPOCHS  small_clauses=$SMALL_CLAUSES  large_clauses=$LARGE_CLAUSES"
[ -n "$SKIP_DATASETS" ] && echo "  skipping : $SKIP_DATASETS"
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────
# record()/run_step()/has_verified_recipe() come from _lib.sh (shared with
# the other automation_scripts/*.sh)

is_skipped() {
    [[ ",$SKIP_DATASETS," == *",$1,"* ]]
}

write_training_config() {   # write_training_config <path> <model_name> <clauses> <classes> <features> <train_txt> <test_txt>
    cat > "$1" <<EOF
tm_type: vanilla
clauses: $3
classes: $4
features: $5
s: 5.0
T: 200
epochs: $EPOCHS
max_included_literals: 32
seed: 42
train_data: $6
test_data: $7
model_name: $2
output_dir: $WORK_DIR
EOF
}

booleanize_dataset() {   # booleanize_dataset <dataset>
    local ds="$1" out_dir="$WORK_DIR/booleanised"
    if has_verified_recipe "$ds"; then
        run_step "$ds" "booleanize" "$LOG_DIR/${ds}_booleanize.log" \
            matador booleanize --dataset "$ds" --raw-dir "$WORK_DIR/raw" --output-dir "$out_dir"
    else
        local skeleton_cfg="$WORK_DIR/${ds}_booleanisation_config.yaml"
        if ! matador booleanize --dataset "$ds" --show-recipe \
                --raw-dir "$WORK_DIR/raw" --output-dir "$out_dir" \
                > "$skeleton_cfg" 2> "$LOG_DIR/${ds}_show_recipe.log"; then
            record "$ds" "booleanize" "FAIL" "--show-recipe itself failed, see ${ds}_show_recipe.log" "0"
            return 1
        fi
        run_step "$ds" "booleanize" "$LOG_DIR/${ds}_booleanize.log" \
            matador booleanize --config "$skeleton_cfg"
    fi
}

train_and_validate() {   # train_and_validate <dataset> <size> <clauses> <classes> <features> <train_txt> <test_txt>
    local ds="$1" size="$2" clauses="$3" classes="$4" features="$5" train_txt="$6" test_txt="$7"
    local model_name="${ds}_${size}"
    local cfg_path="$WORK_DIR/${model_name}_training_config.yaml"
    write_training_config "$cfg_path" "$model_name" "$clauses" "$classes" "$features" "$train_txt" "$test_txt"

    if run_step "$ds" "train_${size}" "$LOG_DIR/${model_name}_train.log" \
            matador train --config "$cfg_path"; then
        local val_cfg="$WORK_DIR/TMIR/${model_name}/validation_config.yaml"
        if [ -f "$val_cfg" ]; then
            run_step "$ds" "validate_${size}" "$LOG_DIR/${model_name}_validate.log" \
                matador validate --config "$val_cfg"
        else
            record "$ds" "validate_${size}" "FAIL" "validation_config.yaml not found after train" "0"
        fi
    fi
}

process_dataset() {   # process_dataset <dataset>
    local ds="$1"
    echo "=== $ds ==="

    if is_skipped "$ds"; then
        echo "  SKIPPED (SKIP_DATASETS)"
        record "$ds" "ingest" "SKIPPED" "excluded via SKIP_DATASETS" "0"
        return
    fi

    if ! run_step "$ds" "ingest" "$LOG_DIR/${ds}_ingest.log" \
            matador ingest --dataset "$ds" --output-dir "$WORK_DIR/raw"; then
        echo "  ingest failed -- skipping the rest of $ds"
        return
    fi

    if ! booleanize_dataset "$ds"; then
        echo "  booleanize failed -- skipping training for $ds"
        return
    fi

    local report="$WORK_DIR/booleanised/${ds}_report.json"
    if [ ! -f "$report" ]; then
        record "$ds" "booleanize" "FAIL" "report.json not found after a reported PASS" "0"
        return
    fi

    local n_classes n_features
    read -r n_classes n_features < <(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["n_classes"], d["n_features_bool"])
' "$report")

    local train_txt="$WORK_DIR/booleanised/${ds}_train.txt"
    local test_txt="$WORK_DIR/booleanised/${ds}_test.txt"

    train_and_validate "$ds" "small" "$SMALL_CLAUSES" "$n_classes" "$n_features" "$train_txt" "$test_txt"
    train_and_validate "$ds" "large" "$LARGE_CLAUSES" "$n_classes" "$n_features" "$train_txt" "$test_txt"
}

# ── Main ──────────────────────────────────────────────────────────────────

RUN_START=$(date +%s)

DATASETS=()
while IFS= read -r ds_line; do
    [ -n "$ds_line" ] && DATASETS+=("$ds_line")
done < <(python3 -c '
from matador.preprocessing.registry import list_datasets
for n in list_datasets():
    print(n)
')

echo "Registered datasets (${#DATASETS[@]}): ${DATASETS[*]}"
echo ""

for ds in "${DATASETS[@]}"; do
    process_dataset "$ds"
    echo ""
done

RUN_END=$(date +%s)

# ── Report ────────────────────────────────────────────────────────────────

generate_report "$RESULTS_JSONL" "$REPORT_DIR/report.json" "$REPORT_DIR/report.md" \
    "$WORK_DIR" "$RUN_START" "$RUN_END" "Matador multi-dataset validation report"
STATUS=$?

echo ""
echo "Full report written to:"
echo "  $REPORT_DIR/report.md    (human-readable -- share this one)"
echo "  $REPORT_DIR/report.json  (structured)"
echo "  $LOG_DIR/                (full logs per step)"

exit "$STATUS"
