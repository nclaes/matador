#!/usr/bin/env bash
#
# _lib.sh -- shared helpers for automation_scripts/*.sh. Not meant to be run
# directly: source it after setting RESULTS_JSONL (and, before calling
# generate_report, after the run loop has finished).
#
# set -uo pipefail is expected to already be set by the sourcing script.

record() {   # record <dataset> <step> <status> <detail> <duration_s>
    python3 -c '
import json, sys
dataset, step, status, detail, duration, path = sys.argv[1:7]
with open(path, "a") as f:
    f.write(json.dumps({
        "dataset": dataset, "step": step, "status": status,
        "detail": detail, "duration_s": float(duration),
    }) + "\n")
' "$1" "$2" "$3" "$4" "$5" "$RESULTS_JSONL"
}

run_step() {   # run_step <dataset> <step_name> <log_file> <command...>
    local dataset="$1" step="$2" logfile="$3" start end status detail
    shift 3
    start=$(date +%s)
    if "$@" > "$logfile" 2>&1; then
        status="PASS"
    else
        status="FAIL"
    fi
    end=$(date +%s)
    detail=$(tail -n 5 "$logfile" | tr '\n' ' ' | tr -s ' ')
    record "$dataset" "$step" "$status" "$detail" "$((end - start))"
    echo "  [$status] $step (${logfile##*/})"
    [ "$status" = "PASS" ]
}

has_verified_recipe() {
    python3 -c '
import sys
from matador.preprocessing import registry
sys.exit(0 if registry.get_default_booleanization(sys.argv[1]) is not None else 1)
' "$1"
}

# Trains exactly one model for <dataset> into <work_dir>, logging under
# <log_dir>. On success, writes its TMIR yaml path / n_features_bool /
# n_classes as one space-separated line to <work_dir>/.trained_<dataset>.txt
# for the caller to read back -- a plain state file rather than eval-based
# pseudo-references, so there's no risk of this function's own local
# variable names shadowing whatever the caller happens to use for its own.
# Handles both the --dataset shortcut (verified recipe) and the
# --show-recipe -> --config round-trip (unverified recipe) booleanize
# paths, and returns non-zero (without aborting the caller's loop) on any
# real failure -- callers decide whether to exclude that dataset and carry
# on, or treat it as fatal.
train_one_model() {   # train_one_model <dataset> <work_dir> <log_dir> <clauses> <epochs> <t_threshold>
    local ds="$1" work_dir="$2" log_dir="$3" clauses="$4" epochs="$5" t_threshold="$6"

    if ! run_step "$ds" "ingest" "$log_dir/${ds}_ingest.log" \
            matador ingest --dataset "$ds" --output-dir "$work_dir/raw"; then
        return 1
    fi

    if has_verified_recipe "$ds"; then
        if ! run_step "$ds" "booleanize" "$log_dir/${ds}_booleanize.log" \
                matador booleanize --dataset "$ds" --raw-dir "$work_dir/raw" --output-dir "$work_dir/booleanised"; then
            return 1
        fi
    else
        local skeleton_cfg="$work_dir/${ds}_booleanisation_config.yaml"
        if ! matador booleanize --dataset "$ds" --show-recipe \
                --raw-dir "$work_dir/raw" --output-dir "$work_dir/booleanised" \
                > "$skeleton_cfg" 2> "$log_dir/${ds}_show_recipe.log"; then
            record "$ds" "booleanize" "FAIL" "--show-recipe itself failed, see ${ds}_show_recipe.log" "0"
            return 1
        fi
        if ! run_step "$ds" "booleanize" "$log_dir/${ds}_booleanize.log" \
                matador booleanize --config "$skeleton_cfg"; then
            return 1
        fi
    fi

    local report="$work_dir/booleanised/${ds}_report.json"
    if [ ! -f "$report" ]; then
        record "$ds" "booleanize" "FAIL" "report.json not found after a reported PASS" "0"
        return 1
    fi

    local n_classes n_features
    read -r n_classes n_features < <(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["n_classes"], d["n_features_bool"])
' "$report")

    local cfg_path="$work_dir/${ds}_training_config.yaml"
    cat > "$cfg_path" <<EOF
tm_type: vanilla
clauses: $clauses
classes: $n_classes
features: $n_features
s: 5.0
T: $t_threshold
epochs: $epochs
max_included_literals: 32
seed: 42
train_data: $work_dir/booleanised/${ds}_train.txt
test_data: $work_dir/booleanised/${ds}_test.txt
model_name: $ds
output_dir: $work_dir
EOF

    if ! run_step "$ds" "train" "$log_dir/${ds}_train.log" matador train --config "$cfg_path"; then
        return 1
    fi

    local tmir_yaml
    tmir_yaml=$(find "$work_dir/TMIR/$ds" -maxdepth 1 -name 'TM_TMIR_*.yaml' | head -n1)
    if [ -z "$tmir_yaml" ]; then
        record "$ds" "train" "FAIL" "no TM_TMIR_*.yaml found under TMIR/$ds after a reported PASS" "0"
        return 1
    fi

    echo "$tmir_yaml $n_features $n_classes" > "$work_dir/.trained_${ds}.txt"
    return 0
}

generate_report() {   # generate_report <results_jsonl> <report_json> <report_md> <work_dir> <run_start> <run_end> <title>
    python3 -c '
import json, sys
from collections import defaultdict

results_path, report_json_path, report_md_path, work_dir, run_start, run_end, title = sys.argv[1:8]

rows = []
with open(results_path) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

by_dataset = defaultdict(list)
for r in rows:
    by_dataset[r["dataset"]].append(r)

n_pass = sum(1 for r in rows if r["status"] == "PASS")
n_fail = sum(1 for r in rows if r["status"] == "FAIL")
n_skip = sum(1 for r in rows if r["status"] == "SKIPPED")

summary = {
    "work_dir": work_dir,
    "run_started": int(run_start),
    "run_ended": int(run_end),
    "duration_s": int(run_end) - int(run_start),
    "groups": len(by_dataset),
    "steps_passed": n_pass,
    "steps_failed": n_fail,
    "steps_skipped": n_skip,
    "all_passed": n_fail == 0,
}

with open(report_json_path, "w") as f:
    json.dump({"summary": summary, "results": rows}, f, indent=2)

overall = "PASS" if summary["all_passed"] else "FAIL"

lines = []
lines.append("# %s" % title)
lines.append("")
lines.append("- Work dir: `%s`" % work_dir)
lines.append("- Duration: %ss" % summary["duration_s"])
lines.append("- Groups: %s" % summary["groups"])
lines.append("- Steps: %s passed, %s failed, %s skipped" % (n_pass, n_fail, n_skip))
lines.append("- **Overall: %s**" % overall)
lines.append("")
lines.append("| Group | Step | Status | Duration (s) | Detail |")
lines.append("|---|---|---|---|---|")
for ds in sorted(by_dataset):
    for r in by_dataset[ds]:
        detail = r["detail"].replace("|", "\\|")[:160]
        lines.append("| %s | %s | %s | %.0f | %s |" % (ds, r["step"], r["status"], r["duration_s"], detail))

with open(report_md_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print("")
print("Summary: %s passed, %s failed, %s skipped across %s group(s)" % (n_pass, n_fail, n_skip, len(by_dataset)))
print("Overall: %s" % overall)
sys.exit(0 if summary["all_passed"] else 1)
' "$1" "$2" "$3" "$4" "$5" "$6" "$7"
}
