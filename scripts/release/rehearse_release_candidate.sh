#!/usr/bin/env bash
# Run a release-candidate rehearsal and aggregate fresh local evidence.
set -u -o pipefail

EVIDENCE_DIR=".omo/evidence"
SUMMARY="$EVIDENCE_DIR/release-candidate-summary.md"
LOG="$EVIDENCE_DIR/task-23-rc-rehearsal.log"
mkdir -p "$EVIDENCE_DIR"
: > "$LOG"

resolve_python() {
    if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
        printf '%s\n' "$PYTHON_BIN"
        return 0
    fi
    if [ -x /opt/homebrew/bin/python3 ]; then
        printf '%s\n' /opt/homebrew/bin/python3
        return 0
    fi
    command -v python3
}

PYTHON_BIN=$(resolve_python)
export PYTHON_BIN
PY_CMD=$(printf '%q' "$PYTHON_BIN")

now_utc() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

append_summary() { printf '%s\n' "$*" >> "$SUMMARY"; }

run_cmd() {
    local label=$1
    local command=$2
    local mode=$3
    local start rc end
    start=$(now_utc)
    {
        printf '\n--- %s ---\n' "$label"
        printf 'command: %s\n' "$command"
        printf 'cwd: %s\n' "$(pwd)"
        printf 'start: %s\n' "$start"
    } | tee -a "$LOG"
    bash -lc "$command" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    end=$(now_utc)
    {
        printf 'end: %s\n' "$end"
        printf 'exit_code: %s\n' "$rc"
    } | tee -a "$LOG"
    append_summary "| ${label} | ${mode} | \`${command}\` | ${rc} |"
    return "$rc"
}

record_skip() {
    local label=$1
    local reason=$2
    append_summary "| ${label} | SKIPPED | ${reason} | n/a |"
    printf 'SKIPPED %s: %s\n' "$label" "$reason" | tee -a "$LOG"
}

main() {
    local first_failure=0
    : > "$SUMMARY"
    append_summary "# Release candidate rehearsal"
    append_summary ""
    append_summary "- start_utc: $(now_utc)"
    append_summary "- git_commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)-dirty"
    append_summary "- python: $PYTHON_BIN"
    append_summary "- dirty_worktree: see section below"
    append_summary "- log: $LOG"
    append_summary ""

    append_summary "## Default gates"
    append_summary "| Gate | Mode | Command | Exit code |"
    append_summary "| --- | --- | --- | --- |"
    for item in \
        "versions|PASS_REQUIRED|${PY_CMD} scripts/release/check_versions.py" \
        "support matrix|PASS_REQUIRED|${PY_CMD} scripts/release/check_support_matrix.py" \
        "research scope|PASS_REQUIRED|${PY_CMD} scripts/release/check_research_scope.py" \
        "portable CI policy|PASS_REQUIRED|${PY_CMD} scripts/release/check_ci_policy.py .github/workflows/ci.yml" \
        "README/build sync|PASS_REQUIRED|${PY_CMD} scripts/release/check_readme_build_steps.py" \
        "engine tests|PASS_REQUIRED|cd engine && zig build test --summary all" \
        "engine build|PASS_REQUIRED|cd engine && zig build --summary all" \
        "Labrat tests|PASS_REQUIRED|cd labrat && zig build test --summary all" \
        "Labrat API offline|PASS_REQUIRED|cd labrat && zig build api-offline-test --summary all"; do
        IFS='|' read -r label mode command <<< "$item"
        run_cmd "$label" "$command" "$mode" || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    done

    append_summary ""
    append_summary "## Artifact validation"
    append_summary "| Gate | Mode | Command | Exit code |"
    append_summary "| --- | --- | --- | --- |"
    run_cmd "artifact fixtures" "${PY_CMD} scripts/release/validate_artifact.py --all-fixtures" "PASS_REQUIRED" || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    run_cmd "engine MNIST artifact" "cd engine && zig build run --summary all && cd .. && ${PY_CMD} scripts/release/validate_artifact.py artifacts/bolt-mnist-run.json" "PASS_REQUIRED" || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    run_cmd "Labrat agent artifact" "cd labrat && zig build mnist-agent --summary all && cd .. && ${PY_CMD} scripts/release/validate_artifact.py artifacts/labrat-mnist-agent.json" "PASS_REQUIRED" || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }

    append_summary ""
    append_summary "## Security/docs checks"
    append_summary "| Gate | Mode | Command | Exit code |"
    append_summary "| --- | --- | --- | --- |"
    for item in \
        "real asset workflow|PASS_REQUIRED|${PY_CMD} scripts/release/check_real_asset_workflow.py .github/workflows/real-assets.yml" \
        "provenance policy|PASS_REQUIRED|${PY_CMD} scripts/release/check_provenance_policy.py" \
        "performance policy|PASS_REQUIRED|${PY_CMD} scripts/release/check_performance_policy.py" \
        "macOS distribution docs|PASS_REQUIRED|${PY_CMD} scripts/release/check_macos_distribution_docs.py" \
        "production runbook|PASS_REQUIRED|${PY_CMD} scripts/release/check_runbook.py docs/production-runbook.md" \
        "evidence policy|PASS_REQUIRED|${PY_CMD} scripts/release/check_evidence_policy.py .omo/evidence"; do
        IFS='|' read -r label mode command <<< "$item"
        run_cmd "$label" "$command" "$mode" || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    done

    append_summary ""
    append_summary "## Optional Metal"
    append_summary "| Gate | Mode | Command/reason | Exit code |"
    append_summary "| --- | --- | --- | --- |"
    run_cmd "Metal workflow policy" "${PY_CMD} scripts/release/check_metal_workflow.py .github/workflows/metal.yml" "PASS_REQUIRED" || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    if [ "${BOLT_FORCE_OPTIONAL_SKIP:-0}" = "1" ]; then
        record_skip "Metal shader compile" "BOLT_FORCE_OPTIONAL_SKIP=1 requested optional skip rehearsal"
    else
        run_cmd "Metal shader compile" "cd engine && zig build test-metal-shaders --summary all" "OPTIONAL" || record_skip "Metal shader compile" "local Metal/Xcode capability unavailable; see $LOG"
    fi

    append_summary ""
    append_summary "## Optional real assets"
    append_summary "| Gate | Mode | Command/reason | Exit code |"
    append_summary "| --- | --- | --- | --- |"
    if [ "${BOLT_FORCE_OPTIONAL_SKIP:-0}" = "1" ]; then
        record_skip "real asset gates" "BOLT_FORCE_OPTIONAL_SKIP=1 requested optional skip rehearsal"
    elif [ "${BOLT_RUN_REAL_ASSETS:-0}" != "1" ]; then
        record_skip "real asset gates" "BOLT_RUN_REAL_ASSETS is not 1; opt-in real-asset workflow remains manual/self-hosted"
    else
        run_cmd "validate assets" "cd engine && zig build validate-assets -- ../artifacts/assets/bolt-parity/asset-manifest.json" "OPTIONAL" || record_skip "validate assets" "assets unavailable or incomplete; see $LOG"
        run_cmd "validate tokenizer" "cd engine && zig build validate-tokenizer -- ../artifacts/assets/bolt-parity/asset-manifest.json" "OPTIONAL" || record_skip "validate tokenizer" "tokenizer assets unavailable or incomplete; see $LOG"
    fi

    append_summary ""
    append_summary "## Skipped with reason"
    append_summary "Rows marked SKIPPED above are optional gates only; default gates remain blocking."

    append_summary ""
    append_summary "## Dirty worktree"
    append_summary '```text'
    git status --short --untracked-files=all >> "$SUMMARY" || true
    append_summary '```'

    append_summary ""
    append_summary "## Evidence links"
    find "$EVIDENCE_DIR" -maxdepth 1 -type f | sort | sed 's#^#- #' >> "$SUMMARY"
    append_summary ""
    append_summary "- end_utc: $(now_utc)"
    append_summary "- final_exit_code: $first_failure"
    cat "$SUMMARY"
    return "$first_failure"
}

main "$@"
