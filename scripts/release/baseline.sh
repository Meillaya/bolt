#!/bin/sh
# Capture Bolt's portable baseline release gates with dirty-worktree protection.

set -u

EVIDENCE_DIR=".omo/evidence"
EVIDENCE_FILE="$EVIDENCE_DIR/task-1-baseline.txt"

mkdir -p "$EVIDENCE_DIR"

now_utc() {
    date -u '+%Y-%m-%dT%H:%M:%SZ'
}

status_path() {
    # git status --short --untracked-files=all prints two status columns, a
    # space, then the path. Rename entries include "old -> new"; validate the
    # destination path because that is the new dirty path in the worktree.
    path=$(printf '%s\n' "$1" | sed 's/^...//')
    case "$path" in
        *' -> '*) path=${path##* -> } ;;
    esac
    printf '%s\n' "$path"
}

matches_explicit_dirty_allowlist() {
    path=$1
    allowlist=${BOLT_BASELINE_ALLOWED_DIRTY:-}
    [ -n "$allowlist" ] || return 1

    old_ifs=$IFS
    IFS=:
    for item in $allowlist; do
        IFS=$old_ifs
        [ -n "$item" ] || continue
        case "$path" in
            "$item"|"$item"/*) return 0 ;;
        esac
        IFS=:
    done
    IFS=$old_ifs
    return 1
}

is_allowed_dirty_path() {
    case "$1" in
        .omo/*|.omo) return 0 ;;
        AGENTS.md|*/AGENTS.md) return 0 ;;
    esac
    matches_explicit_dirty_allowlist "$1"
}

check_dirty_worktree() {
    dirty=$(git status --short --untracked-files=all 2>&1)
    rc=$?
    if [ "$rc" -ne 0 ]; then
        printf 'dirty_check: git status failed with exit %s\n%s\n' "$rc" "$dirty" >&2
        return "$rc"
    fi

    bad_file=$(mktemp "${TMPDIR:-/tmp}/bolt-baseline-dirty.XXXXXX") || return 1
    printf '%s\n' "$dirty" | while IFS= read -r line; do
        [ -n "$line" ] || continue
        path=$(status_path "$line")
        if ! is_allowed_dirty_path "$path"; then
            printf '%s\n' "$line" >> "$bad_file"
        fi
    done

    if [ -s "$bad_file" ]; then
        printf 'refusing dirty worktree outside default .omo/AGENTS allowlist or BOLT_BASELINE_ALLOWED_DIRTY:\n' >&2
        cat "$bad_file" >&2
        rm -f "$bad_file"
        return 2
    fi

    rm -f "$bad_file"
    return 0
}

append_toolchain() {
    {
        printf 'toolchain:\n'
        printf '  git: '
        git --version 2>&1 || true
        printf '  zig: '
        zig version 2>&1 || true
        printf '  uname: '
        uname -a 2>&1 || true
    } >> "$EVIDENCE_FILE"
}

run_gate() {
    cmd=$1
    start=$(now_utc)
    cwd=$(pwd)

    {
        printf '\n--- gate ---\n'
        printf 'command: %s\n' "$cmd"
        printf 'cwd: %s\n' "$cwd"
        printf 'start: %s\n' "$start"
    } >> "$EVIDENCE_FILE"

    printf 'running: %s\n' "$cmd"
    sh -c "$cmd" >> "$EVIDENCE_FILE" 2>&1
    rc=$?
    end=$(now_utc)

    {
        printf 'end: %s\n' "$end"
        printf 'exit_code: %s\n' "$rc"
    } >> "$EVIDENCE_FILE"

    return "$rc"
}

main() {
    check_dirty_worktree
    dirty_rc=$?
    if [ "$dirty_rc" -ne 0 ]; then
        return "$dirty_rc"
    fi

    : > "$EVIDENCE_FILE"
    {
        printf 'task: T1 baseline evidence and dirty-worktree guard\n'
        printf 'command: bash scripts/release/baseline.sh\n'
        printf 'cwd: %s\n' "$(pwd)"
        printf 'start: %s\n' "$(now_utc)"
    } >> "$EVIDENCE_FILE"
    append_toolchain

    first_failure=0

    run_gate 'cd engine && zig build test --summary all' || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    run_gate 'cd engine && zig build --summary all' || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    run_gate 'cd labrat && zig build test --summary all' || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }
    run_gate 'cd labrat && zig build api-offline-test --summary all' || { rc=$?; [ "$first_failure" -eq 0 ] && first_failure=$rc; }

    {
        printf '\nend: %s\n' "$(now_utc)"
        printf 'exit_code: %s\n' "$first_failure"
    } >> "$EVIDENCE_FILE"

    printf 'baseline evidence: %s\n' "$EVIDENCE_FILE"
    return "$first_failure"
}

main "$@"
