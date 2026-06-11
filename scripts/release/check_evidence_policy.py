#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/release/check_evidence_policy.py .omo/evidence
# python3 scripts/release/check_evidence_policy.py --malformed-fixture

"""Validate Bolt run-local release evidence names and DoneClaim metadata."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from typing import TypeAlias

TASK_NAME_RE = re.compile(r"^task-[1-9][0-9]*-[a-z0-9]+(?:-[a-z0-9]+)*\.(?:txt|log|md|json)$")
FINAL_NAME_RE = re.compile(r"^final-[a-z0-9]+(?:-[a-z0-9]+)*\.(?:txt|log|md|json)$")
RELEASE_SUMMARY_NAME = "release-candidate-summary.md"
HEADER_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
DIGEST_RE = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
EVIDENCE_EXTENSIONS = (".txt", ".log", ".md", ".json")
REQUIRED_DONECLAIM_HEADERS = (
    "command",
    "cwd",
    "timestamp",
    "exit_code",
    "commit",
    "toolchain",
    "stale_state",
)
HeaderMap: TypeAlias = dict[str, str]

T1_COMMANDS = (
    "cd engine && zig build test --summary all",
    "cd engine && zig build --summary all",
    "cd labrat && zig build test --summary all",
    "cd labrat && zig build api-offline-test --summary all",
)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    errors: tuple[str, ...]
    notes: tuple[str, ...]


def has_valid_evidence_name(path: Path) -> bool:
    is_task_name = TASK_NAME_RE.fullmatch(path.name) is not None
    is_final_name = FINAL_NAME_RE.fullmatch(path.name) is not None
    is_release_summary = path.name == RELEASE_SUMMARY_NAME
    return is_task_name or is_final_name or is_release_summary


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_headers(text: str) -> HeaderMap:
    headers: dict[str, str] = {}
    for line in text.splitlines():
        match_result = HEADER_RE.match(line)
        if match_result is not None:
            headers[match_result.group(1)] = match_result.group(2).strip()
    return headers


def has_doneclaim(text: str) -> bool:
    return any(line.strip() == "DoneClaim:" for line in text.splitlines())


def is_integer(text: str) -> bool:
    return re.fullmatch(r"-?[0-9]+", text) is not None


def validate_doneclaim(path: Path, text: str) -> tuple[str, ...]:
    headers = parse_headers(text)
    errors: list[str] = []
    for header in REQUIRED_DONECLAIM_HEADERS:
        if header not in headers or headers[header] == "":
            errors.append(f"{path}: missing DoneClaim metadata header `{header}:`")

    exit_code = headers.get("exit_code", "")
    timestamp = headers.get("timestamp", "")
    commit = headers.get("commit", "")
    stale_state = headers.get("stale_state", "")
    artifact_digest = headers.get("artifact_digest", "")

    if exit_code != "" and not is_integer(exit_code):
        errors.append(f"{path}: `exit_code:` must be an integer")
    if timestamp != "" and TIMESTAMP_RE.fullmatch(timestamp) is None:
        errors.append(f"{path}: `timestamp:` must be an ISO-8601 UTC timestamp ending in Z")
    if commit != "" and SHA_RE.fullmatch(commit) is None:
        errors.append(f"{path}: `commit:` must be a 7- to 40-character hex Git SHA")
    if stale_state not in ("", "rejected", "not_applicable"):
        errors.append(f"{path}: `stale_state:` must be `rejected` or `not_applicable`")
    if "artifact:" in headers or "artifact_path" in headers:
        if DIGEST_RE.fullmatch(artifact_digest) is None:
            errors.append(f"{path}: artifact claims require `artifact_digest: sha256:<64-hex>`")
    if artifact_digest != "" and DIGEST_RE.fullmatch(artifact_digest) is None:
        errors.append(f"{path}: `artifact_digest:` must use sha256:<64-hex>")

    return tuple(errors)


def validate_t1_baseline(path: Path, text: str) -> tuple[str, ...]:
    headers = parse_headers(text)
    errors: list[str] = []
    for command in T1_COMMANDS:
        if f"command: {command}" not in text:
            errors.append(f"{path}: missing T1 baseline command `{command}`")
    for header in ("command", "cwd", "exit_code"):
        if header not in headers or headers[header] == "":
            errors.append(f"{path}: missing T1 baseline metadata `{header}:`")
    if "toolchain:" not in text:
        errors.append(f"{path}: missing T1 baseline `toolchain:` block")
    if headers.get("exit_code", "") != "" and not is_integer(headers["exit_code"]):
        errors.append(f"{path}: T1 baseline `exit_code:` must be an integer")
    return tuple(errors)


def validate_file(path: Path) -> tuple[str, ...]:
    errors: list[str] = []
    if not has_valid_evidence_name(path):
        errors.append(f"{path}: invalid evidence file name")
    text = read_text(path)
    if path.name == "task-1-baseline.txt":
        errors.extend(validate_t1_baseline(path, text))
    if has_doneclaim(text):
        errors.extend(validate_doneclaim(path, text))
    return tuple(errors)


def validate_directory(evidence_dir: Path) -> ValidationReport:
    if not evidence_dir.exists():
        return ValidationReport(
            errors=(),
            notes=(f"bootstrap: evidence directory missing: {evidence_dir}",),
        )
    if not evidence_dir.is_dir():
        return ValidationReport(
            errors=(f"{evidence_dir}: evidence path is not a directory",),
            notes=(),
        )

    all_files = tuple(
        sorted(
            path
            for path in evidence_dir.iterdir()
            if path.is_file() and not path.name.startswith(".")
        )
    )
    evidence_files = tuple(path for path in all_files if path.suffix in EVIDENCE_EXTENSIONS)
    ignored_files = tuple(path for path in all_files if path.suffix not in EVIDENCE_EXTENSIONS)
    if len(evidence_files) == 0:
        return ValidationReport(
            errors=(),
            notes=(f"bootstrap: evidence directory is empty: {evidence_dir}",),
        )

    errors: list[str] = []
    notes: list[str] = []
    for path in evidence_files:
        errors.extend(validate_file(path))
    for path in ignored_files:
        notes.append(f"ignored non-evidence file with unsupported extension: {path}")

    t1_baseline = evidence_dir / "task-1-baseline.txt"
    if t1_baseline.exists():
        notes.append(f"validated T1 baseline evidence: {t1_baseline}")
    else:
        notes.append(
            "bootstrap: task-1-baseline.txt not present yet; "
            "T1 validation will activate once it exists"
        )

    return ValidationReport(errors=tuple(errors), notes=tuple(notes))


def run_malformed_fixture() -> int:
    with TemporaryDirectory(prefix="bolt-evidence-policy-") as tmp_dir:
        evidence_dir = Path(tmp_dir)
        malformed = evidence_dir / "task-6-malformed.txt"
        malformed.write_text(
            "DoneClaim:\n"
            "cwd: /tmp/bolt\n"
            "timestamp: 2026-06-10T21:34:24Z\n"
            "commit: 741b502\n"
            "toolchain: python 3.14.4\n"
            "stale_state: rejected\n",
            encoding="utf-8",
        )
        report = validate_directory(evidence_dir)
        print_report(report)
        if report.errors:
            print("malformed fixture: rejected as expected")
            return 1
        print("malformed fixture: unexpectedly accepted")
        return 2


def print_report(report: ValidationReport) -> None:
    for note in report.notes:
        print(note)
    for error in report.errors:
        print(f"ERROR: {error}")
    if not report.errors:
        print("evidence policy: ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--malformed-fixture":
        return run_malformed_fixture()
    if len(argv) != 2:
        print(
            "usage: check_evidence_policy.py <evidence-dir> | --malformed-fixture",
            file=sys.stderr,
        )
        return 2
    report = validate_directory(Path(argv[1]))
    print_report(report)
    if report.errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
