#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Final, TypeAlias

DEFAULT_WORKFLOW: Final[Path] = Path(".github/workflows/ci.yml")
REQUIRED_COMMANDS: Final[tuple[str, ...]] = (
    "cd engine && zig build test --summary all",
    "cd engine && zig build --summary all",
    "cd labrat && zig build test --summary all",
    "cd labrat && zig build api-offline-test --summary all",
)
FORBIDDEN_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("workflow references secrets", re.compile(r"\bsecrets\.")),
    ("default workflow enables Labrat live mode", re.compile(r"\bLABRAT_LIVE\s*=\s*1\b")),
    ("default workflow runs real-asset Bonsai gates", re.compile(r"\brun-bonsai(?:-|\b)")),
    ("default workflow runs Metal shader gate", re.compile(r"\btest-metal-shaders\b")),
    ("workflow mentions provider key", re.compile(r"ANTHROPIC_API_KEY|OPENAI_API_KEY|API_TOKEN")),
)
USES_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)
SHA_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"@[0-9a-fA-F]{40}$")
FIRST_PARTY_ACTION_PREFIX: Final = "actions/"
FIRST_PARTY_EXCEPTION_MARKER: Final = "first-party-action-exception"
RETENTION_PATTERN: Final[re.Pattern[str]] = re.compile(r"retention-days:\s*(\d+)")

FailureList: TypeAlias = list[str]


@dataclass(frozen=True, slots=True)
class CiPolicyReport:
    """Static CI policy checker result."""

    path: Path
    failures: tuple[str, ...]
    required_commands: tuple[str, ...]
    action_refs: tuple[str, ...]

    def is_valid(self) -> bool:
        """Return whether the workflow satisfies portable release policy."""
        return not self.failures


def read_text(path: Path) -> str:
    """Read a UTF-8 workflow file."""
    return path.read_text(encoding="utf-8")


def require_present(text: str, failures: FailureList) -> None:
    """Record missing required portable gate commands."""
    for command in REQUIRED_COMMANDS:
        if command not in text:
            failures.append(f"missing required command: {command}")


def reject_forbidden(text: str, failures: FailureList) -> None:
    """Record forbidden release-policy patterns."""
    for label, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            failures.append(label)


def check_permissions(text: str, failures: FailureList) -> None:
    """Require least-privilege top-level read-only contents permission."""
    if "permissions:" not in text:
        failures.append("missing top-level permissions block")
    if not re.search(r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$", text):
        failures.append("permissions must include contents: read")
    if re.search(r"(?m)^\s+id-token:\s*write\s*$", text):
        failures.append("default CI must not request id-token: write")


def action_refs(text: str) -> tuple[str, ...]:
    """Return all action references used by the workflow."""
    return tuple(match.group(1) for match in USES_PATTERN.finditer(text))


def check_action_pinning(text: str, failures: FailureList) -> tuple[str, ...]:
    """Require third-party SHA pinning and explicit first-party exceptions."""
    refs = action_refs(text)
    for ref in refs:
        name = ref.split("@", maxsplit=1)[0]
        if ref.startswith(FIRST_PARTY_ACTION_PREFIX):
            marker = f"{FIRST_PARTY_EXCEPTION_MARKER}: {name}"
            if marker not in text and not SHA_REF_PATTERN.search(ref):
                failures.append(f"first-party action lacks exception marker: {ref}")
            continue
        if not SHA_REF_PATTERN.search(ref):
            failures.append(f"third-party action is not SHA-pinned: {ref}")
    return refs


def check_artifact_retention(text: str, failures: FailureList) -> None:
    """Require short artifact retention if artifacts are uploaded."""
    if "upload-artifact" not in text:
        failures.append("workflow must upload short-retention logs")
        return
    match = RETENTION_PATTERN.search(text)
    if match is None:
        failures.append("upload-artifact step missing retention-days")
        return
    days = int(match.group(1))
    if days < 1 or days > 7:
        failures.append(f"artifact retention must be 1-7 days, got {days}")


def check_workflow(path: Path) -> CiPolicyReport:
    """Run all static policy checks for the portable CI workflow."""
    failures: FailureList = []
    try:
        text = read_text(path)
    except FileNotFoundError:
        return CiPolicyReport(path, (f"workflow missing: {path}",), REQUIRED_COMMANDS, ())

    require_present(text, failures)
    reject_forbidden(text, failures)
    check_permissions(text, failures)
    refs = check_action_pinning(text, failures)
    check_artifact_retention(text, failures)
    if "runs-on: macos" not in text:
        failures.append("default health job must run on hosted macOS")
    return CiPolicyReport(path, tuple(failures), REQUIRED_COMMANDS, refs)


def selected_path(args: tuple[str, ...]) -> Path:
    """Parse the optional workflow path."""
    match args:
        case ():
            return DEFAULT_WORKFLOW
        case ("--help",) | ("-h",):
            print("usage: check_ci_policy.py [.github/workflows/ci.yml]")
            raise SystemExit(0)
        case (raw_path,):
            return Path(raw_path)
        case _:
            print("usage: check_ci_policy.py [.github/workflows/ci.yml]", file=sys.stderr)
            raise SystemExit(2)


def render(report: CiPolicyReport) -> int:
    """Print checker output and return a process status."""
    if report.is_valid():
        print(f"CI policy OK: {report.path}")
        print(f"required commands: {len(report.required_commands)}")
        print("actions: " + ", ".join(report.action_refs))
        return 0

    print(f"CI policy FAILED: {report.path}", file=sys.stderr)
    for failure in report.failures:
        print(f"- {failure}", file=sys.stderr)
    return 1


def main() -> int:
    """CLI entrypoint."""
    return render(check_workflow(selected_path(tuple(sys.argv[1:]))))


if __name__ == "__main__":
    raise SystemExit(main())
