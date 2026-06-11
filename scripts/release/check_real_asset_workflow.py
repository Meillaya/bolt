#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

"""Validate the opt-in real-asset workflow release policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Final, TypeAlias

DEFAULT_WORKFLOW: Final[Path] = Path(".github/workflows/real-assets.yml")
DEFAULT_CI: Final[Path] = Path(".github/workflows/ci.yml")
REQUIRED_ORDER: Final[tuple[str, ...]] = (
    "validate-assets",
    "validate-tokenizer",
    "run-bonsai-golden",
    "run-bonsai-bench",
    "run-bonsai-q4-golden",
    "run-bonsai-q4-bench",
    "bonsai-researcher",
    "bonsai-q4-researcher",
)
FORBIDDEN_DEFAULT: Final[tuple[str, ...]] = (
    "validate-assets",
    "validate-tokenizer",
    "run-bonsai",
    "run-bonsai-golden",
    "run-bonsai-bench",
    "run-bonsai-q4-golden",
    "run-bonsai-q4-bench",
    "bonsai-researcher",
    "bonsai-q4-researcher",
    "~/models/",
    "BOLT_ASSET_MANIFEST",
)
USES_PATTERN: Final[re.Pattern[str]] = re.compile(r"^\s*uses:\s*([^\s#]+)", re.MULTILINE)
SHA_REF_PATTERN: Final[re.Pattern[str]] = re.compile(r"@[0-9a-fA-F]{40}$")
FIRST_PARTY_PREFIX: Final = "actions/"
EXCEPTION_MARKER: Final = "first-party-action-exception"
FailureList: TypeAlias = list[str]


@dataclass(frozen=True, slots=True)
class RealAssetReport:
    """Static report for the real-asset workflow."""

    workflow: Path
    failures: tuple[str, ...]

    def is_valid(self) -> bool:
        """Return true when there are no policy violations."""
        return not self.failures


def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


def check_manual_self_hosted(text: str, failures: FailureList) -> None:
    """Require explicit manual opt-in and self-hosted runner labels."""
    if "workflow_dispatch:" not in text:
        failures.append("real-asset workflow must be workflow_dispatch-only")
    if "pull_request:" in text or "push:" in text:
        failures.append("real-asset workflow must not run on push or pull_request")
    for label in ("self-hosted", "macOS", "ARM64"):
        if label not in text:
            failures.append(f"missing self-hosted runner label: {label}")
    if "permissions:\n  contents: read" not in text:
        failures.append("workflow must use contents: read permissions")
    if "secrets." in text or "ANTHROPIC_API_KEY" in text:
        failures.append("workflow must not require provider secrets")
    if "data/" in text or "~/models/" in text:
        failures.append("workflow must not hard-code local data/model paths")


def check_required_order(text: str, failures: FailureList) -> None:
    """Require all commands in the approved real-asset order."""
    cursor = -1
    for command in REQUIRED_ORDER:
        found = text.find(command, cursor + 1)
        if found == -1:
            failures.append(f"missing required command: {command}")
            continue
        if found < cursor:
            failures.append(f"command out of order: {command}")
        cursor = found


def check_action_pinning(text: str, failures: FailureList) -> None:
    """Require third-party SHA pinning or first-party exceptions."""
    for match in USES_PATTERN.finditer(text):
        ref = match.group(1)
        name = ref.split("@", maxsplit=1)[0]
        if ref.startswith(FIRST_PARTY_PREFIX):
            marker = f"{EXCEPTION_MARKER}: {name}"
            if marker not in text and SHA_REF_PATTERN.search(ref) is None:
                failures.append(f"first-party action lacks exception marker: {ref}")
            continue
        if SHA_REF_PATTERN.search(ref) is None:
            failures.append(f"third-party action is not SHA-pinned: {ref}")


def check_default_ci(ci_path: Path, failures: FailureList) -> None:
    """Ensure real-asset gates are absent from portable default CI."""
    if not ci_path.exists():
        failures.append(f"default CI missing: {ci_path}")
        return
    text = read_text(ci_path)
    for forbidden in FORBIDDEN_DEFAULT:
        if forbidden in text:
            failures.append(f"default CI includes real-asset gate or path: {forbidden}")


def check_workflow(path: Path, ci_path: Path) -> RealAssetReport:
    """Run the static checks."""
    failures: FailureList = []
    try:
        text = read_text(path)
    except FileNotFoundError:
        return RealAssetReport(path, (f"workflow missing: {path}",))
    check_manual_self_hosted(text, failures)
    check_required_order(text, failures)
    check_action_pinning(text, failures)
    check_default_ci(ci_path, failures)
    if "manifest_path" not in text or "shasum -a 256" not in text:
        failures.append("workflow must record manifest path and digest")
    if "validate_artifact.py" not in text:
        failures.append("workflow must validate produced artifacts")
    return RealAssetReport(path, tuple(failures))


def selected_path(args: tuple[str, ...]) -> Path:
    """Parse optional workflow path."""
    match args:
        case ():
            return DEFAULT_WORKFLOW
        case ("--help",) | ("-h",):
            print("usage: check_real_asset_workflow.py [.github/workflows/real-assets.yml]")
            raise SystemExit(0)
        case (raw_path,):
            return Path(raw_path)
        case _:
            print("usage: check_real_asset_workflow.py [.github/workflows/real-assets.yml]", file=sys.stderr)
            raise SystemExit(2)


def render(report: RealAssetReport) -> int:
    """Print checker result."""
    if report.is_valid():
        print(f"real-asset workflow OK: {report.workflow}")
        print("required order: " + " -> ".join(REQUIRED_ORDER))
        print("default CI exclusion: ok")
        return 0
    print(f"real-asset workflow FAILED: {report.workflow}", file=sys.stderr)
    for failure in report.failures:
        print(f"- {failure}", file=sys.stderr)
    return 1


def main() -> int:
    """CLI entrypoint."""
    return render(check_workflow(selected_path(tuple(sys.argv[1:])), DEFAULT_CI))


if __name__ == "__main__":
    raise SystemExit(main())
