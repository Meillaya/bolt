#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/release/check_metal_workflow.py [.github/workflows/metal.yml]
# 3. Or with system Python from the repository root:
#      python3 scripts/release/check_metal_workflow.py [.github/workflows/metal.yml]
# ──────────────────

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Final, TypeAlias

DEFAULT_WORKFLOW: Final[Path] = Path(".github/workflows/metal.yml")
DEFAULT_CI: Final[Path] = Path(".github/workflows/ci.yml")
METAL_COMMAND: Final = "cd engine && zig build test-metal-shaders --summary all"
REQUIRED_METADATA: Final[tuple[str, ...]] = (
    "sw_vers",
    "xcodebuild -version",
    "xcrun --sdk macosx --show-sdk-path",
    "xcrun --sdk macosx --show-sdk-version",
    "xcrun -find metal",
    "system_profiler SPDisplaysDataType",
)
FORBIDDEN_DEFAULT_EVENTS: Final[tuple[str, ...]] = (
    "push:",
    "pull_request:",
    "schedule:",
    "merge_group:",
)
RUNS_ON_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"runs-on:\s*\[(?P<labels>[^\]]+)\]",
    re.IGNORECASE,
)
REQUIRED_RUNNER_LABELS: Final[tuple[str, ...]] = ("self-hosted", "macOS", "ARM64", "Metal")
CONDITIONAL_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"if:\s*\$\{\{\s*github\.event_name\s*==\s*'workflow_dispatch'\s*\}\}"
)
PERMISSIONS_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?m)^permissions:\s*\n\s+contents:\s*read\s*$"
)

FailureList: TypeAlias = list[str]


@dataclass(frozen=True, slots=True)
class MetalWorkflowReport:
    """Static check result for the opt-in Metal workflow."""

    path: Path
    failures: tuple[str, ...]
    command: str
    metadata_terms: tuple[str, ...]

    def is_valid(self) -> bool:
        """Return whether the workflow has safe opt-in Metal semantics."""
        return not self.failures


def read_workflow(path: Path) -> str:
    """Read a workflow file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def require_text(text: str, needle: str, label: str, failures: FailureList) -> None:
    """Record a missing required literal."""
    if needle not in text:
        failures.append(f"missing {label}: {needle}")


def reject_default_events(text: str, failures: FailureList) -> None:
    """Reject events that would make Metal part of default CI."""
    for event in FORBIDDEN_DEFAULT_EVENTS:
        if re.search(rf"(?m)^\s+{re.escape(event)}\s*$", text):
            failures.append(f"Metal workflow must not run on default event: {event[:-1]}")


def check_opt_in_controls(text: str, failures: FailureList) -> None:
    """Require manual dispatch, self-hosted Apple-Silicon labels, and a job guard."""
    require_text(text, "workflow_dispatch:", "manual workflow_dispatch trigger", failures)
    runner_match = RUNS_ON_PATTERN.search(text)
    if runner_match is None:
        failures.append("Metal job must use self-hosted macOS/ARM64/Metal runner labels")
    else:
        labels = runner_match.group("labels")
        for label in REQUIRED_RUNNER_LABELS:
            if label not in labels:
                failures.append(f"Metal runner missing required label: {label}")
    if CONDITIONAL_PATTERN.search(text) is None:
        failures.append("Metal job must be guarded to workflow_dispatch")
    if PERMISSIONS_PATTERN.search(text) is None:
        failures.append("workflow permissions must be contents: read")


def check_required_work(text: str, failures: FailureList) -> None:
    """Require the shader command and platform metadata capture."""
    require_text(text, METAL_COMMAND, "Metal shader command", failures)
    for term in REQUIRED_METADATA:
        require_text(text, term, "metadata command", failures)


def check_default_ci(ci_path: Path, failures: FailureList) -> None:
    """Reject accidental inclusion of the Metal gate in portable default CI."""
    if not ci_path.exists():
        return
    ci_text = read_workflow(ci_path)
    if METAL_COMMAND in ci_text or "test-metal-shaders" in ci_text:
        failures.append(f"default CI must not require Metal shader gate: {ci_path}")


def check_metal_workflow(path: Path) -> MetalWorkflowReport:
    """Run all static checks for the opt-in Metal workflow."""
    failures: FailureList = []
    try:
        text = read_workflow(path)
    except FileNotFoundError:
        return MetalWorkflowReport(path, (f"workflow missing: {path}",), METAL_COMMAND, REQUIRED_METADATA)

    check_opt_in_controls(text, failures)
    reject_default_events(text, failures)
    check_required_work(text, failures)
    check_default_ci(DEFAULT_CI, failures)
    return MetalWorkflowReport(path, tuple(failures), METAL_COMMAND, REQUIRED_METADATA)


def selected_path(args: tuple[str, ...]) -> Path:
    """Choose the workflow path from CLI args."""
    match args:
        case ():
            return DEFAULT_WORKFLOW
        case ("--help",) | ("-h",):
            print("usage: check_metal_workflow.py [.github/workflows/metal.yml]")
            raise SystemExit(0)
        case (raw_path,):
            return Path(raw_path)
        case _:
            print("usage: check_metal_workflow.py [.github/workflows/metal.yml]", file=sys.stderr)
            raise SystemExit(2)


def render_report(report: MetalWorkflowReport) -> int:
    """Print a checker result and return the process status."""
    if report.is_valid():
        print(f"Metal workflow OK: {report.path}")
        print(f"command: {report.command}")
        print(f"metadata commands: {len(report.metadata_terms)}")
        print("fallback: manual self-hosted workflow is outside default CI")
        return 0

    print(f"Metal workflow FAILED: {report.path}", file=sys.stderr)
    for failure in report.failures:
        print(f"- {failure}", file=sys.stderr)
    return 1


def main() -> int:
    """CLI entrypoint for the Metal workflow checker."""
    return render_report(check_metal_workflow(selected_path(tuple(sys.argv[1:]))))


if __name__ == "__main__":
    raise SystemExit(main())
