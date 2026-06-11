#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/release/check_research_scope.py [--workflow-root PATH]
# 3. Or make executable and run:
#      chmod +x scripts/release/check_research_scope.py && ./scripts/release/check_research_scope.py [--workflow-root PATH]
# ──────────────────

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DOC_PATH: Final[Path] = REPO_ROOT / "docs" / "research-scope.md"
RESEARCH_ROOT: Final[Path] = REPO_ROOT / "research"
DEFAULT_WORKFLOW_ROOT: Final[Path] = REPO_ROOT / ".github" / "workflows"
REQUIRED_POLICY_PHRASES: Final[tuple[str, ...]] = (
    "optional diagnostics",
    "parity references",
    "not default product release gates",
    "must not be wired into default workflows",
)
WORKFLOW_EXTENSIONS: Final[frozenset[str]] = frozenset((".yml", ".yaml"))
RESEARCH_RUN_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w/.-])(?:\./)?research\s*/\s*[^\n#]*\.py\b"
)


@dataclass(frozen=True, slots=True)
class CheckFailure:
    """A release-scope policy violation."""

    path: Path
    message: str


def repo_relative(path: Path) -> str:
    """Format a path relative to the repo when possible."""
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def parse_workflow_root(args: tuple[str, ...]) -> Path:
    """Parse the small checker CLI without adding a dependency."""
    match args:
        case ():
            return DEFAULT_WORKFLOW_ROOT
        case ("--workflow-root", value):
            return Path(value).resolve()
        case ("--help",) | ("-h",):
            print("usage: check_research_scope.py [--workflow-root PATH]")
            raise SystemExit(0)
        case _:
            print("usage: check_research_scope.py [--workflow-root PATH]", file=sys.stderr)
            raise SystemExit(2)


def read_text(path: Path) -> str:
    """Read UTF-8 source text for policy and workflow checks."""
    return path.read_text(encoding="utf-8")


def research_scripts() -> tuple[Path, ...]:
    """Return the checked research script inventory."""
    return tuple(sorted(RESEARCH_ROOT.glob("*.py")))


def check_policy_text() -> tuple[CheckFailure, ...]:
    """Verify the docs classify research scripts and map current drift risks."""
    if not DOC_PATH.exists():
        return (CheckFailure(DOC_PATH, "missing research-scope policy document"),)

    text = read_text(DOC_PATH)
    lower_text = text.lower()
    failures: list[CheckFailure] = []

    for phrase in REQUIRED_POLICY_PHRASES:
        if phrase not in lower_text:
            failures.append(CheckFailure(DOC_PATH, f"missing policy phrase: {phrase}"))

    for script in research_scripts():
        script_ref = f"`research/{script.name}`"
        if script_ref not in text:
            failures.append(CheckFailure(DOC_PATH, f"missing drift-control row for {script_ref}"))

    return tuple(failures)


def workflow_files(workflow_root: Path) -> tuple[Path, ...]:
    """Return workflow files from a possibly absent workflow directory."""
    if not workflow_root.exists():
        return ()
    if workflow_root.is_file():
        return (workflow_root,)
    return tuple(
        sorted(
            path
            for path in workflow_root.rglob("*")
            if path.is_file() and path.suffix in WORKFLOW_EXTENSIONS
        )
    )


def check_default_workflows(workflow_root: Path) -> tuple[CheckFailure, ...]:
    """Fail when a default workflow invokes research/*.py."""
    failures: list[CheckFailure] = []
    for workflow in workflow_files(workflow_root):
        for line_number, line in enumerate(read_text(workflow).splitlines(), start=1):
            if RESEARCH_RUN_PATTERN.search(line):
                failures.append(
                    CheckFailure(
                        workflow,
                        f"line {line_number} runs research/*.py in a default workflow",
                    )
                )
    return tuple(failures)


def print_result(failures: tuple[CheckFailure, ...], workflow_root: Path) -> int:
    """Print a stable QA summary."""
    if not failures:
        scripts = ", ".join(f"research/{path.name}" for path in research_scripts())
        print("research scope check passed")
        print(f"policy: {repo_relative(DOC_PATH)}")
        print(f"workflow_root: {repo_relative(workflow_root)}")
        print(f"classified_scripts: {scripts}")
        return 0

    print("research scope check failed", file=sys.stderr)
    for failure in failures:
        print(f"- {repo_relative(failure.path)}: {failure.message}", file=sys.stderr)
    return 1


def main() -> int:
    """Run release-scope checks for optional research diagnostics."""
    workflow_root = parse_workflow_root(tuple(sys.argv[1:]))
    failures = check_policy_text() + check_default_workflows(workflow_root)
    return print_result(failures, workflow_root)


if __name__ == "__main__":
    raise SystemExit(main())
