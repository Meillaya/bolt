#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run scripts/release/check_support_matrix.py [docs/support-matrix.md]
# 3. Or with system Python from the repository root:
#      python3 scripts/release/check_support_matrix.py [docs/support-matrix.md]
# ──────────────────

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Final

DEFAULT_DOC: Final[Path] = Path("docs/support-matrix.md")

REQUIRED_TERMS: Final[tuple[str, ...]] = (
    "Apple Silicon",
    "Metal",
    "self-hosted",
    "manual",
    "real assets",
    "provider credentials",
    "non-goals",
    "SaaS",
    "GUI",
    "App Store",
    "broad cross-platform",
    "hosted provider integrations",
)

DEFAULT_COMMANDS: Final[tuple[str, ...]] = (
    "cd engine && zig build test --summary all",
    "cd engine && zig build --summary all",
    "cd labrat && zig build test --summary all",
    "cd labrat && zig build api-offline-test --summary all",
)

OPT_IN_COMMANDS: Final[tuple[str, ...]] = (
    "cd engine && zig build test-metal-shaders --summary all",
    "cd engine && zig build run-metal-mlp --summary all",
    "cd engine && zig build validate-assets --summary all",
    "cd engine && zig build validate-tokenizer --summary all",
    "cd engine && zig build run-bonsai --summary all",
    "cd engine && zig build run-bonsai-golden --summary all",
    "cd engine && zig build run-bonsai-bench --summary all",
    "cd engine && zig build run-bonsai-q4-golden --summary all",
    "cd engine && zig build run-bonsai-q4-bench --summary all",
    "cd labrat && zig build mnist-agent --summary all",
    "cd labrat && zig build bonsai-agent --summary all",
    "cd labrat && zig build bonsai-q4-agent --summary all",
    "cd labrat && zig build mnist-researcher --summary all",
    "cd labrat && zig build bonsai-researcher --summary all",
    "cd labrat && zig build bonsai-q4-researcher --summary all",
)


@dataclass(frozen=True, slots=True)
class SupportMatrixReport:
    """Result of checking the support matrix document."""

    path: Path
    missing_terms: tuple[str, ...]
    missing_default_commands: tuple[str, ...]
    missing_opt_in_commands: tuple[str, ...]

    def is_valid(self) -> bool:
        """Return whether every required support-matrix item is present."""
        return not (
            self.missing_terms
            or self.missing_default_commands
            or self.missing_opt_in_commands
        )


def missing_items(haystack: str, needles: tuple[str, ...]) -> tuple[str, ...]:
    """Return required strings that do not appear in the document."""
    return tuple(needle for needle in needles if needle not in haystack)


def check_support_matrix(path: Path) -> SupportMatrixReport:
    """Parse the support matrix markdown and report missing scope items."""
    document = path.read_text(encoding="utf-8")
    return SupportMatrixReport(
        path=path,
        missing_terms=missing_items(document, REQUIRED_TERMS),
        missing_default_commands=missing_items(document, DEFAULT_COMMANDS),
        missing_opt_in_commands=missing_items(document, OPT_IN_COMMANDS),
    )


def print_missing(label: str, items: tuple[str, ...]) -> None:
    """Print one missing-item section if needed."""
    if not items:
        return
    print(f"missing {label}:")
    for item in items:
        print(f"  - {item}")


def render_report(report: SupportMatrixReport) -> int:
    """Print a human-readable checker result and return a process code."""
    if report.is_valid():
        print(f"support matrix OK: {report.path}")
        print(f"required terms: {len(REQUIRED_TERMS)}")
        print(f"default commands: {len(DEFAULT_COMMANDS)}")
        print(f"opt-in commands: {len(OPT_IN_COMMANDS)}")
        return 0

    print(f"support matrix FAILED: {report.path}")
    print_missing("terms", report.missing_terms)
    print_missing("default commands", report.missing_default_commands)
    print_missing("opt-in commands", report.missing_opt_in_commands)
    return 1


def selected_path(args: tuple[str, ...]) -> Path:
    """Choose the markdown path from CLI args."""
    match args:
        case ():
            return DEFAULT_DOC
        case (raw_path,):
            return Path(raw_path)
        case _:
            raise SystemExit("usage: check_support_matrix.py [docs/support-matrix.md]")


def main() -> int:
    """CLI entrypoint for the release support-matrix checker."""
    path = selected_path(tuple(sys.argv[1:]))
    report = check_support_matrix(path)
    return render_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
