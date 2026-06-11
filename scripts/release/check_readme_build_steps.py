#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# ///
# ─── How to run ───
# python3 scripts/release/check_readme_build_steps.py
# python3 scripts/release/check_readme_build_steps.py --readme /tmp/README.md
"""Verify README command sections document all public Zig build steps."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

ROOT: Final = Path(__file__).resolve().parents[2]
STEP_PATTERN: Final = re.compile(r'b\.step\("([A-Za-z0-9_-]+)"')
README_STEP_PATTERN: Final = re.compile(r"^\s*zig\s+build\s+([A-Za-z0-9_-]+)\b", re.MULTILINE)
README_PACKAGE_HEADINGS: Final = (
    ("engine", "## Engine command API"),
    ("labrat", "## Labrat command API"),
)


@dataclass(frozen=True, slots=True, order=True)
class BuildStep:
    package: str
    name: str


@dataclass(frozen=True, slots=True)
class CheckerConfig:
    readme_path: Path
    engine_build_path: Path
    labrat_build_path: Path


class CheckerError(Exception):
    """Base error for README/build-step sync failures."""


@dataclass(frozen=True, slots=True)
class CliUsageError(CheckerError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(frozen=True, slots=True)
class MissingReadmeSectionError(CheckerError):
    heading: str

    def __str__(self) -> str:
        return f"README is missing required section: {self.heading}"


# Internal step allowlist. Keep this intentionally explicit so adding a new
# b.step(...) is public by default and fails the checker until documented.
# Current rationale: no engine/ or labrat/ b.step(...) entries are internal-only.
INTERNAL_STEP_ALLOWLIST: Final[frozenset[BuildStep]] = frozenset()


def parse_args(argv: Sequence[str]) -> CheckerConfig:
    readme_path = ROOT / "README.md"
    engine_build_path = ROOT / "engine" / "build.zig"
    labrat_build_path = ROOT / "labrat" / "build.zig"
    index = 1

    while index < len(argv):
        option = argv[index]
        match option:
            case "--readme":
                readme_path = read_path_value(argv, index, option)
                index += 2
            case "--engine-build":
                engine_build_path = read_path_value(argv, index, option)
                index += 2
            case "--labrat-build":
                labrat_build_path = read_path_value(argv, index, option)
                index += 2
            case "--help" | "-h":
                raise CliUsageError(usage())
            case _:
                raise CliUsageError(f"unknown option: {option}\n{usage()}")

    return CheckerConfig(
        readme_path=readme_path,
        engine_build_path=engine_build_path,
        labrat_build_path=labrat_build_path,
    )


def read_path_value(argv: Sequence[str], index: int, option: str) -> Path:
    value_index = index + 1
    if value_index >= len(argv):
        raise CliUsageError(f"{option} requires a path value\n{usage()}")
    return Path(argv[value_index])


def usage() -> str:
    return (
        "usage: check_readme_build_steps.py [--readme PATH] "
        "[--engine-build PATH] [--labrat-build PATH]"
    )


def parse_build_steps(package: str, build_path: Path) -> frozenset[BuildStep]:
    source = build_path.read_text(encoding="utf-8")
    return frozenset(BuildStep(package=package, name=name) for name in STEP_PATTERN.findall(source))


def read_required_section(readme_text: str, heading: str) -> str:
    start = readme_text.find(heading)
    if start < 0:
        raise MissingReadmeSectionError(heading=heading)

    next_heading = readme_text.find("\n## ", start + len(heading))
    if next_heading < 0:
        return readme_text[start:]
    return readme_text[start:next_heading]


def parse_documented_steps(readme_path: Path) -> frozenset[BuildStep]:
    readme_text = readme_path.read_text(encoding="utf-8")
    documented_steps: set[BuildStep] = set()

    for package, heading in README_PACKAGE_HEADINGS:
        section = read_required_section(readme_text, heading)
        documented_steps.update(
            BuildStep(package=package, name=name) for name in README_STEP_PATTERN.findall(section)
        )

    return frozenset(documented_steps)


def public_steps(config: CheckerConfig) -> frozenset[BuildStep]:
    parsed_steps = (
        parse_build_steps("engine", config.engine_build_path)
        | parse_build_steps("labrat", config.labrat_build_path)
    )
    return frozenset(step for step in parsed_steps if step not in INTERNAL_STEP_ALLOWLIST)


def format_steps(steps: frozenset[BuildStep]) -> str:
    return "\n".join(f"  - {step.package}: zig build {step.name}" for step in sorted(steps))


def run(config: CheckerConfig) -> int:
    expected_steps = public_steps(config)
    documented_steps = parse_documented_steps(config.readme_path)
    missing_steps = expected_steps - documented_steps

    if missing_steps:
        print("README is missing public Zig build steps:")
        print(format_steps(missing_steps))
        return 1

    print(f"README build-step sync OK: {len(expected_steps)} public steps documented.")
    return 0


def main(argv: Sequence[str]) -> int:
    try:
        config = parse_args(argv)
        return run(config)
    except CliUsageError as error:
        print(error, file=sys.stderr)
        return 2
    except MissingReadmeSectionError as error:
        print(error, file=sys.stderr)
        return 1
    except OSError as error:
        print(f"file error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
