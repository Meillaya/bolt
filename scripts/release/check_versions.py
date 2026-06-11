#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run check_versions.py
#    Or from the repository root:
#      python3 scripts/release/check_versions.py
# 3. Or make executable and run:
#      chmod +x check_versions.py && ./check_versions.py
# ──────────────────

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final
import os
import re
import sys

EXPECTED_MINIMUM_ZIG_VERSION: Final = "0.16.0"
ROOT_VERSION_ENV: Final = "BOLT_VERSION_ROOT"
ENGINE_ZON_ENV: Final = "BOLT_ENGINE_ZON"
LABRAT_ZON_ENV: Final = "BOLT_LABRAT_ZON"
VERSION_PATTERN: Final = re.compile(r'\.version\s*=\s*"([^"]+)"')
MINIMUM_ZIG_PATTERN: Final = re.compile(r'\.minimum_zig_version\s*=\s*"([^"]+)"')


@dataclass(frozen=True, slots=True)
class PackageMetadata:
    """Release metadata parsed from one Zig package manifest."""

    path: Path
    version: str
    minimum_zig_version: str


@dataclass(frozen=True, slots=True)
class VersionPaths:
    """Repository files that define the product and package release versions."""

    root_version: Path
    engine_zon: Path
    labrat_zon: Path


@dataclass(frozen=True, slots=True)
class VersionCheckError(Exception):
    """A version authority check failed for a specific file."""

    path: Path
    detail: str

    def __str__(self) -> str:
        return f"{self.path}: {self.detail}"


def repo_root() -> Path:
    """Return the repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def override_path(env_name: str, default_path: Path) -> Path:
    """Return a fixture override path when an environment variable is set."""
    override = os.environ.get(env_name)
    if override is None or override == "":
        return default_path
    return Path(override)


def version_paths(root: Path) -> VersionPaths:
    """Resolve version authority paths, allowing env-based fixture overrides."""
    return VersionPaths(
        root_version=override_path(ROOT_VERSION_ENV, root / "VERSION"),
        engine_zon=override_path(ENGINE_ZON_ENV, root / "engine" / "build.zig.zon"),
        labrat_zon=override_path(LABRAT_ZON_ENV, root / "labrat" / "build.zig.zon"),
    )


def read_text(path: Path) -> str:
    """Read a UTF-8 source file or raise a typed check error."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise VersionCheckError(path=path, detail="file is missing") from error


def required_match(path: Path, pattern: re.Pattern[str], field_name: str) -> str:
    """Extract a required quoted field from a build.zig.zon manifest."""
    match = pattern.search(read_text(path))
    if match is None:
        raise VersionCheckError(path=path, detail=f"missing {field_name}")
    return match.group(1)


def read_root_version(path: Path) -> str:
    """Read the root product release version."""
    version = read_text(path).strip()
    if version == "":
        raise VersionCheckError(path=path, detail="VERSION is empty")
    return version


def read_package_metadata(path: Path) -> PackageMetadata:
    """Parse version and Zig floor metadata from one package manifest."""
    return PackageMetadata(
        path=path,
        version=required_match(path, VERSION_PATTERN, ".version"),
        minimum_zig_version=required_match(path, MINIMUM_ZIG_PATTERN, ".minimum_zig_version"),
    )


def require_equal(path: Path, field_name: str, actual: str, expected: str) -> None:
    """Raise when a package field does not match the release policy."""
    if actual != expected:
        detail = f"{field_name} is {actual!r}; expected {expected!r}"
        raise VersionCheckError(path=path, detail=detail)


def check_versions(paths: VersionPaths) -> tuple[str, PackageMetadata, PackageMetadata]:
    """Validate root, engine, and labrat version authority consistency."""
    root_version = read_root_version(paths.root_version)
    engine = read_package_metadata(paths.engine_zon)
    labrat = read_package_metadata(paths.labrat_zon)

    require_equal(engine.path, ".version", engine.version, root_version)
    require_equal(labrat.path, ".version", labrat.version, root_version)
    require_equal(
        engine.path,
        ".minimum_zig_version",
        engine.minimum_zig_version,
        EXPECTED_MINIMUM_ZIG_VERSION,
    )
    require_equal(
        labrat.path,
        ".minimum_zig_version",
        labrat.minimum_zig_version,
        EXPECTED_MINIMUM_ZIG_VERSION,
    )
    return root_version, engine, labrat


def print_success(root_version: str, engine: PackageMetadata, labrat: PackageMetadata) -> None:
    """Print the validated version authority summary."""
    print(f"root VERSION: {root_version}")
    print(f"engine {engine.path}: version {engine.version}, minimum Zig {engine.minimum_zig_version}")
    print(f"labrat {labrat.path}: version {labrat.version}, minimum Zig {labrat.minimum_zig_version}")


def main() -> int:
    """Run the version authority check."""
    paths = version_paths(repo_root())
    try:
        root_version, engine, labrat = check_versions(paths)
    except VersionCheckError as error:
        print(f"version check failed: {error}", file=sys.stderr)
        return 1
    print_success(root_version, engine, labrat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
