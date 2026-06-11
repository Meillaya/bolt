#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/release/validate_artifact.py --all-fixtures
# python3 scripts/release/validate_artifact.py docs/artifacts/fixtures/pass/*.json
# python3 scripts/release/validate_artifact.py docs/artifacts/fixtures/fail/nested-pass-top-fail.json
# ──────────────────

"""Validate Bolt release artifact JSON files against the schema registry."""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path
from typing import Final, TypeAlias, cast

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | Mapping[str, "JsonValue"]
Scalar: TypeAlias = bool | int | str

SCHEMA_VERSION: Final = "1"
DEFAULT_REGISTRY: Final = Path("docs/artifacts/registry.md")
PASS_FIXTURES: Final = Path("docs/artifacts/fixtures/pass")
FAIL_FIXTURES: Final = Path("docs/artifacts/fixtures/fail")
TIMESTAMP_RE: Final = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
GIT_COMMIT_RE: Final = re.compile(r"^(?:[0-9a-fA-F]{7,40}(?:-dirty)?|dirty)$")


@dataclass(frozen=True, slots=True)
class PassCondition:
    """One explicit top-level artifact pass condition."""

    path: tuple[str, ...]
    expected: Scalar


@dataclass(frozen=True, slots=True)
class ArtifactValidationError(Exception):
    """Validation failed for one artifact path."""

    path: Path
    detail: str

    def message(self) -> str:
        """Render a stable human-readable validation error."""
        return f"{self.path}: {self.detail}"


REAL_ASSET_TYPES: Final[frozenset[str]] = frozenset((
    "engine.bonsai.smoke",
    "engine.bonsai.readiness",
    "engine.bonsai.bench",
    "engine.q4.golden",
    "engine.q4.bench",
))
MANIFEST_DIGEST_RE: Final = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


CONDITIONS: Final[Mapping[str, PassCondition]] = {
    "engine.mnist.run": PassCondition(("passes_reference_relative_threshold",), True),
    "engine.mnist.run_1bit": PassCondition(("selected_label_parity",), True),
    "engine.mnist.run_infer": PassCondition(("inference", "passed"), True),
    "engine.metal.mlp_runtime": PassCondition(("runtime", "passed"), True),
    "engine.bonsai.smoke": PassCondition(("smoke", "passed"), True),
    "engine.bonsai.readiness": PassCondition(("readiness", "passed"), True),
    "engine.bonsai.bench": PassCondition(("benchmark", "passed"), True),
    "engine.q4.golden": PassCondition(("golden", "tokens_match"), True),
    "engine.q4.bench": PassCondition(("benchmark", "passed"), True),
    "labrat.agent": PassCondition(("agent", "passed"), True),
    "labrat.researcher": PassCondition(("researcher", "passed"), True),
    "labrat.audit": PassCondition(("audit", "findings_block_release"), False),
    "labrat.blockers": PassCondition(("blockers", "open_count"), 0),
    "labrat.summary": PassCondition(("summary", "passed"), True),
}


def require_mapping(path: Path, value: JsonValue) -> Mapping[str, JsonValue]:
    """Parse the root JSON value as a mapping."""
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(path, "root JSON value must be a mapping")
    return value


def require_string(path: Path, data: Mapping[str, JsonValue], field: str) -> str:
    """Read one required non-empty string field."""
    value = data.get(field)
    if not isinstance(value, str) or value == "":
        raise ArtifactValidationError(path, f"missing or invalid top-level `{field}` string")
    return value


def require_top_level(path: Path, data: Mapping[str, JsonValue]) -> str:
    """Validate the common artifact envelope and return the artifact type."""
    schema_version = require_string(path, data, "schema_version")
    status = require_string(path, data, "status")
    artifact_type = require_string(path, data, "artifact_type")
    _command = require_string(path, data, "command")
    _cwd = require_string(path, data, "cwd")
    git_commit = require_string(path, data, "git_commit")
    timestamp = require_string(path, data, "timestamp_utc")
    if schema_version != SCHEMA_VERSION:
        raise ArtifactValidationError(path, f"schema_version must be {SCHEMA_VERSION!r}")
    if status != "pass":
        raise ArtifactValidationError(path, f"top-level status must be 'pass', got {status!r}")
    if GIT_COMMIT_RE.fullmatch(git_commit) is None:
        raise ArtifactValidationError(path, "git_commit must be a 7- to 40-character hex SHA, optional -dirty suffix, or dirty marker")
    if TIMESTAMP_RE.fullmatch(timestamp) is None:
        raise ArtifactValidationError(path, "timestamp_utc must be an ISO-8601 UTC timestamp ending in Z")
    require_toolchain(path, data)
    if artifact_type not in CONDITIONS:
        raise ArtifactValidationError(path, f"unregistered artifact_type {artifact_type!r}")
    return artifact_type


def require_toolchain(path: Path, data: Mapping[str, JsonValue]) -> None:
    """Require non-empty toolchain metadata with string version values."""
    value = data.get("toolchain")
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(path, "missing or invalid top-level `toolchain` mapping")
    versions = tuple(item for item in value.values() if isinstance(item, str) and item != "")
    if len(versions) == 0:
        raise ArtifactValidationError(path, "toolchain must contain at least one non-empty string")


def read_path(data: Mapping[str, JsonValue], path_parts: tuple[str, ...]) -> JsonValue | None:
    """Read an explicit registry path without recursive truthy-field fallback."""
    current: JsonValue = data
    for part in path_parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def require_pass_condition(path: Path, data: Mapping[str, JsonValue], artifact_type: str) -> None:
    """Validate the registered per-artifact pass condition."""
    condition = CONDITIONS[artifact_type]
    actual = read_path(data, condition.path)
    if actual != condition.expected:
        joined = ".".join(condition.path)
        raise ArtifactValidationError(path, f"{artifact_type} requires `{joined}` == {condition.expected!r}")
    if artifact_type in REAL_ASSET_TYPES:
        require_manifest_digest(path, data)
    if artifact_type == "labrat.agent":
        require_labrat_agent(path, data)


def require_manifest_digest(path: Path, data: Mapping[str, JsonValue]) -> None:
    """Require a manifest digest for real-asset artifacts."""
    digest = require_string(path, data, "manifest_digest")
    if MANIFEST_DIGEST_RE.fullmatch(digest) is None:
        raise ArtifactValidationError(path, "manifest_digest must be sha256:<64 hex chars>")


def require_labrat_agent(path: Path, data: Mapping[str, JsonValue]) -> None:
    """Validate Labrat wrapper success semantics explicitly."""
    agent_value = data.get("agent")
    if not isinstance(agent_value, Mapping):
        raise ArtifactValidationError(path, "labrat.agent requires top-level `agent` mapping")
    if agent_value.get("exit_code") != 0:
        raise ArtifactValidationError(path, "labrat.agent requires `agent.exit_code` == 0")
    if agent_value.get("compiled") is not True:
        raise ArtifactValidationError(path, "labrat.agent requires `agent.compiled` == true")


def load_artifact(path: Path) -> Mapping[str, JsonValue]:
    """Load one artifact JSON mapping."""
    try:
        raw_value = cast(object, json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as error:
        raise ArtifactValidationError(path, "file is missing") from error
    except JSONDecodeError as error:
        raise ArtifactValidationError(path, f"invalid JSON: {error.msg}") from error
    return require_mapping(path, normalize_json(path, raw_value))


def normalize_json(path: Path, value: object) -> JsonValue:
    """Convert a decoded JSON object into the explicit recursive JsonValue type."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list):
        array_items = cast(list[object], value)
        return [normalize_json(path, item) for item in array_items]
    if isinstance(value, dict):
        object_items = cast(dict[object, object], value)
        normalized: dict[str, JsonValue] = {}
        for key, item in object_items.items():
            if not isinstance(key, str):
                raise ArtifactValidationError(path, "JSON object key must be a string")
            normalized[key] = normalize_json(path, item)
        return normalized
    raise ArtifactValidationError(path, "unsupported JSON value type")


def validate_artifact(path: Path) -> None:
    """Validate one release artifact file."""
    data = load_artifact(path)
    artifact_type = require_top_level(path, data)
    require_pass_condition(path, data, artifact_type)


def sorted_json_files(directory: Path) -> tuple[Path, ...]:
    """Return deterministic JSON fixtures from a directory."""
    return tuple(sorted(directory.glob("*.json")))


def require_registry(path: Path) -> None:
    """Check that the human-readable registry exists and names all artifact types."""
    text = path.read_text(encoding="utf-8")
    missing = tuple(name for name in CONDITIONS if name not in text)
    if missing:
        joined = ", ".join(missing)
        raise ArtifactValidationError(path, f"registry is missing artifact types: {joined}")


def validate_expected(path: Path, should_pass: bool) -> bool:
    """Validate one fixture with an expected outcome."""
    try:
        validate_artifact(path)
    except ArtifactValidationError as error:
        if should_pass:
            print(f"ERROR: {error.message()}", file=sys.stderr)
            return False
        print(f"rejected as expected: {error.message()}")
        return True
    if should_pass:
        print(f"accepted: {path}")
        return True
    print(f"ERROR: {path}: invalid fixture was accepted", file=sys.stderr)
    return False


@dataclass(frozen=True, slots=True)
class Args:
    """Typed CLI arguments for artifact validation."""

    artifacts: tuple[Path, ...]
    registry: Path
    fixtures: tuple[Path, ...]
    all_fixtures: bool


def fixture_paths(args: Args) -> tuple[Path, ...]:
    """Collect positional, explicit, and all-fixture paths."""
    paths = args.artifacts + args.fixtures
    if args.all_fixtures:
        return paths + sorted_json_files(PASS_FIXTURES) + sorted_json_files(FAIL_FIXTURES)
    return paths


def usage() -> str:
    """Return CLI usage text."""
    return "usage: validate_artifact.py [--registry PATH] [--fixture PATH ...] [--all-fixtures] [ARTIFACT ...]"


def parse_args(argv: list[str]) -> Args:
    """Parse CLI arguments without untyped argparse namespaces."""
    artifacts: list[Path] = []
    fixtures: list[Path] = []
    registry = DEFAULT_REGISTRY
    all_fixtures = False
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in ("-h", "--help"):
            print(usage())
            raise SystemExit(0)
        if arg == "--all-fixtures":
            all_fixtures = True
        elif arg == "--registry":
            index += 1
            if index >= len(argv):
                print(usage(), file=sys.stderr)
                raise SystemExit(2)
            registry = Path(argv[index])
        elif arg == "--fixture":
            index += 1
            if index >= len(argv):
                print(usage(), file=sys.stderr)
                raise SystemExit(2)
            while index < len(argv) and not argv[index].startswith("--"):
                fixtures.append(Path(argv[index]))
                index += 1
            index -= 1
        elif arg.startswith("--"):
            print(usage(), file=sys.stderr)
            raise SystemExit(2)
        else:
            artifacts.append(Path(arg))
        index += 1
    return Args(tuple(artifacts), registry, tuple(fixtures), all_fixtures)


def main(argv: list[str]) -> int:
    """Run artifact validation."""
    args = parse_args(argv)
    try:
        require_registry(args.registry)
    except (ArtifactValidationError, FileNotFoundError) as error:
        message = error.message() if isinstance(error, ArtifactValidationError) else str(error)
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    paths = fixture_paths(args)
    if len(paths) == 0:
        print("usage error: provide artifacts or --all-fixtures", file=sys.stderr)
        return 2
    ok = True
    for path in paths:
        should_pass = not (args.all_fixtures and FAIL_FIXTURES in path.parents)
        ok = validate_expected(path, should_pass) and ok
    if ok:
        print("artifact schema validation: ok")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
