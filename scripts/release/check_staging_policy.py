#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/release/check_staging_policy.py --forbid .omo/drafts .omo/evidence

"""Reject forbidden run-local paths from the Git index or intent-to-add set."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
import subprocess
import sys


@dataclass(frozen=True, slots=True)
class ChangedPath:
    status: str
    path: str


@dataclass(frozen=True, slots=True)
class Args:
    forbid: tuple[str, ...]


def normalize_prefix(raw_prefix: str) -> str:
    prefix = raw_prefix.strip().replace("\\", "/").strip("/")
    if prefix == "":
        raise ValueError("forbidden prefix cannot be empty")
    return prefix


def is_under_prefix(path: str, prefix: str) -> bool:
    normalized_path = str(PurePosixPath(path.replace("\\", "/")))
    return normalized_path == prefix or normalized_path.startswith(f"{prefix}/")


def git_status_records() -> tuple[ChangedPath, ...]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"git status failed: {stderr.strip()}")

    parts = tuple(part.decode("utf-8", errors="surrogateescape") for part in result.stdout.split(b"\0") if part)
    records: list[ChangedPath] = []
    index = 0
    while index < len(parts):
        entry = parts[index]
        if len(entry) < 4:
            raise RuntimeError(f"unexpected porcelain record: {entry!r}")
        status = entry[:2]
        path = entry[3:]
        if status[0] in ("R", "C"):
            index += 1
            if index >= len(parts):
                raise RuntimeError(f"missing rename target for: {path}")
            path = parts[index]
        records.append(ChangedPath(status=status, path=path))
        index += 1
    return tuple(records)


def is_index_or_intent_to_add(record: ChangedPath) -> bool:
    if record.status == "??":
        return False
    return record.status[0] != " " or record.status[1] != " "


def forbidden_records(records: tuple[ChangedPath, ...], prefixes: tuple[str, ...]) -> tuple[ChangedPath, ...]:
    rejected: list[ChangedPath] = []
    for record in records:
        if not is_index_or_intent_to_add(record):
            continue
        if any(is_under_prefix(record.path, prefix) for prefix in prefixes):
            rejected.append(record)
    return tuple(rejected)


def parse_args(argv: list[str]) -> Args:
    if len(argv) < 2 or argv[0] != "--forbid":
        raise SystemExit("usage: check_staging_policy.py --forbid <prefix> [<prefix> ...]")
    return Args(forbid=tuple(argv[1:]))


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        prefixes = tuple(normalize_prefix(prefix) for prefix in args.forbid)
        rejected = forbidden_records(git_status_records(), prefixes)
    except (RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if rejected:
        print("forbidden staged or intent-to-add paths found:")
        for record in rejected:
            print(f"{record.status} {record.path}")
        print("forbidden prefixes:", ", ".join(prefixes))
        return 1

    print("staging policy: ok")
    print("forbidden prefixes:", ", ".join(prefixes))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
