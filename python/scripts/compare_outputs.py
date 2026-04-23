#!/usr/bin/env python3
"""Minimal fixture comparison utility for Phase 1 scaffolding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> object:
    return json.loads(path.read_text())


def format_path(path: str) -> str:
    return path or "$"


def collect_differences(left: Any, right: Any, path: str = "") -> list[str]:
    differences: list[str] = []

    if type(left) is not type(right):
        differences.append(
            f"{format_path(path)}: type mismatch ({type(left).__name__} != {type(right).__name__})"
        )
        return differences

    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        for key in sorted(left_keys - right_keys):
            differences.append(f"{format_path(path)}.{key}: missing on right")
        for key in sorted(right_keys - left_keys):
            differences.append(f"{format_path(path)}.{key}: missing on left")
        for key in sorted(left_keys & right_keys):
            differences.extend(collect_differences(left[key], right[key], f"{format_path(path)}.{key}"))
        return differences

    if isinstance(left, list):
        if len(left) != len(right):
            differences.append(f"{format_path(path)}: length mismatch ({len(left)} != {len(right)})")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            differences.extend(collect_differences(left_item, right_item, f"{format_path(path)}[{index}]"))
        return differences

    if left != right:
        differences.append(f"{format_path(path)}: {left!r} != {right!r}")

    return differences


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left = load_json(args.left)
    right = load_json(args.right)

    if left != right:
        differences = collect_differences(left, right)
        print(f"mismatch: {args.left} != {args.right}")
        for line in differences[:20]:
            print(f"  - {line}")
        if len(differences) > 20:
            print(f"  - ... {len(differences) - 20} more difference(s)")
        return 1

    print(f"match: {args.left} == {args.right}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
