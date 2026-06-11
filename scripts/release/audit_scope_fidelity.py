#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/release/audit_scope_fidelity.py

"""Audit v1 production scope boundaries from the approved plan."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    evidence: str


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(needle.lower() in lowered for needle in needles)


def workflow_texts() -> tuple[tuple[str, str], ...]:
    root = Path(".github/workflows")
    if not root.exists():
        return ()
    return tuple((str(path), path.read_text(encoding="utf-8")) for path in sorted(root.glob("*.yml")))


def default_ci_text() -> str:
    path = Path(".github/workflows/ci.yml")
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def checks() -> tuple[Check, ...]:
    readme = read("README.md")
    security = read("SECURITY.md")
    release = read("RELEASE.md")
    research_scope = read("docs/research-scope.md")
    macos = read("docs/macos-distribution.md")
    ci = default_ci_text()
    workflows = workflow_texts()
    all_workflows = "\n".join(text for _, text in workflows)
    return (
        Check(
            "provider live mode disabled/out of v1",
            contains_any(readme + security + release, ("live provider mode is out of v1 production scope",)),
            "README/SECURITY/RELEASE contain the v1 live-provider scope lock",
        ),
        Check(
            "default CI excludes provider credentials",
            "LABRAT_LIVE=1" not in ci and "ANTHROPIC_API_KEY" not in ci and "secrets." not in ci,
            ".github/workflows/ci.yml has no live-provider enablement or secrets reference",
        ),
        Check(
            "research remains optional diagnostics",
            contains_any(research_scope + readme, ("optional diagnostics", "not a default release gate")),
            "research docs classify research/*.py outside default release gates",
        ),
        Check(
            "default CI excludes first-class research expansion",
            "research/" not in ci and "*.py" not in ci,
            ".github/workflows/ci.yml does not run research scripts",
        ),
        Check(
            "GUI/SaaS/App Store excluded",
            contains_any(macos + release + readme, ("not app store", "not an app store", "deferred after v1")),
            "distribution docs keep direct macOS distribution deferred and not App Store",
        ),
        Check(
            "no default real-asset or Metal blockers",
            "run-bonsai" not in ci and "test-metal-shaders" not in ci and "self-hosted" not in ci,
            "default CI avoids real-asset, shader, and self-hosted requirements",
        ),
        Check(
            "optional workflows are explicitly opt-in/manual",
            "workflow_dispatch" in all_workflows
            and "self-hosted" in all_workflows
            and "Opt-in real asset gate" in all_workflows,
            "Metal/real-asset workflows are separate opt-in self-hosted surfaces",
        ),
    )


def main() -> int:
    failures: list[str] = []
    print("# Final scope fidelity audit")
    for check in checks():
        if check.ok:
            print(f"PASS: {check.name} — {check.evidence}")
        else:
            print(f"ERROR: {check.name} — {check.evidence}")
            failures.append(check.name)
    if failures:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
