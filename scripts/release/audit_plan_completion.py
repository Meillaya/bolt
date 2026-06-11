#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
# ─── How to run ───
# python3 scripts/release/audit_plan_completion.py --plan .omo/plans/production-grade-bolt.md --evidence .omo/evidence

"""Audit that the approved production plan has local evidence for every task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys

TASK_RE = re.compile(r"^- \[[ xX]\] T([1-9][0-9]*)\.", re.MULTILINE)
REQUIRED_SUMMARY_SECTIONS = (
    "Default gates",
    "Artifact validation",
    "Security/docs checks",
    "Optional Metal",
    "Optional real assets",
    "Skipped with reason",
    "Dirty worktree",
)
DEFAULT_SECTIONS = ("Default gates", "Artifact validation", "Security/docs checks")


@dataclass(frozen=True, slots=True)
class AuditResult:
    errors: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Args:
    plan: Path
    evidence: Path


def section_text(markdown: str, heading: str) -> str:
    marker = f"## {heading}"
    start = markdown.find(marker)
    if start == -1:
        return ""
    next_start = markdown.find("\n## ", start + len(marker))
    if next_start == -1:
        return markdown[start:]
    return markdown[start:next_start]


def task_ids(plan_text: str) -> tuple[int, ...]:
    return tuple(sorted(int(match.group(1)) for match in TASK_RE.finditer(plan_text)))


def evidence_for_task(evidence_dir: Path, task_id: int) -> tuple[Path, ...]:
    return tuple(sorted(evidence_dir.glob(f"task-{task_id}-*")))


def audit_summary(summary_path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    if not summary_path.exists():
        return [f"missing release summary: {summary_path}"], notes
    text = summary_path.read_text(encoding="utf-8")
    for heading in REQUIRED_SUMMARY_SECTIONS:
        if f"## {heading}" not in text:
            errors.append(f"missing release summary section: {heading}")
        else:
            notes.append(f"summary section present: {heading}")
    for heading in DEFAULT_SECTIONS:
        body = section_text(text, heading)
        if "SKIPPED" in body:
            errors.append(f"default section contains SKIPPED row: {heading}")
    if "- final_exit_code: 0" not in text:
        errors.append("release summary final_exit_code is not 0")
    return errors, notes


def audit(plan_path: Path, evidence_dir: Path) -> AuditResult:
    errors: list[str] = []
    notes: list[str] = []
    if not plan_path.exists():
        return AuditResult(errors=(f"missing plan: {plan_path}",), notes=())
    if not evidence_dir.is_dir():
        return AuditResult(errors=(f"missing evidence dir: {evidence_dir}",), notes=())

    ids = task_ids(plan_path.read_text(encoding="utf-8"))
    if ids != tuple(range(1, 25)):
        errors.append(f"expected tasks T1-T24, found: {ids}")
    for task_id in range(1, 25):
        paths = evidence_for_task(evidence_dir, task_id)
        if not paths:
            errors.append(f"missing evidence for T{task_id}")
        else:
            notes.append(f"T{task_id}: {len(paths)} evidence file(s)")

    summary_errors, summary_notes = audit_summary(evidence_dir / "release-candidate-summary.md")
    errors.extend(summary_errors)
    notes.extend(summary_notes)

    final_required = (
        "final-plan-compliance.md",
        "final-code-quality.md",
        "final-manual-qa.md",
        "final-scope-fidelity.md",
    )
    for name in final_required:
        path = evidence_dir / name
        if path.exists():
            notes.append(f"final evidence present: {path}")
        else:
            notes.append(f"final evidence pending during audit: {path}")
    return AuditResult(errors=tuple(errors), notes=tuple(notes))


def parse_args(argv: list[str]) -> Args:
    plan: Path | None = None
    evidence: Path | None = None
    index = 0
    while index < len(argv):
        option = argv[index]
        if option == "--plan" and index + 1 < len(argv):
            plan = Path(argv[index + 1])
            index += 2
        elif option == "--evidence" and index + 1 < len(argv):
            evidence = Path(argv[index + 1])
            index += 2
        else:
            raise SystemExit("usage: audit_plan_completion.py --plan <path> --evidence <dir>")
    if plan is None or evidence is None:
        raise SystemExit("usage: audit_plan_completion.py --plan <path> --evidence <dir>")
    return Args(plan=plan, evidence=evidence)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    result = audit(args.plan, args.evidence)
    print("# Final plan compliance audit")
    for note in result.notes:
        print(f"PASS: {note}")
    for error in result.errors:
        print(f"ERROR: {error}")
    if result.errors:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
