#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import re
import shlex
import sys
from typing import Final

ENGINE_BUILD: Final[Path] = Path('engine/build.zig')
LABRAT_BUILD: Final[Path] = Path('labrat/build.zig')
SCRIPT_DIR: Final[Path] = Path('scripts/release')
REQUIRED_EVIDENCE: Final[tuple[str, ...]] = ('task-1-baseline.txt', 'task-15-metal-manual.txt', 'bolt-q4-bench.json', 'labrat-mnist-agent.json')
COMMAND_RE: Final[re.Pattern[str]] = re.compile(r'^(?:python3 scripts/release/[\w_]+\.py|bash scripts/release/[\w_]+\.sh|cd (engine|labrat) && zig build(?: ([\w-]+))?|cd labrat && unset ANTHROPIC_API_KEY && zig build ([\w-]+))')

def build_steps(path: Path) -> set[str]:
    text = path.read_text(encoding='utf-8')
    return set(re.findall(r'b\.step\("([^"]+)"', text)) | {'build'}

def zig_step(command: str) -> tuple[str, str] | None:
    """Return package and zig build step from a runbook command."""
    parts = shlex.split(command)
    if len(parts) < 5 or parts[0] != 'cd' or parts[2] != '&&' or parts[3] != 'zig' or parts[4] != 'build':
        if len(parts) >= 8 and parts[0] == 'cd' and parts[1] == 'labrat' and parts[2] == '&&':
            tail = parts[3:]
            if len(tail) >= 5 and tail[0] == 'unset' and tail[2] == '&&' and tail[3] == 'zig' and tail[4] == 'build':
                step = tail[5] if len(tail) > 5 and not tail[5].startswith('-') else 'build'
                return 'labrat', step
        return None
    step = parts[5] if len(parts) > 5 and not parts[5].startswith('-') else 'build'
    return parts[1], step


def validate_command(command: str, engine_steps: set[str], labrat_steps: set[str]) -> str | None:
    if command.startswith('python3 scripts/release/'):
        script = command.split()[1]
        return None if Path(script).exists() else f'missing release script: {script}'
    if command.startswith('bash scripts/release/'):
        script = command.split()[1]
        return None if Path(script).exists() else f'missing release script: {script}'
    parsed = zig_step(command)
    if parsed is None:
        return None
    package, step = parsed
    allowed = engine_steps if package == 'engine' else labrat_steps
    if step not in allowed:
        return f'unknown {package} build step: {step}'
    return None

def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) == 2 else Path('docs/production-runbook.md')
    text = path.read_text(encoding='utf-8')
    engine_steps = build_steps(ENGINE_BUILD)
    labrat_steps = build_steps(LABRAT_BUILD)
    failures: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(('python3 ', 'bash ', 'cd ')):
            failure = validate_command(stripped, engine_steps, labrat_steps)
            if failure is not None:
                failures.append(failure)
    for item in REQUIRED_EVIDENCE:
        if item not in text:
            failures.append(f'missing evidence/artifact path: {item}')
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print(f'runbook OK: {path}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
