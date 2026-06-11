#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
from typing import Final

DOCS: Final[tuple[Path, ...]] = (Path('RELEASE.md'), Path('docs/release-provenance.md'))
REQUIRED: Final[tuple[str, ...]] = ('SLSA', 'SPDX', 'attestation', 'entry criteria', 'not default CI blockers')
COMPLETE_PATTERNS: Final[tuple[str, ...]] = ('attestation complete: true', 'SLSA complete: true', 'SBOM complete: true')
ALLOW_COMPLETE_MARKER: Final = 'provenance workflow complete: true'

def combined(paths: tuple[Path, ...]) -> str:
    return '\n'.join(path.read_text(encoding='utf-8') for path in paths)

def main() -> int:
    text = combined(DOCS)
    missing = [item for item in REQUIRED if item not in text]
    complete_claims = [item for item in COMPLETE_PATTERNS if item.lower() in text.lower()]
    completion_allowed = ALLOW_COMPLETE_MARKER in text
    if missing or (complete_claims and not completion_allowed):
        for item in missing:
            print(f'missing provenance policy term: {item}', file=sys.stderr)
        for item in complete_claims:
            print(f'unsupported completion claim: {item}', file=sys.stderr)
        return 1
    print('provenance policy OK')
    print('docs: RELEASE.md docs/release-provenance.md')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
