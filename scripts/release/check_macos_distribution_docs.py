#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
from typing import Final

DOCS: Final[tuple[Path, ...]] = (Path('docs/macos-distribution.md'), Path('RELEASE.md'))
WORKFLOWS: Final[Path] = Path('.github/workflows')
REQUIRED: Final[tuple[str, ...]] = ('Developer ID', 'Hardened Runtime', 'notarization', 'stapling', 'not App Store')
FORBIDDEN: Final[tuple[str, ...]] = ('APPLE_ID_PASSWORD', 'APP_STORE_CONNECT', 'notarytool store-credentials', 'xcrun notarytool submit')

def workflow_text() -> str:
    return '\n'.join(path.read_text(encoding='utf-8') for path in WORKFLOWS.glob('*.yml'))

def main() -> int:
    text = '\n'.join(path.read_text(encoding='utf-8') for path in DOCS)
    missing = [item for item in REQUIRED if item not in text]
    wf = workflow_text()
    forbidden = [item for item in FORBIDDEN if item in wf]
    if missing or forbidden:
        for item in missing:
            print(f'missing macOS distribution term: {item}', file=sys.stderr)
        for item in forbidden:
            print(f'default/workflow signing credential detected: {item}', file=sys.stderr)
        return 1
    print('macOS distribution docs OK')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
