#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
from typing import Final

DOC: Final[Path] = Path('docs/performance-policy.md')
CI: Final[Path] = Path('.github/workflows/ci.yml')
REQUIRED: Final[tuple[str, ...]] = (
    'run-infer', 'bolt-mnist-run-infer.json', 'run-metal-mlp', 'bolt-metal-mlp-runtime.json',
    'run-bonsai-bench', 'bolt-bonsai-bench.json', 'run-bonsai-q4-bench', 'bolt-q4-bench.json',
    'opt-in real assets', 'manifest_digest',
)
FORBIDDEN_DEFAULT: Final[tuple[str, ...]] = ('run-bonsai-bench', 'run-bonsai-q4-bench', 'run-metal-mlp')

def main() -> int:
    doc = DOC.read_text(encoding='utf-8')
    ci = CI.read_text(encoding='utf-8')
    missing = [item for item in REQUIRED if item not in doc]
    forbidden = [item for item in FORBIDDEN_DEFAULT if item in ci]
    if missing or forbidden:
        for item in missing:
            print(f'missing performance policy item: {item}', file=sys.stderr)
        for item in forbidden:
            print(f'default CI includes non-portable benchmark: {item}', file=sys.stderr)
        return 1
    print('performance policy OK')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
