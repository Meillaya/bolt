#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_latest="artifacts/debug/latest"
artifact_run="artifacts/debug/${timestamp}"
mkdir -p "$artifact_latest" "$artifact_run"

python3 ./python/scripts/generate_goldens.py --output-dir ./python/fixtures --family all >/dev/null

(
  cd engine
  zig build
)

./engine/zig-out/bin/proof_fixture debug ./python/fixtures/mnist/manifest.json | tee "$artifact_run/mnist-debug.json" > "$artifact_latest/mnist-debug.json"
echo
./engine/zig-out/bin/proof_fixture debug ./python/fixtures/llm/manifest.json 3 | tee "$artifact_run/llm-debug.json" > "$artifact_latest/llm-debug.json"

echo "debug artifacts: $artifact_run"


echo

echo "==> nnzap real golden diagnostics"
(
  cd engine
  zig build run-bonsai-golden
  zig build run-bonsai-q4-golden
)
cp artifacts/nnzap-milestone6-bonsai-readiness.json "$artifact_run/bonsai-readiness.json"
cp artifacts/nnzap-milestone7-q4-golden.json "$artifact_run/q4-golden.json"
cp artifacts/nnzap-milestone6-bonsai-readiness.json "$artifact_latest/bonsai-readiness.json"
cp artifacts/nnzap-milestone7-q4-golden.json "$artifact_latest/q4-golden.json"
python3 - <<'PYDEBUG' "$artifact_run/bonsai-readiness.json" "$artifact_run/q4-golden.json"
import json, sys
for path in sys.argv[1:]:
    data=json.load(open(path))
    if data.get('status') != 'pass':
        raise SystemExit(f'{path} did not pass')
PYDEBUG
