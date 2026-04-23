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
