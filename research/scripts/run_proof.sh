#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_latest="artifacts/proof/latest"
artifact_run="artifacts/proof/${timestamp}"
mkdir -p "$artifact_latest" "$artifact_run"

python3 ./python/scripts/generate_goldens.py --output-dir ./python/fixtures --family all >/dev/null

(
  cd engine
  zig build
)

./engine/zig-out/bin/proof_fixture run ./python/fixtures/mnist/manifest.json | tee "$artifact_run/mnist-summary.json" > "$artifact_latest/mnist-summary.json"
./engine/zig-out/bin/proof_fixture check ./python/fixtures/mnist/manifest.json | tee "$artifact_run/mnist-check.txt" > "$artifact_latest/mnist-check.txt"

./engine/zig-out/bin/proof_fixture run ./python/fixtures/llm/manifest.json | tee "$artifact_run/llm-summary.json" > "$artifact_latest/llm-summary.json"
./engine/zig-out/bin/proof_fixture check ./python/fixtures/llm/manifest.json | tee "$artifact_run/llm-check.txt" > "$artifact_latest/llm-check.txt"

echo "proof artifacts: $artifact_run"
