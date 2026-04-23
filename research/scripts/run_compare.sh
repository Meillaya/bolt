#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_latest="artifacts/compare/latest"
artifact_run="artifacts/compare/${timestamp}"
mkdir -p "$artifact_latest" "$artifact_run"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

python3 ./python/scripts/generate_goldens.py --output-dir "$tmpdir" >/dev/null
(
  cd engine
  zig build
)

./engine/zig-out/bin/fixture_inspect "$tmpdir/mnist/manifest.json" | tee "$artifact_run/mnist-inspect.txt" > "$artifact_latest/mnist-inspect.txt"
./engine/zig-out/bin/proof_fixture run \
  "$tmpdir/mnist/manifest.json" \
  > "$tmpdir/mnist/native-summary.json"
cp "$tmpdir/mnist/native-summary.json" "$artifact_run/mnist-native-summary.json"
cp "$tmpdir/mnist/native-summary.json" "$artifact_latest/mnist-native-summary.json"
./engine/zig-out/bin/proof_fixture check "$tmpdir/mnist/manifest.json" | tee "$artifact_run/mnist-check.txt" > "$artifact_latest/mnist-check.txt"
python3 ./python/scripts/compare_outputs.py \
  "$tmpdir/mnist/mnist-smoke.expected.json" \
  "$tmpdir/mnist/native-summary.json" | tee "$artifact_run/mnist-compare.txt" > "$artifact_latest/mnist-compare.txt"

./engine/zig-out/bin/fixture_inspect "$tmpdir/llm/manifest.json" | tee "$artifact_run/llm-inspect.txt" > "$artifact_latest/llm-inspect.txt"
./engine/zig-out/bin/proof_fixture run \
  "$tmpdir/llm/manifest.json" \
  > "$tmpdir/llm/native-summary.json"
cp "$tmpdir/llm/native-summary.json" "$artifact_run/llm-native-summary.json"
cp "$tmpdir/llm/native-summary.json" "$artifact_latest/llm-native-summary.json"
./engine/zig-out/bin/proof_fixture check "$tmpdir/llm/manifest.json" | tee "$artifact_run/llm-check.txt" > "$artifact_latest/llm-check.txt"
python3 ./python/scripts/compare_outputs.py \
  "$tmpdir/llm/llm-smoke.expected.json" \
  "$tmpdir/llm/native-summary.json" | tee "$artifact_run/llm-compare.txt" > "$artifact_latest/llm-compare.txt"

echo "compare artifacts: $artifact_run"
