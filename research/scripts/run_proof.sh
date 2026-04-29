#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_latest="artifacts/proof/latest"
artifact_run="artifacts/proof/${timestamp}"
mkdir -p "$artifact_latest" "$artifact_run"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

python3 ./python/scripts/generate_goldens.py --output-dir ./python/fixtures --family all >/dev/null

(
  cd engine
  zig build
)

./engine/zig-out/bin/proof_fixture run ./python/fixtures/mnist/manifest.json | tee "$artifact_run/mnist-summary.json" > "$artifact_latest/mnist-summary.json"
./engine/zig-out/bin/proof_fixture check ./python/fixtures/mnist/manifest.json | tee "$artifact_run/mnist-check.txt" > "$artifact_latest/mnist-check.txt"

./engine/zig-out/bin/proof_fixture run ./python/fixtures/llm/manifest.json | tee "$artifact_run/llm-summary.json" > "$artifact_latest/llm-summary.json"
./engine/zig-out/bin/proof_fixture check ./python/fixtures/llm/manifest.json | tee "$artifact_run/llm-check.txt" > "$artifact_latest/llm-check.txt"

unsupported_dir="$tmpdir/llm-unsupported-safetensors"
mkdir -p "$unsupported_dir"
cp ./python/fixtures/llm/* "$unsupported_dir/"
python3 - <<'PY' "$unsupported_dir/llm-smoke.runtime.json"
import json
import sys
from pathlib import Path

runtime_path = Path(sys.argv[1])
runtime = json.loads(runtime_path.read_text())
runtime["weights_file"] = "llm-smoke.weights.safetensors"
runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
(runtime_path.parent / "llm-smoke.weights.safetensors").write_bytes(b"unsupported placeholder")
PY

set +e
./engine/zig-out/bin/dump_llm_runtime "$unsupported_dir/manifest.json" \
  > "$artifact_run/llm-unsupported-safetensors.txt" 2>&1
unsupported_status=$?
set -e
cp "$artifact_run/llm-unsupported-safetensors.txt" "$artifact_latest/llm-unsupported-safetensors.txt"
if [ "$unsupported_status" -eq 0 ]; then
  echo "expected safetensors loader check to fail" >&2
  exit 1
fi
grep -q "UnsupportedWeightsFormat" "$artifact_run/llm-unsupported-safetensors.txt"

echo "proof artifacts: $artifact_run"
