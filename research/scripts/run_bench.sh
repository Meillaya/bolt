#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
artifact_latest="artifacts/bench/latest"
artifact_run="artifacts/bench/${timestamp}"
mkdir -p "$artifact_latest" "$artifact_run"

./research/scripts/run_compare.sh
(
  cd engine
  zig build
)

./engine/zig-out/bin/benchmark_kernels \
  | tee "$artifact_run/kernel-bench.json" \
  > "$artifact_latest/kernel-bench.json"

./engine/zig-out/bin/run_mnist_fixture ./python/fixtures/mnist/manifest.json \
  | tee "$artifact_run/mnist-inference.json" \
  > "$artifact_latest/mnist-inference.json"

./engine/zig-out/bin/run_llm_fixture ./python/fixtures/llm/manifest.json \
  | tee "$artifact_run/llm-inference.json" \
  > "$artifact_latest/llm-inference.json"

python3 ./python/scripts/validate_bench_artifacts.py \
  "$artifact_run" \
  --latest-dir "$artifact_latest" \
  --timestamp "$timestamp"

cat > "$artifact_run/manifest.json" <<MANIFEST
{
  "schema_version": 1,
  "benchmark_kind": "kernel-and-inference",
  "backend": "metal",
  "timestamp_utc": "$timestamp",
  "artifacts": {
    "kernel_report": "kernel-bench.json",
    "mnist_inference": "mnist-inference.json",
    "llm_inference": "llm-inference.json",
    "summary_json": "summary.json",
    "summary_text": "summary.txt"
  },
  "correctness_gates": [
    "primitive_benchmark_status",
    "primitive_coverage",
    "mnist_real_gpu_path",
    "llm_real_decoder_path"
  ],
  "performance_observations": [
    "primitive_elapsed_ns",
    "model_smoke_shapes"
  ]
}
MANIFEST
cp "$artifact_run/manifest.json" "$artifact_latest/manifest.json"

python3 - <<'PY' "$artifact_run" "$artifact_latest"
import json
import sys
from pathlib import Path
for directory in (Path(sys.argv[1]), Path(sys.argv[2])):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / manifest["artifacts"]["summary_json"]).read_text())
    if manifest["schema_version"] != 1 or summary["schema_version"] != 1:
        raise SystemExit(f"invalid benchmark schema in {directory}")
    if manifest["benchmark_kind"] != summary["benchmark_kind"]:
        raise SystemExit(f"benchmark kind mismatch in {directory}")
    gate_names = {gate["name"] for gate in summary["correctness_gates"]}
    missing = set(manifest["correctness_gates"]) - gate_names
    if missing:
        raise SystemExit(f"summary missing gates in {directory}: {sorted(missing)}")
PY

echo "benchmark artifacts: $artifact_run"
