#!/usr/bin/env bash
set -euo pipefail

python3 ./python/scripts/generate_goldens.py --output-dir ./python/fixtures --family all >/dev/null

(
  cd engine
  zig build
)

echo "==> native mnist"
./engine/zig-out/bin/proof_fixture run ./python/fixtures/mnist/manifest.json

echo
echo "==> native llm"
./engine/zig-out/bin/proof_fixture run ./python/fixtures/llm/manifest.json

echo
echo "==> compare"
./research/scripts/run_compare.sh
