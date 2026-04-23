#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
doctor_dir_latest="artifacts/doctor/latest"
doctor_dir_run="artifacts/doctor/${timestamp}"
mkdir -p "$doctor_dir_latest" "$doctor_dir_run"

doctor_tmp="$(mktemp)"
trap 'rm -f "$doctor_tmp"' EXIT

./research/scripts/doctor.sh | tee "$doctor_tmp"
cp "$doctor_tmp" "$doctor_dir_latest/doctor.txt"
cp "$doctor_tmp" "$doctor_dir_run/doctor.txt"

(
  cd engine
  zig build
  zig build test
)

python3 ./tests/python/test_fixture_generation.py

echo "doctor artifact: $doctor_dir_run/doctor.txt"
