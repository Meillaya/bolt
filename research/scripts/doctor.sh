#!/usr/bin/env bash
set -euo pipefail

failures=0

check() {
  local name="$1"
  local cmd="$2"
  echo "==> ${name}"
  if eval "$cmd"; then
    echo "ok: ${name}"
  else
    echo "fail: ${name}"
    failures=$((failures + 1))
  fi
  echo
}

echo "bolt toolchain doctor"
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "shell: ${SHELL:-unknown}"
echo "developer_dir: ${DEVELOPER_DIR:-<unset>}"
echo

check "zig installed" "command -v zig >/dev/null && zig version"
check "xcode developer dir" "test -d /Applications/Xcode.app/Contents/Developer && DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -version"
check "metal compiler available" "DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun --find metal"
check "metal compile smoke test" '
  tmpdir=$(mktemp -d)
  trap "rm -rf \"$tmpdir\"" RETURN
  cat > "$tmpdir/minimal.metal" <<"MSL"
#include <metal_stdlib>
using namespace metal;
kernel void noop(device float *buffer [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    buffer[id] = buffer[id];
}
MSL
  DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun -sdk macosx metal -c "$tmpdir/minimal.metal" -o "$tmpdir/minimal.air"
  test -f "$tmpdir/minimal.air"
'
check "python3 available" "command -v python3 >/dev/null && python3 --version"

if [ "$failures" -eq 0 ]; then
  echo "result: PASS"
else
  echo "result: FAIL (${failures} check(s) failed)"
  exit 1
fi
