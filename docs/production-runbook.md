# Production runbook

This runbook sequences local release verification for Bolt. Capture command
output under `.omo/evidence/` using the task evidence policy. Use Python
3.11 or newer for every `python3 scripts/release/...` command; the CI
workflows provision Python 3.11 explicitly before invoking release validators.

## 1. Default portable health

```sh
bash scripts/release/baseline.sh
python3 scripts/release/check_versions.py
python3 scripts/release/check_support_matrix.py
python3 scripts/release/check_research_scope.py
python3 scripts/release/check_ci_policy.py .github/workflows/ci.yml
python3 scripts/release/check_readme_build_steps.py
python3 scripts/release/validate_artifact.py --all-fixtures
cd engine && zig build test --summary all
cd engine && zig build --summary all
cd labrat && zig build test --summary all
cd labrat && zig build api-offline-test --summary all
```

Expected evidence paths: `.omo/evidence/task-1-baseline.txt`,
`.omo/evidence/task-23-rc-rehearsal.txt`, and release checker outputs.

## 2. Optional Metal gate

```sh
python3 scripts/release/check_metal_workflow.py .github/workflows/metal.yml
cd engine && zig build test-metal-shaders --summary all
```

Expected evidence paths: `.omo/evidence/task-15-metal-workflow.txt` and
`.omo/evidence/task-15-metal-manual.txt`. If Metal/Xcode/xcrun is unavailable,
record `SKIPPED` with the exact missing capability.

## 3. Optional real assets

```sh
python3 scripts/release/check_real_asset_workflow.py .github/workflows/real-assets.yml
cd engine && zig build validate-assets -- ../artifacts/assets/bolt-parity/asset-manifest.json
cd engine && zig build validate-tokenizer -- ../artifacts/assets/bolt-parity/asset-manifest.json
cd engine && zig build run-bonsai-golden --summary all
cd engine && zig build run-bonsai-bench --summary all
cd engine && zig build run-bonsai-q4-golden --summary all
cd engine && zig build run-bonsai-q4-bench --summary all
cd labrat && zig build bonsai-researcher --summary all
cd labrat && zig build bonsai-q4-researcher --summary all
```

Expected artifact paths: `artifacts/bolt-bonsai-readiness.json`,
`artifacts/bolt-bonsai-bench.json`, `artifacts/bolt-q4-golden.json`,
`artifacts/bolt-q4-bench.json`, `artifacts/labrat-bonsai-researcher.json`,
and `artifacts/labrat-bonsai-q4-researcher.json`. Validate produced artifacts
with `python3 scripts/release/validate_artifact.py <artifact paths>`.

## 4. Labrat offline and live-provider policy

Expected artifact path: `artifacts/labrat-mnist-agent.json`.

```sh
cd labrat && zig build mnist-agent --summary all
cd labrat && zig build bonsai-agent --summary all
cd labrat && zig build bonsai-q4-agent --summary all
```

Live provider mode is out of v1 production scope. If `LABRAT_LIVE=1` is tested,
record fail-closed redacted evidence and do not include provider credentials in
artifacts.

## Troubleshooting

- Missing Zig: install Zig 0.16.0 and rerun `python3 scripts/release/check_versions.py`.
- Missing Xcode/xcrun: skip optional Metal with reason; default CI remains valid.
- Missing real assets: run `validate-assets` to identify the missing manifest,
  entry, referenced file, or placeholder metadata.
- Tokenizer mismatch: run `validate-tokenizer` and update only local assets or
  manifest metadata, not source policy.
- Schema validation failure: inspect `schema_version`, `status`, `command`,
  `cwd`, `git_commit`, `timestamp_utc`, `toolchain`, and `manifest_digest` for
  real-asset artifacts.
- Labrat sandbox denial: check the denied path/command against Labrat allowlists;
  do not weaken allowlists without a separate audited task.
- Live provider disabled: this is expected for v1; use offline mock/API tests.
