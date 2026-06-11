# Bolt artifact schema registry

This registry defines release-evidence JSON contracts for Bolt engine and Labrat artifacts. Release validators must evaluate the top-level artifact contract and the per-artifact fields listed here; they must not recursively search for truthy nested fields such as `tokens_match`, `matches_reference`, `ok`, or `passed` and treat them as whole-artifact pass criteria.

## Common envelope

Every release artifact used as evidence must include these top-level fields:

| Field | Type | Requirement |
| --- | --- | --- |
| `schema_version` | string | Must equal `1`. |
| `artifact_type` | string | One of the artifact types in this registry. |
| `status` | string | Must be `pass` for release-acceptable evidence. `fail`, `skip`, `error`, or missing status is not acceptable. |
| `command` | string | Exact command that produced the artifact. |
| `cwd` | string | Working directory used to run `command`. |
| `git_commit` | string | 7- to 40-character hex Git commit, `<sha>-dirty`, or `dirty` marker. Dirty worktree artifacts must be explicit and are not release-clean evidence. |
| `timestamp_utc` | string | ISO-8601 UTC timestamp ending in `Z`. |
| `toolchain` | object | Tool versions relevant to the artifact producer. Must contain at least one non-empty string value such as `zig` or `python`. |
| `manifest_digest` | string | Required for real Bonsai/Q4 asset artifacts. Must be `sha256:<64 hex chars>` for the manifest that selected the asset set. |

## Registered artifacts and pass conditions

| Artifact path/pattern | `artifact_type` | Release pass condition |
| --- | --- | --- |
| `artifacts/bolt-mnist-run.json` | `engine.mnist.run` | Common envelope passes and top-level `passes_reference_relative_threshold` is `true`. |
| `artifacts/bolt-mnist-run-1bit.json` | `engine.mnist.run_1bit` | Common envelope passes and top-level `selected_label_parity` is `true`. |
| `artifacts/bolt-mnist-run-infer.json` | `engine.mnist.run_infer` | Common envelope passes and top-level `inference.passed` is `true`. |
| `artifacts/bolt-metal-mlp-runtime.json` | `engine.metal.mlp_runtime` | Common envelope passes and top-level `runtime.passed` is `true`. |
| `artifacts/bolt-bonsai-smoke.json` | `engine.bonsai.smoke` | Common envelope passes and top-level `smoke.passed` is `true`. |
| `artifacts/bolt-bonsai-readiness.json` | `engine.bonsai.readiness` | Common envelope passes and top-level `readiness.passed` is `true`. |
| `artifacts/bolt-bonsai-bench.json` | `engine.bonsai.bench` | Common envelope passes and top-level `benchmark.passed` is `true`. |
| `artifacts/bolt-q4-golden.json` | `engine.q4.golden` | Common envelope passes and top-level `golden.tokens_match` is `true`. |
| `artifacts/bolt-q4-bench.json` | `engine.q4.bench` | Common envelope passes and top-level `benchmark.passed` is `true`. |
| `artifacts/labrat-*-agent.json` | `labrat.agent` | Common envelope passes and top-level `agent.exit_code` is `0`, `agent.compiled` is `true`, and `agent.passed` is `true`. |
| `artifacts/labrat-*-researcher.json` | `labrat.researcher` | Common envelope passes and top-level `researcher.passed` is `true`. |
| Labrat audit evidence | `labrat.audit` | Common envelope passes and top-level `audit.findings_block_release` is `false`. |
| Labrat blocker evidence | `labrat.blockers` | Common envelope passes and top-level `blockers.open_count` is `0`. |
| Labrat summary evidence | `labrat.summary` | Common envelope passes and top-level `summary.passed` is `true`. |

## Negative rule

The top-level `status` is authoritative. A document with `status: "fail"` is rejected even if a nested object contains fields like `tokens_match: true`, `matches_reference: true`, or `passed: true`.
