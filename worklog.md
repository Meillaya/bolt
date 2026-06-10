# Bolt Worklog

This worklog records Bolt as a standalone Apple-Silicon Zig + Metal execution
project.  It describes the evolution of the engine, Labrat harness, asset gates,
and verification story in terms of Bolt-owned decisions, artifacts, and health
checks.  Historical compatibility work is described as external behavior
validation, not as a port narrative.

## Current operating shape

Bolt now consists of two active source packages:

- `engine/`: the runtime, Zig/Metal kernels, tensor/model loaders, local CLIs,
  real-asset gates, and source-local Zig tests.
- `labrat/`: the sandboxed researcher/agent harness, offline/live safety gates,
  toolbox, and domain researcher summaries.

Default health is portable and source-local:

```sh
cd engine && zig build test
cd engine && zig build
cd labrat && zig build test
```

Real model behavior is opt-in.  It is claimable only when local assets are
present, the asset manifest validates, and the corresponding golden/bench gates
emit pass artifacts.

## Timeline

### 2026-04-23 — Deterministic engine baseline

- Established the first deterministic proof baseline for the Bolt engine.
- Built the initial shape around explicit fixtures, bounded tensor layouts, and
  local reproducibility rather than ad hoc smoke output.
- Created the first real shared-buffer Metal substrate, giving the engine a
  concrete Apple-Silicon execution path instead of treating Metal as a future
  placeholder.
- Early focus: make CPU/GPU contracts visible, testable, and narrow enough to
  debug when kernels or layouts disagree.

Evidence in git history:

- `f0dbc29` — Freeze deterministic v1 proof baseline before expanding scope.
- `0d6b0c1` — Establish first real shared-buffer Metal substrate.

### 2026-04-29 — Benchmark and replay gates become real inputs

- Grounded benchmark gates in real GPU fixture paths.
- Made local experiment replay consume benchmark gate outputs instead of relying
  on disconnected scripts.
- Recorded reference-gap closure evidence as a project artifact so future work
  could distinguish measured behavior from assumptions.
- Shifted the project toward command-level accountability: if a flow matters, it
  should have a runnable command, an output artifact, and an explicit pass/fail
  condition.

Evidence in git history:

- `ec50342` — Ground benchmark gates in real GPU fixture paths.
- `e6a0e4e` — Make local experiment replay consume benchmark gates.
- `1997836` — Record reference-gap closure completion evidence.

### 2026-06-09 — Engine and Labrat closures land

- Closed the main engine behavior gap with default health checks passing.
- Completed Labrat Phase 2 as a Bolt-owned research and agent harness:
  - offline API-shaped scenarios,
  - credential-safe live-mode blocking,
  - sandboxed toolbox behavior,
  - researcher summaries,
  - rollback/audit-oriented agent CLIs.
- Hardened Labrat after independent review by fixing unsafe cleanup, symlink
  sandbox handling, subprocess bounds, redaction, parser edge cases, and command
  failure propagation.
- The outcome was a coherent local system: engine commands produce artifacts;
  Labrat consumes and summarizes them without requiring provider access by
  default.

Evidence in git history:

- `923ea94` — Close low-level engine parity before Labrat phase two.
- `b3364af` — Complete Labrat Phase 2 parity closure.

### 2026-06-10 — Production-readiness cleanup

- Reworked the root handoff docs so the repository presents itself as Bolt: a
  concise Apple-Silicon Zig + Metal workbench with a Labrat harness.
- Tightened `.gitignore` around local data, generated artifacts, build caches,
  secrets, editor state, and local runtime state.
- Preserved the project rule that default gates stay portable and real assets
  remain opt-in.
- Kept the root `tests/` tree absent; required coverage lives in source-local
  Zig `test` blocks.

Evidence in git history:

- `1e9d4f7` — Harden handoff docs for production readiness.

### 2026-06-10 — Full command surface and cleanup pass

- Rebuilt the engine command surface around Bolt-neutral names:
  - `run`, `run-1bit`, `run-infer`,
  - `validate-assets`, `validate-tokenizer`,
  - `run-bonsai`, `run-bonsai-golden`, `run-bonsai-bench`,
  - `run-bonsai-q4-golden`, `run-bonsai-q4-bench`,
  - `test-metal-shaders`.
- Removed stale root test references from the engine build graph and kept tests
  source-local.
- Deleted stale/generated scripts, docs, root test files, and obsolete artifact
  placeholders after recording deletion evidence.
- Verified engine and Labrat health after cleanup:
  - engine build passed,
  - engine tests passed,
  - Labrat tests passed,
  - Metal shader compilation passed on the local Apple-Silicon toolchain.
- Archived the durable execution evidence under `.omx/` and cancelled the active
  workflow state after completion.

Representative evidence paths:

- `.omx/ultragoal-cancelled-20260610T155540Z/evidence/g043/final2-engine-test.log`
- `.omx/ultragoal-cancelled-20260610T155540Z/evidence/g043/final2-engine-build.log`
- `.omx/ultragoal-cancelled-20260610T155540Z/evidence/g043/final2-labrat-test.log`
- `.omx/ultragoal-cancelled-20260610T155540Z/evidence/g043/final2-metal-shaders.log`

### 2026-06-10 — Local asset manifest recreated

- Confirmed the workstation asset layout:
  - full-precision Bonsai assets live under `~/models/bonsai-1.7b`,
  - Q4 and MNIST assets live under ignored `data/` directories.
- Added a local `data/bonsai-1.7b` symlink to the model cache for ergonomic
  discovery while keeping the manifest able to use explicit `~/models/` paths.
- Recreated `artifacts/assets/bolt-parity/asset-manifest.json` with concrete
  SHA-256 digests, sizes, roles, provenance fields, and model/tokenizer pairing
  metadata.
- Hardened manifest path resolution so only `data/`, `artifacts/`, and
  `~/models/` asset paths are accepted.  Absolute paths, `.`/`..` segments, and
  doubled separators are rejected.
- Validated the manifest and tokenizer pairing:
  - `validate-assets` passed,
  - `validate-tokenizer` passed,
  - `run-bonsai` smoke passed.

Current manifest:

- `artifacts/assets/bolt-parity/asset-manifest.json`

### 2026-06-10 — Real Bonsai and Q4 generated-output proof

- Addressed the last unproven real-output gates.
- Initial bounded run showed the full-precision Bonsai golden gate exceeded 120s
  because it repeatedly reloaded layers and rescanned embeddings.
- Optimized the full-precision golden path by preloading the F16 model once,
  avoiding prompt-token logit scans, and stopping early on mismatch.
- Fixed a misleading artifact behavior: the Bonsai blocker artifact is now
  removed on pass and only written when the gate is actually blocked.
- Re-ran the complete real-asset chain under explicit time bounds.

Final real-output gate evidence:

| Gate | Result | Evidence |
| --- | --- | --- |
| `validate-assets` | pass | `.omx/adhoc-evidence/heavy-gates/final/validate-assets.log` |
| `validate-tokenizer` | pass | `.omx/adhoc-evidence/heavy-gates/final/validate-tokenizer.log` |
| `run-bonsai` | pass | `.omx/adhoc-evidence/heavy-gates/final/run-bonsai.log` |
| `run-bonsai-golden` | pass, 11/11 exact generated tokens | `.omx/adhoc-evidence/heavy-gates/final/run-bonsai-golden.log` |
| `run-bonsai-bench` | pass, correctness-gated timing emitted | `.omx/adhoc-evidence/heavy-gates/final/run-bonsai-bench.log` |
| `run-bonsai-q4-golden` | pass, 20/20 exact generated tokens, integrated Metal Q4 decode used | `.omx/adhoc-evidence/heavy-gates/final/run-bonsai-q4-golden.log` |
| `run-bonsai-q4-bench` | pass, correctness-gated timing emitted | `.omx/adhoc-evidence/heavy-gates/final/run-bonsai-q4-bench.log` |

Current real-output artifacts:

- `artifacts/bolt-bonsai-readiness.json`
- `artifacts/bolt-bonsai-bench.json`
- `artifacts/bolt-q4-golden.json`
- `artifacts/bolt-q4-bench.json`

The real-output claim is therefore valid for this workstation's supplied local
asset set and manifest.  It remains intentionally contingent on those external
assets being present and matching the manifest.

## Standing project decisions

- Bolt is maintained as a standalone codebase and product surface.
- Default checks must remain portable and must not require large local assets,
  provider credentials, or Metal shader tooling unless the command is explicitly
  opt-in.
- Real model claims require concrete manifest entries and passing real gates.
- Tests belong next to the implementation unless the project policy changes.
- Generated artifacts, model weights, datasets, provider transcripts, and local
  runtime state stay out of version control.
- Documentation should describe Bolt's current behavior and execution evidence,
  not its relationship to an external codebase.

## Current health snapshot

As of the latest verification pass:

- `cd engine && zig build test --summary all` — pass, `71/71` tests.
- `cd engine && zig build --summary all` — pass, `43/43` steps.
- `cd engine && zig build validate-assets --summary all` — pass with complete
  MNIST, Bonsai, Q4 group-size-64, and diagnostic Q4 assets.
- `cd engine && zig build validate-tokenizer --summary all` — pass with fixed
  chat-template token IDs and config/tokenizer compatibility checks.
- `cd engine && zig build run-bonsai-golden` — pass with exact selected-token
  generation for full-precision Bonsai.
- `cd engine && zig build run-bonsai-q4-golden` — pass with exact selected-token
  generation for Q4 and integrated Metal Q4 decode.
