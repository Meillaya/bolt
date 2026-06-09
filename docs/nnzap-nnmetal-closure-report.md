# nnzap nnmetal Closure / Labrat Phase 2 Readiness Report

Date: 2026-06-09
Scope: close `reference/nnzap/nnmetal` parity before starting full `reference/nnzap/labrat` parity.

## Closure Decision

Status: **closed/pass; Labrat Phase 2 may start only under its separate PRD/test-spec and credential-safe gates**.

The nnmetal parity surface has executable Bolt gates for Metal/runtime foundations, MNIST data/metrics, safetensors/model loading, tokenizer, Bonsai F16 selected-token decode, Q4 selected-token decode, benchmarks, build steps, and research-script coverage. The previously blocked Metal/runtime execution concern is resolved with Metal-backed MNIST gates, Q1 qmv regression coverage, and an integrated Q4 decode path that dispatches `q4mv_f32` during actual generation (`6468` projection dispatches plus `33` logits dispatches) while preserving exact 20/20 token parity. Final verification reran `zig build test`, `run_check.sh`, `run_smoke.sh`, and `run_bench.sh`; independent code-reviewer/architect reruns recorded APPROVE/CLEAR. Full Labrat implementation remains a separate Phase 2.

## Evidence Summary

| Area | Evidence |
| --- | --- |
| Milestones 1-8 | `.omx/ultragoal/goals.json` shows G002-G009 complete |
| Command mapping | `docs/nnzap-build-command-mapping.md` |
| Real asset gates | `artifacts/assets/nnzap-parity/asset-manifest.json`; `zig build validate-assets` requires MNIST, Bonsai, and `qwen3-1.7b-q4-gs64` |
| Tokenizer pairing | `zig build validate-tokenizer` validates Bonsai and reference-compatible Q4 tokenizers |
| Bonsai F16 golden | `artifacts/nnzap-milestone6-bonsai-readiness.json` status `pass` |
| Bonsai bench | `artifacts/nnzap-milestone8-bonsai-bench.json` status `pass`, gated on F16 golden and reruns real Bonsai decode benchmark workload with `real_decode_benchmark_ns` |
| Q4 golden | `artifacts/nnzap-milestone7-q4-golden.json` status `pass`, 20/20 exact tokens, integrated Metal decode `q4_projection_dispatches=6468` + `q4_logits_dispatches=33`, and sampled all-projection probe `dispatches=196` with `max_abs_error=0` |
| Q4 bench | `artifacts/nnzap-milestone7-q4-bench.json` status `pass`, gated on Q4 golden and reruns real integrated Metal Q4 decode benchmark workload with `real_decode_benchmark_ns` |
| Build/example closure | `artifacts/nnzap-milestone8-closure.json` status `pass` |
| Research scripts | `run_check.sh`, `run_smoke.sh`, and `run_bench.sh` reran for final closure; Milestone 8 also covered `run_compare.sh`, `run_proof.sh`, and `run_debug.sh` |

## Stable Interfaces Labrat May Consume

- Build commands:
  - `cd engine && zig build run`
  - `cd engine && zig build run-1bit`
  - `cd engine && zig build run-infer`
  - `cd engine && zig build run-bonsai`
  - `cd engine && zig build run-bonsai-golden`
  - `cd engine && zig build run-bonsai-bench`
  - `cd engine && zig build run-bonsai-q4-golden`
  - `cd engine && zig build run-bonsai-q4-bench`
  - `cd engine && zig build validate-assets`
  - `cd engine && zig build validate-tokenizer`
  - `cd engine && zig build test`
- Artifact schemas:
  - MNIST train/infer/Q1 JSON artifacts under `artifacts/nnzap-milestone3-*`.
  - Bonsai readiness/golden JSON under `artifacts/nnzap-milestone6-bonsai-readiness.json`.
  - Bonsai bench JSON under `artifacts/nnzap-milestone8-bonsai-bench.json`.
  - Q4 golden/bench JSON under `artifacts/nnzap-milestone7-q4-*.json`.
  - Research script timestamped artifacts under `artifacts/{bench,compare,debug,doctor,proof}/`.
- Ownership/contracts:
  - `docs/nnzap-interface-contracts.md`
  - `docs/nnzap-lane-ownership.md`
  - `docs/nnzap-thresholds.md`
  - `docs/nnzap-provenance.md`

## Labrat Phase 2 Boundary

Allowed next phase: port/adapt `reference/nnzap/labrat` toolbox, researcher, agent_core, API client, sandbox/snapshot/edit/build/test/bench/rollback/history behavior against the stable nnmetal commands above.

Not allowed in this milestone: enabling external API calls, storing secrets, or claiming autonomous Labrat parity without its own PRD/test-spec gates and explicit credential-safe execution policy.

## Final Review Resolution

- `artifacts/blockers/milestone9-final-review-blockers.json` is `resolved` after independent final rerun: code-reviewer `019eae7c-b1d4-71f2-94a8-d77673711a7e` recorded APPROVE and architect `019eae7c-cb7d-7681-8875-42422c8e04f2` recorded CLEAR.
- MNIST `run` and `run-infer` execute Metal-backed `matmul_f32` over real MNIST assets instead of CPU/toy final-gate paths.
- The runtime embeds and tests reference-compatible `qmv` and `q4mv_f32` Metal dispatch foundations; Q4 golden uses `q4mv_f32` for the actual integrated decode/generation path (`6468` projection dispatches plus `33` logits dispatches), and retains a sampled all-projection tensor probe (`196` dispatches, `max_abs_error=0`).
- `run-bonsai` is smoke-only; selected-token parity is enforced by `run-bonsai-golden`.
- Bench artifacts rerun real Bonsai/Q4 decode workloads, record `real_decode_benchmark_ns`, and `run_bench.sh` refreshed the `artifacts/bench/latest/*` pointers after review. Performance parity is not claimed beyond correctness-gated timing metadata.
- Final ultragoal completion requires only Codex goal reconciliation and OMX ledger checkpointing; the quality gate JSON is `.omx/state/ultragoal/final-quality-gate-g011.json`.

## Residual Risks for Phase 2

- Reference Labrat uses agent/API orchestration and may require current-Zig porting separate from nnmetal.
- API-backed agent flows are credential-gated and must have offline mocks before any live-provider run.
- Toolbox edit/build/test/rollback behavior is side-effectful; Phase 2 must isolate sandboxes and artifacts before write operations.
