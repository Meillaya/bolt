# bolt reference-gap closure v1 completion report

Date: 2026-04-29

Related plan: `.omx/plans/prd-bolt-reference-gap-closure-v1.md`
Related test spec: `.omx/plans/test-spec-bolt-reference-gap-closure-v1.md`

## Executive verdict

The six scoped structural gaps against `reference/nnzap` v1 are closed for the intentionally small bolt slice. The implementation remains fixture-sized and deterministic, but every accepted path now has concrete runtime, kernel, proof, benchmark, and experiment evidence rather than proof-only summaries.

Final verdict: **complete for v1 scope**.

## Commits in this closure slice

- `0d6b0c1` — `Establish the first real shared-buffer Metal substrate`
- `ec50342` — `Ground benchmark gates in real GPU fixture paths`
- `e6a0e4e` — `Make local experiment replay consume benchmark gates`

## Phase-by-phase acceptance evidence

### Phase 1 — Shared-buffer runtime foundation

Status: **accepted**

Evidence:
- Metal-backed tensor/runtime APIs support shared-buffer allocation and explicit materialization/readback.
- Existing proof/compare flows still pass on the compatibility path.
- Unit coverage is included in `tests/unit/tensor_test.zig` and engine module tests.

Final verification commands:
- `cd engine && zig build test`
- `./research/scripts/run_compare.sh`
- `./research/scripts/run_proof.sh`

### Phase 2 — Core NN kernel suite + microbenchmark baseline

Status: **accepted**

Evidence:
- Reusable Metal primitive coverage exists for `add_f32`, `matmul_f32`, `bias_add_f32`, `relu_f32`, `reduce_sum_f32`, and `softmax_f32`.
- Correctness coverage is included in `tests/unit/kernel_test.zig` and engine module tests.
- Benchmark artifacts include primitive tensor metadata, elapsed timing observations, checksums, backend labels, and correctness gates.

Final artifact:
- `artifacts/bench/20260429T182601Z/kernel-bench.json`

### Phase 3 — Real MNIST GPU compute path

Status: **accepted**

Evidence:
- MNIST no longer passes through proof-only pixel-sum logic for inference.
- `run_mnist_fixture` reports `backend=metal` and dispatched kernels: `matmul_f32`, `bias_add_f32`, `softmax_f32`.
- Golden/integration coverage is included in `tests/integration/mnist_golden_test.zig`.

Final artifacts:
- `artifacts/bench/20260429T182601Z/mnist-inference.json`
- `artifacts/proof/20260429T182602Z/mnist-summary.json`
- `artifacts/proof/20260429T182602Z/mnist-check.txt`

### Phase 4 — Decoder / transformer / model-loading expansion

Status: **accepted for v1 scope**

Evidence:
- LLM runtime bundle now includes explicit model metadata in `llm-smoke.model.json`.
- `dump_llm_runtime`, `trace_llm_fixture`, and `run_llm_fixture` expose loader, model metadata, weights format, backend, and decoder kernel evidence.
- The conditioned decoder path dispatches Metal `bias_add_f32` and `softmax_f32` when Metal is available.
- Unsupported deferred formats such as `.safetensors` fail deterministically with `UnsupportedWeightsFormat` rather than silent stubs.
- Golden/integration coverage is included in `tests/integration/llm_golden_test.zig`.

Final artifacts:
- `artifacts/bench/20260429T182601Z/llm-inference.json`
- `artifacts/proof/20260429T182602Z/llm-summary.json`
- `artifacts/proof/20260429T182602Z/llm-check.txt`
- `artifacts/proof/20260429T182602Z/llm-unsupported-safetensors.txt`

### Phase 5 — Full benchmark infrastructure and regression gates

Status: **accepted**

Evidence:
- `./research/scripts/run_bench.sh` runs compare first, then kernel, MNIST, and LLM benchmark/evidence collection.
- `artifacts/bench/latest/manifest.json` is the unified benchmark manifest.
- `artifacts/bench/latest/summary.json` separates correctness gates from timing/model-shape observations.
- `python/scripts/validate_bench_artifacts.py` validates benchmark artifact consistency.

Final artifacts:
- `artifacts/bench/20260429T182601Z/manifest.json`
- `artifacts/bench/20260429T182601Z/summary.json`
- `artifacts/bench/20260429T182601Z/summary.txt`

Final benchmark gates:
- `primitive_benchmark_status`: pass
- `primitive_coverage`: pass
- `mnist_real_gpu_path`: pass
- `llm_real_decoder_path`: pass

### Phase 6 — Labrat-like autonomous experiment loop

Status: **accepted**

Evidence:
- `./research/scripts/run_experiment_recipe.sh ./research/recipes/smoke-local.json` runs an offline/local allowlisted recipe.
- The recipe runs compare, benchmark, and proof gates, then emits a replayable report with command exit codes, stdout/stderr captures, artifact paths, benchmark correctness gates, and a stable control-flow fingerprint.
- The loop consumes benchmark/proof artifacts instead of bypassing them.
- External services and arbitrary shell commands are rejected by recipe validation.

Final artifact:
- `artifacts/experiments/20260429T182601Z-smoke-local/report.json`

Final experiment evidence:
- outcome: `pass`
- control-flow fingerprint: `7836cb897a5773fab5b421e93e02a7fcf5c8f54bab58bee647542f39291cb874`
- commands: `compare-goldens`, `benchmark-gates`, `proof-gates`
- all command exit codes: `0`

## Final verification sweep

Executed successfully on 2026-04-29:

```bash
cd engine && zig build test
python3 ./tests/python/test_fixture_generation.py
./research/scripts/run_compare.sh
./research/scripts/run_bench.sh
./research/scripts/run_proof.sh
./research/scripts/run_debug.sh
./research/scripts/run_experiment_recipe.sh ./research/recipes/smoke-local.json
```

Final artifact directories:
- compare: `artifacts/compare/20260429T182601Z`
- benchmark: `artifacts/bench/20260429T182601Z`
- proof: `artifacts/proof/20260429T182602Z`
- debug: `artifacts/debug/20260429T182557Z`
- experiment: `artifacts/experiments/20260429T182601Z-smoke-local`

## Final architecture review notes

Accepted design constraints:
- The implementation extends the existing bolt proof harness instead of replacing it.
- MNIST and LLM paths share the same Metal primitive/runtime substrate rather than becoming separate demos.
- Benchmarks are informative gates with correctness/performance separation, not fragile hard timing thresholds.
- The experiment loop is local-only, replayable, and reviewable; mutation and network/model-agent calls remain out of v1 scope.

Known limitations retained intentionally:
- Safetensors/model import is not implemented; `.safetensors` is a deterministic unsupported format in v1.
- The LLM path is a tiny deterministic decoder smoke path, not a full transformer block.
- MNIST is inference-only; training/update support remains deferred.
- Non-Metal host fallback behavior for the new benchmark gate was not revalidated in the final sweep.
- Performance numbers are hardware/load-sensitive observations and should not be compared as strict thresholds.

## Recommended next direction

Start a new follow-on plan for **a tiny transformer block** before broad model import. The substrate now has shared buffers, primitive kernels, model metadata, proof/compare/benchmark gates, and an experiment loop, so the next most valuable slice is to add a minimal attention/MLP-style decoder block that can still be validated deterministically before safetensors or broader model import work.
