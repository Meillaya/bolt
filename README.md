# bolt

macOS-first Zig + Metal engine workbench with Python reference validation.

## Current status

This repo is now past the initial bootstrap/proof milestones and has landed its first real Metal vertical slice on top of the deterministic v1 proof surface.

The approved v1 direction is:
- one shared native runtime
- one small decoder-only LLM path
- one compact non-LLM path
- Python goldens/reference checks before benchmark work
- current milestone state: shared runtime + Phase 2 primitive kernels + real MNIST GPU inference + small decoder-only LLM proof are implemented and green
- current native slice state:
  - committed Metal shader source in `engine/src/metal/kernels.metal`
  - Objective-C Metal bridge in `engine/src/metal/bridge.m`
  - Zig Metal runtime wrapper in `engine/src/metal/context.zig`
  - tensor buffer abstraction in `engine/src/tensor/buffer.zig`
  - Phase 1 shared-buffer runtime foundation is now landed: Metal-backed tensors keep CPU-visible shared storage and only materialize host-owned copies on demand
  - Phase 2 primitive baseline is now landed: matrix multiply, bias add, ReLU, reduce-sum, and softmax kernels have correctness tests and benchmark artifacts
  - Phase 3 MNIST inference now uses a runtime bundle plus Metal `matmul_f32` → `bias_add_f32` → `softmax_f32`
  - decoder proof path now loads explicit model metadata and uses Metal-backed `bias_add_f32` + `softmax_f32` for the conditioned route

## Toolchain

Verify the local setup with:

```bash
./research/scripts/doctor.sh
```

See:

```text
docs/toolchain-mac.md
docs/development.md
docs/architecture.md
```

## Current smoke commands

```bash
./research/scripts/run_check.sh
./research/scripts/run_smoke.sh
./research/scripts/run_compare.sh
./research/scripts/run_bench.sh
./research/scripts/run_proof.sh
./research/scripts/run_debug.sh
cd engine && zig build
./engine/zig-out/bin/proof_fixture run python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture check python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture debug python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture run python/fixtures/llm/manifest.json
./engine/zig-out/bin/proof_fixture check python/fixtures/llm/manifest.json
./engine/zig-out/bin/proof_fixture debug python/fixtures/llm/manifest.json 3
```

## Current LLM asset contract

The decoder proof path currently uses:

```text
manifest.json -> runtime bundle JSON -> model JSON + tokenizer JSON + weights BIN
```

The current `weights.bin` is a tiny little-endian float32 bundle with:
- output projection values
- prompt-tail-conditioned transition bias values

The current model JSON is deliberately tiny metadata (`loader`, `model_name`, `architecture`, vocab/hidden/context sizes) so native LLM runs can report which runtime bundle was loaded. `run_llm_fixture`, `trace_llm_fixture`, and `proof_fixture run` now report backend, loader/model metadata, weights format, dispatched decoder kernels, and the conditioned-route probability.
Deferred formats such as safetensors are intentionally rejected in v1 with deterministic `UnsupportedWeightsFormat` evidence rather than silent compatibility stubs.

Inspect the resolved contract with:

```bash
./engine/zig-out/bin/fixture_inspect ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/dump_llm_runtime ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/trace_llm_fixture ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/decode_llm_fixture ./python/fixtures/llm/manifest.json 3
./engine/zig-out/bin/proof_fixture debug ./python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture debug ./python/fixtures/llm/manifest.json 3
```

This keeps the local proof flow deterministic while moving one step closer to a more realistic asset layout.

The current multi-step decoder rollout now uses an internal score surface derived from the runtime weights bundle rather than replaying fixture logits for every generated step.
The debug/trace path now exposes both:
- fixture-logit-conditioned next-token scoring
- internal model-score next-token routing
- full-prompt context routing for the first decode step and summary/debug comparison
- explicit divergence markers in the run summary showing whether conditioning flipped the raw winner and how far the internal model score moved from the conditioned score
- internal prompt-context bias totals, so full-prompt routing is visible as projection + accumulated context bias

The rollout now keeps a growing decode context:
- step 0 uses the original prompt token list
- later steps use prompt + generated tokens
- each decode step now reports `projection_milli` and `context_bias_milli` so the internal score decomposition is visible

The run summary now treats `generated_last_token` as the internal prompt-context preview token rather than the raw fixture-logit winner.
That summary preview now also exposes:
- `raw_next_score_milli`
- `generated_projection_milli`
- `generated_context_bias_milli`

The unified debug trace now also exposes top-level score comparisons for:
- raw top logit
- conditioned top score
- model top score
- prompt-context top score
- route-to-route gain values between those winners
- an explicit preferred internal route marker

It now also includes a structured `route_comparison` object that groups:
- raw
- conditioned
- model
- prompt-context
- preferred
- gains

## Project layout

```text
engine/           Zig + Metal native core
python/           reference scripts and golden-fixture tooling
research/         deterministic local run/check/compare/bench scripts
docs/             architecture and development notes
reference/        local source/reference project
```

## Current first Metal vertical slice

The repo now contains real committed Metal code rather than only toolchain checks:

- `engine/src/metal/kernels.metal` — `copy_f32` and `add_f32`
- `engine/src/metal/kernels.metal` — `copy_f32`, `add_f32`, `matmul_f32`, `bias_add_f32`, `relu_f32`, `reduce_sum_f32`, and `softmax_f32`
- `engine/src/metal/bridge.m` — Objective-C bridge that compiles the embedded MSL source and dispatches compute kernels
- `engine/src/metal/context.zig` — Zig wrapper for context init, shared-buffer allocation, explicit materialization, vector ops, and primitive NN kernels
- `engine/src/tensor/buffer.zig` — owned tensor/buffer abstraction used by the proof paths

The current Phase 1 substrate change is that Metal-backed tensors now keep their values in CPU-visible shared Metal buffers instead of forcing a fresh host output allocation for every dispatch. Host copies still exist, but only when a caller explicitly materializes one for compatibility or artifact reporting.

The current Phase 2 addition keeps that substrate small but broadens the kernel surface enough for the next real-model phases. The Phase 3 MNIST path now consumes that surface through a deterministic dense-classifier runtime bundle:

```text
python/fixtures/mnist/manifest.json
  -> mnist-smoke.runtime.json
       -> mnist-smoke.weights.bin
```

`run_mnist_fixture` now reports `backend`, dispatched kernel evidence, logits in milli-units, probability milli-units, and the predicted label. `./research/scripts/run_bench.sh` emits timestamped primitive benchmark reports plus an MNIST inference artifact under `artifacts/bench/` after compare/reference checks pass.

This is still a deliberately small slice: it proves shader compilation, shared-buffer allocation, dispatch, real MNIST GPU inference, the first decoder-side Metal operation, and a reusable primitive-kernel baseline without breaking golden checks.
