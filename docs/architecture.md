# Architecture

## v1 shape

The v1 proof milestone keeps the reference project's spirit while shrinking scope. The repo now has both proof paths implemented on that shared shape:

- native Zig + Metal center of gravity
- Python reference/golden validation
- deterministic local researcher workflow
- no autonomous self-editing loop in v1

## Current proof-oriented repo shape

```text
engine/
  build.zig
  src/
    root.zig
    metal/
      bridge.m
      context.zig
      kernels.metal
    tensor/
      buffer.zig
      layout.zig
    models/
      mnist.zig
      decoder.zig
python/
  reference/
  scripts/
research/
  scripts/
docs/
```

## Shared-core rule

The MNIST-class path and the small decoder-only LLM path must sit on top of the same runtime primitives rather than becoming separate demos.

## Current Metal execution slice

The repo now has a real, intentionally small Metal execution path:

- `tensor/buffer.zig` owns proof-tensor storage and explicit host materialization from shared Metal buffers
- `metal/context.zig` owns the Zig-side Metal runtime wrapper and shared-buffer lifecycle
- `metal/bridge.m` owns Objective-C framework interop
- `metal/kernels.metal` owns the committed compute kernels

Current committed kernels:

- `copy_f32` — used to round-trip MNIST tensor data through Metal before CPU-side proof summarization
- `add_f32` — used to start the decoder-side Metal path by computing conditioned logits plus one transition-bias row
- `matmul_f32` — first reusable dense-layer primitive for MNIST/decoder expansion
- `bias_add_f32` — row-wise bias application for batched dense outputs
- `relu_f32` — activation primitive
- `reduce_sum_f32` — scalar reduction primitive used to validate reduction dispatch shape
- `softmax_f32` — normalization primitive for logits/probability vectors

This is deliberately not a full graph runtime yet. It is the first vertical slice plus the Phase 2 primitive baseline and Phase 3 MNIST inference path that prove:

1. committed shader source
2. runtime shader compilation
3. Metal device + command queue setup
4. CPU-visible shared-buffer allocation plus explicit materialization-on-demand
5. a real MNIST dense-inference path backed by Metal matmul, bias add, and softmax
6. one initial LLM-side Metal-backed op after MNIST parity stayed green
7. a minimal reusable kernel suite with correctness coverage and timestamped benchmark artifacts

## MNIST runtime bundle contract

The MNIST fixture now has a small runtime bundle rather than deriving the label from pixel-sum proof math:

```text
manifest.json
  -> mnist-smoke.runtime.json
       -> mnist-smoke.weights.bin
```

Current binary layout:

```text
[ dense_weights (784 * 10 float32 values, row-major input x class) ]
[ bias (10 float32 values) ]
```

Native inference shape:

```text
pixels[1 x 784]
  -> matmul_f32(dense_weights[784 x 10])
  -> bias_add_f32(bias[10])
  -> softmax_f32
  -> argmax(logits)
```

`run_mnist_fixture` emits backend evidence and kernel-dispatch booleans alongside logits/probabilities in milli-units so the real path is distinguishable from the older proof-only pixel-sum label.

## Benchmark artifact contract

`research/scripts/run_bench.sh` is now a real benchmark gate rather than a placeholder. It first runs `run_compare.sh`, then builds and executes `engine/zig-out/bin/benchmark_kernels` plus the committed MNIST and LLM inference fixtures.

Each run writes:

```text
artifacts/bench/<timestamp>/kernel-bench.json
artifacts/bench/<timestamp>/mnist-inference.json
artifacts/bench/<timestamp>/llm-inference.json
artifacts/bench/<timestamp>/summary.json
artifacts/bench/<timestamp>/summary.txt
artifacts/bench/<timestamp>/manifest.json
artifacts/bench/latest/...
```

The benchmark report is intentionally simple and deterministic: it records backend, iteration count, primitive names, tensor-size metadata, elapsed nanoseconds, and a checksum for each primitive, then records MNIST and LLM backend/kernel evidence. `summary.json` separates `correctness_gates` from `performance_observations` so smoke-fixture pass/fail evidence is not confused with timing claims. Performance claims should remain scoped to this primitive + smoke-fixture baseline until broader MNIST/decoder benchmark suites land.

## Proof-surface rule

The preferred local execution surface is now the unified proof CLI:

```bash
./engine/zig-out/bin/proof_fixture run <manifest-path>
./engine/zig-out/bin/proof_fixture check <manifest-path>
```

Family-specific binaries still exist, but repo-facing scripts should prefer the shared proof surface unless a lower-level debugging step specifically needs the per-family entrypoints.

## LLM runtime bundle contract

The decoder-only proof path now uses a structured runtime asset bundle rather than loose sibling references in the manifest.

Current shape:

```text
manifest.json
  -> llm-smoke.runtime.json
       -> llm-smoke.model.json
       -> llm-smoke.tokenizer.json
       -> llm-smoke.weights.bin
```

Responsibilities:
- `manifest.json` selects the proof family, payload, expected summary, and one runtime bundle file.
- `llm-smoke.runtime.json` resolves the runtime assets for that fixture.
- `llm-smoke.model.json` carries tiny deterministic model metadata: loader, model name, architecture, vocab size, hidden size, and context length.
- `llm-smoke.tokenizer.json` carries the tiny vocab contract.
- `llm-smoke.weights.bin` carries raw little-endian float32 decoder weights.

Current binary layout:

```text
[ output_projection (vocab_size) ]
[ transition_bias (vocab_size * vocab_size) ]
```

Interpretation:
- `output_projection` preserves the raw top-logit proof signal.
- `transition_bias` adds a prompt-tail-conditioned next-token score so the proof path can act slightly more like a decode step instead of only summarizing logits.
- the multi-step decode rollout now uses this internal projection+bias score surface directly rather than reusing fixture logits at every generated step.
- the decode rollout now scores against a growing prompt context (prompt tokens plus generated tokens so far).
- each generated decode step now carries an explicit `projection + accumulated context bias` decomposition.
- the summary/trace surfaces now expose both views:
  - fixture-logit-conditioned scoring
  - internal model-score routing from projection+bias
- the conditioned route dispatches Metal `bias_add_f32` and `softmax_f32` when Metal is available, and reports backend plus dispatched-kernel evidence in the run/trace/debug surfaces
- unsupported deferred formats such as `.safetensors` fail deterministically with `UnsupportedWeightsFormat`; `run_proof.sh` captures that negative loader artifact alongside the positive proof summaries
- plus a whole-prompt context route derived from the full prompt token list
- and a decomposed whole-prompt bias total so context accumulation is visible separately from projection weights
- the summary’s one-step generated preview now aligns with the whole-prompt internal route instead of the raw fixture-logit winner
- that one-step preview is also decomposed into projection and accumulated context-bias components
- the summary still carries the raw winner score explicitly so the internal preview can be compared without switching to the debug trace
- the debug trace now exposes top-level winner scores for raw, conditioned, model, and prompt-context routes so the candidate table is not required for basic comparison
- the debug trace also exposes route-to-route gains so the effect of each internal routing layer is visible at a glance
- the preferred internal route is now called out explicitly so downstream tooling can treat prompt-context as the default top-level interpretation
- the debug trace additionally ships a structured `route_comparison` object so flat compatibility fields and grouped machine-readable route data can coexist
- the LLM summary also records whether conditioning flipped the raw winner and the milli-gap between conditioned and model-routed scores

Why this matters:
- keeps the manifest simpler
- creates one expansion point for future decoder asset growth
- gives the deterministic proof path one small conditioning surface beyond raw logits
- stays small enough for deterministic fixture generation and repo-local smoke runs

Native inspection surfaces now split into:
- `fixture_inspect` for manifest + bundle metadata
- `dump_llm_runtime` for decoded tokenizer/projection/transition-bias values
- `trace_llm_fixture` for one prompt-conditioned proof-step decision trace
- `decode_llm_fixture` for a tiny multi-step deterministic rollout
- `proof_fixture debug` for one unified family-aware proof-debug report

This is still a fixture-scoped contract, not a production model format. It is intentionally a stepping stone toward a more realistic imported asset story.

## Adding the next model family

The next family should extend the existing proof shape rather than fork it:

1. add a new native adapter under `engine/src/models/`
2. extend the shared manifest/runtime contract only where the existing proof surface cannot express the new family cleanly
3. add deterministic Python fixture generation under `python/scripts/` and committed smoke fixtures under `python/fixtures/`
4. add one explicit unit/integration test entrypoint under `tests/unit/` or `tests/integration/`
5. wire the family into the proof/check/debug workflow before any benchmark claims

The rule is still shared-core first: a new family is not complete until it participates in the same run/check/compare/debug loop as MNIST and the decoder proof path.

## MLIR stance

MLIR is intentionally out of the v1 proof path.

If adopted later, it should come from an explicit architecture decision with a clear payoff such as model import or lowering. It is not part of the v1 proof milestone's required toolchain.
