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

This is deliberately not a full graph runtime yet. It is the first vertical slice that proves:

1. committed shader source
2. runtime shader compilation
3. Metal device + command queue setup
4. CPU-visible shared-buffer allocation plus explicit materialization-on-demand
5. one MNIST Metal-backed op first
6. one initial LLM-side Metal-backed op after MNIST parity stayed green

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
       -> llm-smoke.tokenizer.json
       -> llm-smoke.weights.bin
```

Responsibilities:
- `manifest.json` selects the proof family, payload, expected summary, and one runtime bundle file.
- `llm-smoke.runtime.json` resolves the runtime assets for that fixture.
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
