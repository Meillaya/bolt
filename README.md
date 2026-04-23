# bolt

macOS-first Zig + Metal engine workbench with Python reference validation.

## Current status

This repo is past the initial bootstrap/proof milestones and is currently in a cleanup/hardening pass around the deterministic v1 proof surface before Phase 4 researcher-harness work.

The approved v1 direction is:
- one shared native runtime
- one small decoder-only LLM path
- one compact non-LLM path
- Python goldens/reference checks before benchmark work
- current milestone state: shared runtime + compact MNIST proof + small decoder-only LLM proof are implemented and green

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
manifest.json -> runtime bundle JSON -> tokenizer JSON + weights BIN
```

The current `weights.bin` is a tiny little-endian float32 bundle with:
- output projection values
- prompt-tail-conditioned transition bias values

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
