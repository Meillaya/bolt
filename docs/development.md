# Development

## Prerequisites

- Xcode installed at `/Applications/Xcode.app`
- Metal Toolchain downloaded
- Zig installed
- Python 3 available

Run:

```bash
./research/scripts/doctor.sh
```

That script is the first gate before native work.

## Local workflow

### 1. Verify toolchain

```bash
./research/scripts/doctor.sh
```

### 2. Build the engine proof runtime

```bash
cd engine
zig build
```

That build now also compiles the Objective-C Metal bridge used by the first real compute kernels.

### 3. Current proof commands

These now provide the current deterministic proof/check/debug workflow:

```bash
research/scripts/run_check.sh
research/scripts/run_smoke.sh
research/scripts/run_compare.sh
research/scripts/run_bench.sh
research/scripts/run_proof.sh
research/scripts/run_debug.sh
```

Each script now saves its latest evidence under:

```text
artifacts/doctor/
artifacts/compare/
artifacts/proof/
artifacts/debug/
```

### 4. Native proof paths

Generate deterministic fixtures:

```bash
python3 ./python/scripts/generate_goldens.py --output-dir ./python/fixtures --family all
```

Build and run the native consumer:

```bash
cd engine && zig build
cd ..
./engine/zig-out/bin/proof_fixture run ./python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture check ./python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture run ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/proof_fixture check ./python/fixtures/llm/manifest.json
```

The current native proof implementation now includes:

- `engine/src/tensor/buffer.zig` for owned tensor storage plus explicit host materialization from shared Metal buffers
- `engine/src/metal/kernels.metal` for committed Metal kernels
- `engine/src/metal/bridge.m` for Metal device/library/pipeline dispatch and shared-buffer allocation
- `engine/src/metal/context.zig` for Zig-side runtime access and shared-buffer lifecycle

Current wiring:

- Metal-backed tensors now keep CPU-visible shared buffer storage and only materialize a host-owned copy on demand
- the MNIST proof path round-trips tensor data through the `copy_f32` Metal kernel before summary and trace logic
- the decoder proof path uses the `add_f32` Metal kernel for the conditioned logits-plus-bias route before selecting the top conditioned token

Compare expected vs native summary:

```bash
./research/scripts/run_compare.sh
```

That script saves fixture-inspect output, native summaries, and compare results into
`artifacts/compare/latest/` plus a timestamped run directory.

### 5. LLM runtime bundle layout

The LLM proof path now resolves assets through one bundle file:

```text
python/fixtures/llm/manifest.json
python/fixtures/llm/llm-smoke.runtime.json
python/fixtures/llm/llm-smoke.tokenizer.json
python/fixtures/llm/llm-smoke.weights.bin
```

Inspect the manifest-level contract with:

```bash
./engine/zig-out/bin/fixture_inspect ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/dump_llm_runtime ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/trace_llm_fixture ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/decode_llm_fixture ./python/fixtures/llm/manifest.json 3
./engine/zig-out/bin/proof_fixture debug ./python/fixtures/mnist/manifest.json
./engine/zig-out/bin/proof_fixture debug ./python/fixtures/llm/manifest.json 3
```

That now reports the resolved runtime bundle fields too:

```text
tokenizer_file=...
weights_file=...
tokenizer_vocab_size=4
weights_vocab_size=4
transition_bias_value_count=16
```

Current binary weights layout:
- first `vocab_size` float32 values: output projection
- next `vocab_size * vocab_size` float32 values: prompt-tail-conditioned transition bias matrix

The runtime dump command exposes the decoded native view:
- tokenizer vocab strings
- output projection values in milli-units
- transition-bias rows in milli-units

The trace command exposes one proof-step decision surface:
- prompt tail token
- raw top-token choice from logits
- conditioned top-token choice after transition bias
- model-top-token choice from the internal projection+bias score surface
- prompt-context top-token choice from the whole prompt token list
- top-level score values for each of those winners
- top-level gain values between those winner routes
- an explicit preferred-route label pointing at the internal route the repo currently treats as authoritative
- a structured `route_comparison` object for downstream tooling that wants grouped route data instead of flat fields
- per-token candidate scores in milli-units

The LLM run summary also exposes compact divergence markers:
- `raw_conditioning_flipped`
- `conditioned_matches_model`
- `model_condition_gap_milli`
- `prompt_context_next_*` fields for the whole-prompt internal score path
- `prompt_context_bias_milli` for the accumulated full-prompt transition bias on the chosen internal token
- `generated_last_token` now follows the internal prompt-context preview path rather than the raw fixture-logit winner
- `raw_next_score_milli` for the raw fixture-logit winner's score
- `generated_projection_milli` / `generated_context_bias_milli` for the one-step internal preview decomposition

The decode command exposes a tiny multi-step deterministic rollout:
- starts from the prompt tail token
- now uses an internal score surface built from:
  - output projection
  - transition bias
- each generated step uses a growing prompt-context window
- the decode JSON now includes `source_context_token_count` for each generated step
- the decode JSON also includes `projection_milli` and `context_bias_milli` for each generated token
- shows the conditioned token chosen at each decode step

The unified debug command combines the current LLM proof-debug surfaces into one JSON report:
- manifest metadata
- runtime bundle metadata
- one-step trace
- multi-step rollout

For MNIST, the same command now exposes:
- manifest metadata
- image shape
- predicted label
- top pixel location/value
- explicit non-zero pixel list

Preferred proof commands:

```bash
./engine/zig-out/bin/proof_fixture run ./python/fixtures/llm/manifest.json
./engine/zig-out/bin/proof_fixture check ./python/fixtures/llm/manifest.json
```

## Explicit verification files

In addition to inline Zig module tests, the repo now keeps spec-shaped test entrypoints in:

```text
tests/unit/layout_test.zig
tests/unit/tensor_test.zig
tests/unit/tokenizer_test.zig
tests/unit/weights_test.zig
tests/unit/manifest_test.zig
tests/integration/mnist_golden_test.zig
tests/integration/llm_golden_test.zig
tests/python/test_fixture_generation.py
```

`cd engine && zig build test` runs both the engine module tests and these explicit unit/integration test files.

## Scope notes

- v1 does **not** require an MLIR toolchain.
- Revisit MLIR only if a later architecture phase explicitly adopts an MLIR-based import or lowering path.
- Benchmarks come after golden/reference validation, not before.
