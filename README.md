# bolt ⚡

Apple-Silicon Zig + Metal execution workbench with a sandboxed Labrat research
and agent harness.

## Active source packages

- `engine/` — Bolt engine runtime, Zig/Metal kernels, tensor/model loaders,
  local proof CLIs, real-asset gates, and source-local tests.
- `labrat/` — sandboxed offline/live research-agent harness, toolbox tests,
  researcher summaries, and agent safety gates.
- `research/` — optional MLX/PyTorch comparison and diagnostic scripts for
  MNIST, Bonsai full-precision, and Q4 local analysis.

## Quick health check

```sh
cd engine && zig build test
cd engine && zig build
cd ../labrat && zig build test
```

Default checks are portable: they must not require large model files, provider
credentials, or platform-specific shader tooling unless explicitly requested by
an opt-in command.

## Engine command API

Run from `engine/`.

### Build and tests

```sh
zig build                         # build/install all engine executables
zig build uninstall               # remove installed engine executables
zig build test                    # source-local engine tests
zig build test-metal-shaders      # opt-in Metal shader compilation gate
```

### Installed engine binaries

`zig build` installs these direct CLIs under `engine/zig-out/bin/`.  Prefer the
`zig build <step>` wrappers for normal use because they encode prerequisites and
artifact locations.

```text
fixture_inspect
dump_llm_runtime
trace_llm_fixture
decode_llm_fixture
proof_fixture
run_mnist_fixture
run_llm_fixture
check_mnist_fixture
check_llm_fixture
train_mini_network
train_mnist
mnist_1bit
inference_bench
validate_assets
validate_tokenizer
bonsai
bonsai_golden
bonsai_bench
bonsai_q4_golden
bonsai_q4_bench
benchmark_kernels
```

### MNIST/runtime flows

```sh
zig build run                     # MNIST training/example flow
zig build run-1bit                # 1-bit MNIST integration flow
zig build run-infer               # MNIST inference benchmark flow
```

### Real asset validation

```sh
zig build validate-assets         # validate artifacts/assets/bolt-parity/asset-manifest.json
zig build validate-tokenizer      # validate model/tokenizer pairing and fixed chat-template IDs
```

All real-asset commands accept an explicit manifest path after `--` when needed:

```sh
zig build validate-assets -- ../artifacts/assets/bolt-parity/asset-manifest.json
zig build validate-tokenizer -- ../artifacts/assets/bolt-parity/asset-manifest.json
```

### Bonsai full-precision gates

```sh
zig build run-bonsai              # fast tensor-contract smoke gate
zig build run-bonsai-golden       # exact selected-token generated-output gate
zig build run-bonsai-bench        # correctness-gated decode benchmark
```

### Q4 gates

```sh
zig build run-bonsai-q4-golden    # exact selected-token Q4 generated-output gate with Metal Q4 decode
zig build run-bonsai-q4-bench     # correctness-gated Q4 decode benchmark
```

The real Bonsai/Q4 gates fail nonzero with setup guidance when the manifest or
referenced assets are missing.  With this workstation's current manifest/assets,
the full real-output chain passes and emits:

- `artifacts/bolt-bonsai-readiness.json`
- `artifacts/bolt-bonsai-bench.json`
- `artifacts/bolt-q4-golden.json`
- `artifacts/bolt-q4-bench.json`

## Asset manifest API

Default manifest path:

```text
artifacts/assets/bolt-parity/asset-manifest.json
```

Required real-asset entries:

- `mnist-raw`
- `bonsai-1.7b`
- `qwen3-1.7b-q4-gs64`

Diagnostic entry:

- `qwen3-1.7b-q4`

Manifest file paths are intentionally restricted to:

- repo-local `data/...`
- repo-local `artifacts/...`
- explicit `~/models/...`

Absolute paths, `.`/`..` segments, doubled separators, and unrelated repo paths
are rejected.  Real datasets and model weights remain local-only and ignored by
git.

## Labrat command API

Run from `labrat/`.

### Build and tests

```sh
zig build                         # build/install Labrat executables
zig build uninstall               # remove installed Labrat executables
zig build test                    # toolbox, tools, API, agent-core, researcher-core tests
zig build api-offline-test        # API client + agent-core offline tests only
```

### Agent gates

These build steps run each agent's offline mock path and then verify the live
path blocks safely when `LABRAT_LIVE=1` but no provider key is present.

```sh
zig build mnist-agent
zig build bonsai-agent
zig build bonsai-q4-agent
```

Installed agent binaries can also be run directly after `zig build`:

```sh
./zig-out/bin/mnist_agent
./zig-out/bin/bonsai_agent
./zig-out/bin/bonsai_q4_agent
```

Live provider execution is explicit opt-in and requires caller-provided
environment configuration:

```sh
LABRAT_LIVE=1 ANTHROPIC_API_KEY=... ./zig-out/bin/bonsai_agent
```

### Researcher gates

These build steps run the prerequisite engine gates and then write Labrat history
summaries.

```sh
zig build mnist-researcher
zig build bonsai-researcher
zig build bonsai-q4-researcher
```

Installed researcher binaries support the same command vocabulary:

```sh
./zig-out/bin/mnist_researcher summary
./zig-out/bin/mnist_researcher bench-compare
./zig-out/bin/mnist_researcher summaries

./zig-out/bin/bonsai_researcher summary
./zig-out/bin/bonsai_researcher bench-compare
./zig-out/bin/bonsai_researcher summaries

./zig-out/bin/bonsai_q4_researcher summary
./zig-out/bin/bonsai_q4_researcher bench-compare
./zig-out/bin/bonsai_q4_researcher summaries
```

Researcher outputs are written under:

```text
artifacts/labrat-history/<domain>/
```


## Research helper scripts

The `research/` folder contains optional comparison and diagnostic scripts copied
into Bolt as standalone project utilities. They are not required for normal Zig
verification and may require Python packages such as `mlx`, `mlx-lm`, `numpy`,
`torch`, or `torchvision`.

```sh
python research/mlx_reference.py              # MLX MNIST training/reference run
python research/mlx_inference.py              # MLX MNIST inference benchmark
python research/pytorch_reference.py          # PyTorch MNIST training/reference run
python research/pytorch_inference.py          # PyTorch MNIST inference benchmark
python research/mlx_bonsai.py                 # MLX Bonsai decode benchmark
python research/mlx_bonsai_q4_golden.py       # MLX Q4 golden-token capture
python research/mlx_q4_compare.py             # Q4 prompt/logit/token comparison
python research/mlx_q4_f16_test.py            # Q4 dtype diagnostic
python research/mlx_q4_layer0.py              # Q4 layer-0 activation diagnostic
python research/mlx_q4_layer_dump.py          # Q4 multi-layer activation dump
```

Q4 helper scripts search repo-local `data/qwen3-1.7b-q4-gs64`, then
`data/qwen3-1.7b-q4`, then matching directories under `~/models/`.

## Current full verification sequence

A complete local verification pass with real assets available is:

```sh
cd engine
zig build test --summary all
zig build --summary all
zig build validate-assets --summary all
zig build validate-tokenizer --summary all
zig build run-bonsai --summary all
zig build run-bonsai-golden --summary all
zig build run-bonsai-bench --summary all
zig build run-bonsai-q4-golden --summary all
zig build run-bonsai-q4-bench --summary all

cd ../labrat
zig build test --summary all
zig build api-offline-test --summary all
zig build mnist-agent --summary all
zig build bonsai-agent --summary all
zig build bonsai-q4-agent --summary all
```

Use the researcher build steps when you also want fresh Labrat history summaries;
they may invoke real engine gates and therefore inherit the same asset
requirements.

## Project notes

- `CODEX.md` contains the current repository map, asset contract, and engineering
  rules.
- `worklog.md` records the standalone execution timeline and recent gate
  evidence.
- Root `tests/` is intentionally absent; test coverage lives in source-local Zig
  `test` blocks unless project policy changes.
