# PROJECT KNOWLEDGE BASE

**Generated:** 2026-06-10
**Commit:** 741b502
**Branch:** main

## OVERVIEW

Bolt is a local Apple-Silicon Zig + Metal execution workbench with a sandboxed
Labrat research-agent harness. Source is split into two Zig packages plus
optional Python reference scripts; default gates must stay portable.

## STRUCTURE

```text
bolt/
├── CODEX.md        # Canonical engineering policy and asset contract
├── README.md       # User-facing command API and verification sequence
├── worklog.md      # Timeline and recent gate evidence
├── engine/         # Zig/Metal runtime, CLIs, model loaders, source-local tests
├── labrat/         # Sandboxed agent/research harness and toolbox gates
├── research/       # Optional MLX/PyTorch comparison scripts, not default health
├── artifacts/      # Ignored/generated gate outputs and local evidence
└── data/           # Ignored real datasets/model assets
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Project policy | `CODEX.md` | Asset rules, style, testing, Zig-first tooling |
| Commands | `README.md` | Mirrors `engine/build.zig` and `labrat/build.zig` |
| Engine API | `engine/src/root.zig` | Facade exports; no `src/main.zig` app entry |
| Engine CLIs | `engine/src/cli/` | One file per installed executable |
| Metal runtime | `engine/src/metal/` | Bridge, shader catalog, kernels, dispatch checks |
| Runtime artifacts | `engine/src/runtime/` | Proof runs, reports, asset loaders |
| Tensor/model code | `engine/src/tensor/`, `engine/src/models/` | Layouts, buffers, MNIST/decoder paths |
| Labrat agents | `labrat/src/*_agent.zig` | Thin lane wrappers over `agent_cli_core.zig` |
| Labrat sandbox | `labrat/src/toolbox.zig` | Path guards, command allowlist, experiments |
| Labrat prompts | `labrat/programs/*.md` | Agent protocols and experiment constraints |
| Python references | `research/*.py` | Optional local diagnostics; may need MLX/Torch deps |

## CODE MAP

| Symbol / Surface | Type | Location | Role |
| --- | --- | --- | --- |
| `bolt` module | Zig module | `engine/build.zig`, `engine/src/root.zig` | Shared runtime facade |
| `Context` | Zig/ObjC bridge | `engine/src/metal/context.zig` | Metal buffers, dispatch |
| `bolt_metal_*` | C ABI | `engine/src/metal/bridge.m` | Metal pipeline and buffer implementation |
| `ShaderGroupInfo` | catalog | `engine/src/metal/shader_catalog.zig` | Kernel groups |
| `TransformerConfig` | shape contract | `engine/src/transformer.zig` | LLM shape validation |
| `ToolboxConfig` | sandbox config | `labrat/src/toolbox.zig` | Read/write scopes, commands, histories |
| `AgentSpec` | lane config | `labrat/src/agent_cli_core.zig` | Offline/live agent gate wiring |
| `ResearcherMode` | report mode | `labrat/src/researcher_core.zig` | Summary and benchmark report selection |

LSP note: `zls` currently exits with `ParseError` in this workspace, so use
`rg`, build scripts, and source-local `test` blocks as the reliable codemap.

## CONVENTIONS

- `CODEX.md` is the policy spine; do not silently weaken it in scoped files.
- Use Zig build steps as the command surface. There is no Makefile, package.json,
  pyproject, Docker, Nix, or CI workflow in the current repo.
- Keep tests source-local in Zig `test` blocks; root `tests/` is intentionally
  absent.
- Run `zig fmt` for Zig changes. Keep 4-space indentation and hard 100-column
  lines where practical.
- Prefer Zig for new tooling. Do not add external Zig dependencies unless the
  repo policy changes.
- Default health must not require provider credentials, large models, or real
  datasets.
- Real Bonsai/Q4 claims require the asset manifest and opt-in gates, not default
  tests.
- Generated outputs belong under ignored `artifacts/`, `data/`, `.zig-cache/`,
  `zig-out/`, or local model caches, not committed source.

## ANTI-PATTERNS (THIS PROJECT)

- Do not commit real datasets, model weights, provider transcripts, secrets, or
  local reference checkouts.
- Do not create a root `tests/` tree; add coverage beside the Zig source or via
  package build steps.
- Do not make default gates depend on `artifacts/assets/bolt-parity` being
  complete.
- Do not put placeholder values such as `TODO`, `FIXME`, `UNKNOWN`, `TBD`, or
  `PLACEHOLDER` into asset manifests; validators reject them.
- Do not read Metal shared buffers while the GPU may still write; wait or use
  the existing bridge synchronization pattern.
- Do not bypass Labrat sandbox path/command allowlists when changing agent tools.

## COMMANDS

```sh
cd engine && zig build test
cd engine && zig build
cd engine && zig build test-metal-shaders   # opt-in Apple Metal compile gate
cd labrat && zig build test
cd labrat && zig build api-offline-test
```

Full real-asset verification is documented in `README.md` and includes
`validate-assets`, `validate-tokenizer`, Bonsai/Q4 golden gates, and Labrat agent
smokes with `--summary all`.

## NOTES

- `engine/` and `labrat/` are separate Zig packages; only `labrat/` currently has
  `build.zig.zon`.
- `labrat/build.zig` invokes `../engine` steps for researcher gates; treat those
  as cross-package pipeline edges.
- `research/` scripts are optional diagnostics and may drift from current engine
  artifact names; verify paths before relying on comparisons.
- `artifacts/` and `data/` may exist locally but are ignored final-gate inputs or
  outputs, not durable source.
