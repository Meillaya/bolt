# ENGINE KNOWLEDGE BASE

## OVERVIEW

`engine/` is the Bolt runtime package: Zig modules, Metal bridge/kernels, local
proof CLIs, real-asset gates, and source-local tests.

## STRUCTURE

```text
engine/
├── build.zig          # All engine executable, run, test, and Metal steps
└── src/
    ├── root.zig       # Public `bolt` module facade
    ├── cli/           # Installed proof/check/bench executables
    ├── fixtures/      # Manifest and deterministic payload contracts
    ├── metal/         # Metal bridge, shader catalog, kernels
    ├── models/        # MNIST and decoder paths
    ├── runtime/       # Asset loaders, proof runs, reports
    ├── tensor/        # Layout and buffer primitives
    └── testing/       # Embedded deterministic samples
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Build steps | `build.zig` | Source of truth for installed CLIs and wrappers |
| Public imports | `src/root.zig` | Add exports here when a module becomes public |
| Asset paths | `src/cli/asset_paths.zig` | Allowed roots and path rejection |
| Asset manifest | `src/cli/validate_assets.zig` | Concrete metadata checks |
| MNIST flows | `src/cli/train_mnist.zig`, `src/runtime/mnist_*` | Default portable gates |
| Bonsai full precision | `src/cli/bonsai*.zig`, `src/bonsai_model.zig` | Real-asset opt-in gates |
| Q4 path | `src/q4.zig`, `src/cli/bonsai_q4*.zig` | Q4 golden/bench logic |
| Tensor layout | `src/tensor/layout.zig`, `src/tensor/buffer.zig` | Comptime sizes and offsets |
| LLM decode | `src/transformer.zig`, `src/models/decoder.zig` | Shape and token pipeline |

## CONVENTIONS

- No single `src/main.zig`: `build.zig` installs many CLI entry files from
  `src/cli/*.zig` and exposes shared code through `src/root.zig`.
- Tests live in source modules. The `test` step covers module decl tests plus
  asset path tests.
- Real asset commands must fail nonzero with setup guidance when manifest entries
  or referenced files are missing.
- Keep manifest paths limited to repo-local `data/`, repo-local `artifacts/`, or
  explicit `~/models/` entries; reject absolute paths and `.`/`..` segments.
- Prefer fixed limits, explicit shape validation, source-local assertions, and
  comptime-evaluable layout calculations.
- Use explicit error returns for setup and Metal failures; do not hide them with
  generic panics or `catch unreachable`.

## ANTI-PATTERNS

- Do not add external Zig packages for small domain-specific helpers.
- Do not make `zig build test` require large model files, MNIST downloads, or
  Apple shader tooling beyond the current optional Metal tests.
- Do not rename build steps without updating `README.md`, `labrat/build.zig`, and
  researcher artifact expectations together.
- Do not cache a Metal shared-buffer slice across a GPU submission boundary.
- Do not add generated binaries, `.air` files, local assets, or benchmark output
  to source control.

## COMMANDS

```sh
zig build test --summary all
zig build --summary all
zig build test-metal-shaders --summary all
zig build validate-assets --summary all
zig build validate-tokenizer --summary all
```

Run real Bonsai/Q4 gates only when the default asset manifest or an explicit
manifest path is complete.
