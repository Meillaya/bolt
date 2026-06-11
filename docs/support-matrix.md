# Bolt support matrix and v1 scope

Bolt v1 is a local Apple Silicon Zig + Metal execution workbench with a
sandboxed Labrat research-agent harness. The release target is deliberately
narrow: default health checks are portable within the repository, while native
Metal, real assets, and live/manual paths stay opt-in.

## Supported environment

### Hardware

Apple Silicon Macs. GPU execution is designed for Apple Silicon and Apple's
Metal stack.

### Operating system

macOS with current Apple developer tooling. Default Zig checks should not require
shader tooling, but Metal opt-in gates require Xcode command-line tools.

### Zig

The repository-pinned Zig toolchain policy. Run the exact build-step commands
below from their package directories.

### Metal

Manual/self-hosted opt-in. Metal shader compilation uses `xcrun` and is not a
hosted default gate.

### Labrat

Offline sandbox checks by default. Live provider execution is not a default v1
release requirement.

Unsupported or uncommitted environments include Intel Macs, Linux, Windows,
mobile devices, and generic non-Apple GPUs. Those environments may be useful for
future portability experiments, but they are outside the v1 support claim.

## Default portable commands

These commands are the default release health scope. They are expected to run
without real assets, provider credentials, or platform-specific Metal shader
tooling:

```sh
cd engine && zig build test --summary all
cd engine && zig build --summary all
cd labrat && zig build test --summary all
cd labrat && zig build api-offline-test --summary all
```

The default checks must not require real datasets, local model weights,
`ANTHROPIC_API_KEY`, hosted provider credentials, or `xcrun` shader compilation.

## Opt-in self-hosted/manual commands

The following checks are explicit opt-in gates. They are appropriate for a
self-hosted Apple Silicon runner or a local maintainer machine with the required
assets and Apple shader tooling installed:

```sh
cd engine && zig build test-metal-shaders --summary all
cd engine && zig build run-metal-mlp --summary all
cd engine && zig build validate-assets --summary all
cd engine && zig build validate-tokenizer --summary all
cd engine && zig build run-bonsai --summary all
cd engine && zig build run-bonsai-golden --summary all
cd engine && zig build run-bonsai-bench --summary all
cd engine && zig build run-bonsai-q4-golden --summary all
cd engine && zig build run-bonsai-q4-bench --summary all
cd labrat && zig build mnist-agent --summary all
cd labrat && zig build bonsai-agent --summary all
cd labrat && zig build bonsai-q4-agent --summary all
cd labrat && zig build mnist-researcher --summary all
cd labrat && zig build bonsai-researcher --summary all
cd labrat && zig build bonsai-q4-researcher --summary all
```

When a self-hosted runner is unavailable, release evidence may record the gate as
manual-only with a clear reason, the missing prerequisite, and the latest local
maintainer evidence path. That fallback evidence is not a substitute for claiming
that the gate ran on hosted CI.

## Real assets and credentials

Real assets are local-only and ignored by git. Commands that validate or consume
real assets use the manifest under `artifacts/assets/bolt-parity/` unless an
explicit manifest path is provided after `--`. Real-asset checks must remain
manual/self-hosted opt-in and must fail with setup guidance when required assets
are absent.

Provider credentials are not part of the default v1 release scope. Labrat live
provider execution remains a caller-supplied, explicit opt-in path and must not be
required for default release health.

## non-goals for v1

Bolt v1 does not claim or ship:

- SaaS hosting or managed cloud execution.
- A graphical user interface (GUI).
- App Store packaging or distribution.
- broad cross-platform support beyond the Apple Silicon/macOS target.
- hosted provider integrations as a default release capability.
