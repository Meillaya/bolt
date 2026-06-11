# Contributing to Bolt

Bolt is a local Apple-Silicon Zig + Metal execution workbench with a sandboxed
Labrat research-agent harness. Contributions should keep the v1 product local,
portable by default, and explicit about opt-in native or real-asset gates.

## Before changing code

- Read `CODEX.md` and the nearest `AGENTS.md` for the files you touch.
- Keep changes small, reviewable, and reversible.
- Reuse existing Zig build steps and source-local tests before adding new
  scripts or abstractions.
- Do not add external dependencies unless the release plan explicitly requires
  them and the decision is documented.

## Verification expectations

Run the narrowest relevant checks first, then the package-level checks when the
change affects shared behavior. Default health must not require provider
credentials, large models, real datasets, or platform-specific shader tooling.

Common local gates:

```sh
cd engine && zig build test
cd engine && zig build
cd labrat && zig build test
cd labrat && zig build api-offline-test
```

Opt-in Metal shader and real-asset checks are not default contribution gates.
When you run them, capture fresh evidence and label the platform, asset source,
and command clearly.

## Evidence and generated files

Generated evidence is run-local. Do not commit generated `.omo/evidence/`,
`artifacts/`, `data/`, `.zig-cache/`, `zig-out/`, provider transcripts, model
weights, datasets, secrets, or local reference checkouts.

## Documentation changes

Update the root README, release metadata, or support docs whenever a command,
release gate, security boundary, or artifact contract changes. Public docs must
avoid private credentials, real model paths, and generated evidence.
