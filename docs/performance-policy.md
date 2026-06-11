# Performance gate policy

Bolt separates deterministic smoke performance from opt-in real-model
benchmarks. Default CI must not run long-running real-asset benchmarks.

| Command | Artifact | Mode | Schema/freshness validation |
| --- | --- | --- | --- |
| `cd engine && zig build run-infer --summary all` | `artifacts/bolt-mnist-run-infer.json` | portable smoke/manual | validate with `scripts/release/validate_artifact.py` when produced |
| `cd engine && zig build run-metal-mlp --summary all` | `artifacts/bolt-metal-mlp-runtime.json` | opt-in Metal/manual | validate with artifact registry when produced |
| `cd engine && zig build run-bonsai-bench --summary all` | `artifacts/bolt-bonsai-bench.json` | opt-in real assets/self-hosted | requires schema envelope, freshness metadata, and `manifest_digest` |
| `cd engine && zig build run-bonsai-q4-bench --summary all` | `artifacts/bolt-q4-bench.json` | opt-in real assets/self-hosted | requires schema envelope, freshness metadata, and `manifest_digest` |

Default CI may run deterministic build/test smoke checks only. Real Bonsai/Q4
benchmark gates are available through `.github/workflows/real-assets.yml` or a
manual runbook invocation when assets are provisioned.
