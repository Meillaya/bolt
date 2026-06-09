# nnzap lane and file ownership map — Milestone 0

Non-owner lanes may not edit owned shared surfaces directly. They must submit a patch queue item or coordinate through the owner. Reviewer lanes are responsible for interface conformance, not implementation ownership.

| Surface / files | Owner lane | Reviewer lane | Conflict rule |
| --- | --- | --- | --- |
| `engine/build.zig`, build steps, command aliases | build/CLI lane | verifier lane | Only build/CLI edits build graph; other lanes request executable/test wiring through build owner. |
| `engine/src/root.zig`, public exports | architect/API lane | verifier lane | Export changes require matching inventory/threshold update. |
| `engine/src/metal/context.zig`, `engine/src/metal/bridge.m` | Metal runtime lane | architect/API lane | Runtime lane owns buffer/pipeline APIs; shader/model lanes consume stable APIs. |
| shader files under future `engine/src/metal/*` / current `kernels.metal` | Metal shader lane | test/verifier lane | One shader owner batches kernel additions; Zig lanes add wrappers only after kernel contract exists. |
| `engine/src/tensor/*`, layout contracts | tensor/layout lane | Metal runtime lane | Dtype/layout changes require threshold-table and test updates. |
| future `engine/src/network*`, `engine/src/models/mnist.zig` | network/MNIST lane | test/verifier lane | Training/inference changes require CPU/reference golden evidence. |
| `engine/src/runtime/*assets.zig`, future `safetensors`/`model` modules | model-loader lane | tokenizer lane | Model loader owns tensor metadata; tokenizer pairing semantics go through tokenizer owner. |
| `engine/src/tokenizer.zig` and tokenizer fixtures | tokenizer lane | model-loader lane | Token ID changes block transformer work until parity evidence is updated. |
| future transformer modules / `engine/src/models/decoder.zig` | transformer lane | model-loader + tokenizer lanes | Transformer consumes loader/tokenizer contracts; no local redefinition of those boundaries. |
| future quantized/QMV/Q4 modules and kernels | quantization lane | Metal shader + transformer lanes | Q1/Q4 integration starts only after Milestone 4/5 contracts pass. |
| `tests/**`, fixture expected outputs | test/verifier lane | owning feature lane | Feature lanes add requested tests; verifier owns cross-milestone gates. |
| `research/scripts/**`, `python/scripts/**` benchmark/reporting | benchmark/tooling lane | verifier lane | Benchmark scripts must cite correctness gates before timing claims. |
| `docs/**`, `.omx/plans/**` | docs/planning lane | architect lane | Contract changes require reviewer sign-off and threshold update if acceptance changes. |
| `docs/nnzap-asset-manifest-template.json`, local asset manifests | asset/provenance lane | verifier lane | Final-gate asset entries cannot be marked complete without checksums and provenance notes. |

## Shared-surface process
1. Owner lane proposes change and updates affected contract/threshold docs.
2. Reviewer lane checks interface and acceptance impact.
3. Verifier reruns relevant milestone checks.
4. Cross-owner changes are staged sequentially, not merged in parallel.
