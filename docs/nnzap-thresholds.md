# nnzap executable threshold table — Milestone 0

Every future milestone must use one of these gate types: numeric threshold, exact equality, reference-relative threshold, or explicit unresolved blocker. Qualitative gates do not close milestones.

| Reference parity row | Gate type | Executable pass/fail threshold | Evidence command/artifact | Owner |
| --- | --- | --- | --- | --- |
| Public API/build surface | exact field/API mapping | Every `reference/nnzap/nnmetal/src/root.zig` export is present as a bolt export, documented alias, or explicit blocker; every reference build step has a bolt command mapping. | `docs/nnzap-parity-inventory.md`; future build tests | architect/API lane |
| Metal kernel inventory | exact mapping | Every reference shader kernel is mapped to bolt shader function, covered by CPU-vs-Metal test, or marked unresolved blocker before coding dependent layers. | kernel mapping table + `cd engine && zig build test` | Metal shader lane |
| f32 primitive correctness | numeric | `abs <= 1e-5` for elementwise/matmul outputs unless a kernel-specific stricter CPU reference is declared. | unit kernel tests | Metal/test lanes |
| reductions/softmax | numeric | `abs <= 1e-5` for probabilities/sums with shape-specific tolerance table; row sums within `1e-5` of 1.0. | unit kernel tests | Metal/test lanes |
| f16/bf16 conversion | numeric/reference-relative | Conversion matches reference helper within documented ULP/absolute tolerance before use in model paths. | conversion tests | model-loader lane |
| MNIST training | reference-relative numeric | Same seed/data split as reference; final validation/test accuracy must be at least reference run minus 1 percentage point, and schema fields must exactly match mapped output contract. | real MNIST training artifact | network/MNIST lane |
| MNIST inference | numeric + exact schema | Predicted label/probability golden passes existing or updated expected output; output schema fields exact. | MNIST golden/inference artifact | network/MNIST lane |
| MNIST 1-bit | numeric + exact schema | Packed/1-bit output passes declared quantized tolerance and emits exact required schema; selected labels must match reference for fixed samples unless PRD/test spec amended. | 1-bit artifact | quantization lane |
| Safetensors parser | exact | Valid headers/tensors/dtypes/shapes/offsets exactly match reference metadata; malformed files fail with named errors. | parser tests + asset load report | model-loader lane |
| Tokenizer parity | exact | Fixed prompts produce identical token ID sequences and decode strings to reference. | tokenizer parity tests | tokenizer lane |
| Model/tokenizer pairing | exact | Asset manifest model entry references tokenizer files whose special tokens and prompt template match reference expected IDs. | asset manifest validation | tokenizer + model-loader lanes |
| Unquantized transformer golden | exact token + numeric logits | Greedy selected token IDs match reference for golden prompts; sampled/top-k logits meet declared tolerance from trace fixtures. | Bonsai golden artifact | transformer lane |
| Q4/Q1 golden | exact token preferred + quantized numeric | Selected token IDs match reference for golden prompts; quantized scores meet declared per-path tolerance before benchmark claims. | Q4/Q1 golden artifacts | quantization lane |
| Benchmarks | exact schema, no speed target | Benchmark JSON includes every mapped reference metric field plus `correctness_gates[]`; speed is recorded but not pass/fail unless a later performance goal adds targets. | benchmark JSON + validator | benchmark/tooling lane |
| Asset manifest final gate | exact completeness | Required real assets have path, source/provenance, license/access note, checksums, paired files, and acceptance use; no required `TODO` remains. | completed manifest + checksum report | asset/provenance lane |

## Blocking rule
If a row lacks a threshold at implementation time, the owning lane must stop and update this table before coding or claiming milestone progress.
