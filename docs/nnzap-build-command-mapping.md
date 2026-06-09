# nnzap Build / Example / Test Parity Mapping

Milestone 8 maps the authorized `reference/nnzap/nnmetal` build surface to executable Bolt commands and artifacts. Deviations are diagnostic only; they are not acceptance passes unless the PRD/test spec are amended.

| Reference step | Bolt command | Primary artifact/evidence |
| --- | --- | --- |
| `run` | `cd engine && zig build run` | `artifacts/nnzap-milestone3-run.json` / MNIST real train+eval output |
| `run-1bit` | `cd engine && zig build run-1bit` | `artifacts/nnzap-milestone3-run-1bit.json` |
| `run-infer` | `cd engine && zig build run-infer` | `artifacts/nnzap-milestone3-run-infer.json` |
| `run-bonsai` | `cd engine && zig build run-bonsai` | smoke-only Bonsai asset/tensor contract gate; writes `artifacts/nnzap-milestone6-bonsai-smoke.json` |
| `run-bonsai-golden` | `cd engine && zig build run-bonsai-golden` | `artifacts/nnzap-milestone6-bonsai-readiness.json` with `status: pass` and exact generated token parity |
| `run-bonsai-bench` | `cd engine && zig build run-bonsai-bench` | `artifacts/nnzap-milestone8-bonsai-bench.json`, correctness-gated on Bonsai golden and reruns a real decode benchmark workload |
| `run-bonsai-q4-golden` | `cd engine && zig build run-bonsai-q4-golden` | `artifacts/nnzap-milestone7-q4-golden.json`, exact 20/20 token parity on `qwen3-1.7b-q4-gs64` using integrated Metal Q4MV decode |
| `run-bonsai-q4-bench` | `cd engine && zig build run-bonsai-q4-bench` | `artifacts/nnzap-milestone7-q4-bench.json`, correctness-gated on Q4 golden and reruns a real integrated Metal Q4 decode benchmark workload |
| `test` | `cd engine && zig build test` | Zig unit/integration tests including Metal layout, tokenizer, safetensors, transformer, and Q4 fixtures |
| real asset validation | `cd engine && zig build validate-assets` | final-gate manifest report requiring MNIST, Bonsai, and `qwen3-1.7b-q4-gs64` checksums |
| tokenizer validation | `cd engine && zig build validate-tokenizer` | tokenizer/config pairing for Bonsai and reference-compatible Q4 assets |

Research-script closure commands:

```bash
./research/scripts/run_check.sh
./research/scripts/run_smoke.sh
./research/scripts/run_compare.sh
./research/scripts/run_proof.sh
./research/scripts/run_bench.sh
./research/scripts/run_debug.sh
```

The scripts now include real asset, tokenizer, golden, benchmark, compare/proof/debug, or schema gates according to their lane. Benchmark artifacts are only consumable after their golden correctness gate passes.
