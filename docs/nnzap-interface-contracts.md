# nnzap Milestone 0 interface contracts

These contracts freeze the boundaries required before Milestone 1 coding. They are intentionally executable enough for ownership and test planning, while leaving implementation details to later milestones.

## 1. Tensor layout contract
- Shape order must be documented per operation before kernels are added. Default row-major tensors use trailing dimension as contiguous unless a milestone-specific table says otherwise.
- Dtype tags required before final parity: `f32`, `f16`, `bf16`, `q1`, `q4`.
- Packed layouts (`q1`, `q4`) must define bits-per-value, scale/bias storage, group size, alignment, and row padding before any QMV kernel is added.
- Any tensor view exposed to Metal must have explicit element count and byte count checks.

## 2. Metal buffer ownership contract
- Metal shared buffers are owned by runtime buffer wrappers; CPU slices are views and must not be cached across GPU submission.
- Invalid buffer handles must produce recoverable errors at public boundaries; panics are reserved for impossible internal invariants.
- Dispatch helpers must check buffer sizes, dtype/layout expectations, and thread/grid limits before encoding.

## 3. Bridge/library contract
- Every reference kernel maps to either a bolt shader function or an explicitly documented deferred blocker in the threshold table.
- Pipeline names are stable strings; tests must catch missing pipeline names.
- Runtime shader compilation errors must preserve the Metal compiler message in diagnostics.
- Precompiled `.metallib`/binary archive adoption is optional and must not replace source-level test coverage.

## 4. Network contract
- `NetworkLayout` equivalent must define layer descriptors, activation enum, parameter offsets, gradient offsets, activation buffer sizes, and static allocation bounds.
- Training paths must separate forward, backward, loss, and optimizer/update evidence.
- MNIST training final gate uses real MNIST assets and the threshold table; synthetic mini-network tests are milestone-only.

## 5. Model-loading contract
- Safetensors parsing owns header, dtype, shape, offset, and tensor metadata validation.
- Milestone 4 owns f32/f16/bf16 tensor metadata/load paths only. Q1/Q4 decode integration belongs to Milestone 7.
- Model config parsing must not claim tokenizer semantic parity; tokenizer/model pairing validation belongs to Milestone 5.
- Unsupported or malformed formats must fail with named deterministic errors.

## 6. Tokenizer contract
- Tokenizer parity requires exact token IDs for fixed reference prompts.
- Supported JSON fields, BPE/merge behavior, byte-unicode mapping, special token handling, and chat-template construction must be tested before transformer golden gates.
- Tokenizer/model pairing validation must record which tokenizer files belong to each model asset manifest entry.

## 7. Transformer contract
- Transformer milestones consume loader/tokenizer contracts and must not redefine them.
- Required execution surfaces include embedding lookup, RMSNorm, RoPE, KV cache update, GQA attention, residual add, SiLU/MLP, forward block, decode, sampling, and generation report schema.
- Golden route token IDs must match reference for fixed prompts unless a documented deviation amends the PRD/test spec.

## 8. Benchmark/golden contract
- Correctness gates precede benchmark claims.
- Benchmark artifacts must include metric fields required by the mapped reference command plus `correctness_gates[]` evidence paths.
- Timing values are observations unless a separate performance goal adds numeric targets.
- Any user-approved deviation must amend PRD, test spec, parity matrix, acceptance criteria, and rollback rules before closure.
