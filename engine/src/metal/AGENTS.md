# METAL KNOWLEDGE BASE

## OVERVIEW

`engine/src/metal/` owns the Zig-to-Objective-C Metal boundary, shared-buffer
lifecycle, shader catalog, and compile-checked kernel libraries.

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Zig API | `context.zig` | `Context`, buffer owners, dispatch wrappers |
| Objective-C bridge | `bridge.m` | Metal objects, command buffers, ABI errors |
| Active primitive kernels | `kernels.metal` | Embedded by `context.zig` for runtime use |
| Kernel inventory | `shader_catalog.zig` | Group counts and specialization requirements |
| Reference compute kernels | `shaders/compute.metal` | Large parity/reference group |
| Transformer kernels | `shaders/transformer.metal` | Attention/MLP-oriented kernels |
| QMV specialized kernels | `shaders/qmv_specialized.metal` | Requires specialization macros |
| Q4 BF16 kernels | `shaders/q4mv_bf16_specialized.metal` | BF16-faithful Q4 assumptions |

## CONVENTIONS

- `SharedBufferF32`, `SharedBufferBytes`, `HalfBuffer`, `PackedBuffer`, and
  `Q4Buffer` own Metal handles; callers must `deinit()` them.
- `deinit()` is intentionally idempotent: it destroys only when `handle != null`
  and then clears metadata.
- Use `slice()` in fallible code paths. `asSlice()` is a panic bridge for cases
  where unavailable CPU-visible contents are unrecoverable.
- Keep validation on both sides of the ABI: Zig wrappers check shape/length and
  `bridge.m` repeats buffer, dimension, group-size, and `u32` limits.
- `Context.isAvailable()` guards Metal-dependent tests; skip on non-Metal hosts
  rather than making portable tests fail.
- Specialized shader files must compile with `SPEC_HIDDEN_K=512`,
  `SPEC_INTER_K=1024`, and `SPEC_GS=32` unless the catalog/build graph changes.

## ANTI-PATTERNS

- Do not read from shared buffers before the command buffer has completed.
- Do not add kernels without updating `shader_catalog.zig` and the
  `test-metal-shaders` build step when applicable.
- Do not rely on shader-side comments such as “always aligned” without a host
  precondition in `context.zig` or `bridge.m`.
- Do not weaken Q4 BF16 behavior: scales/biases are raw BF16 bit patterns and Q4
  output writes must remain BF16-rounded where the current path requires it.
- Do not collapse `error_out` diagnostics into silent bool failures.

## COMMANDS

```sh
cd engine && zig build test-metal-shaders --summary all
cd engine && zig build test --summary all
cd engine && zig build run-metal-mlp --summary all
```
