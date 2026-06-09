# nnzap parity inventory — Milestone 0

Generated for Milestone 0 of `.omx/plans/prometheus-strict/prd-nnzap-full-parity.md`. This is a planning/contract artifact only; it does not start Milestone 1 implementation.

## Scope and baseline

- Reference root: `reference/nnzap/nnmetal` (ignored by git but authoritative for this parity effort).
- Current implementation root: `engine/src`.
- Current bolt baseline: `engine/src/metal/kernels.metal` exposes 7 kernels (`copy_f32`, `add_f32`, `matmul_f32`, `bias_add_f32`, `relu_f32`, `reduce_sum_f32`, `softmax_f32`); `engine/src/metal/context.zig` and `engine/src/metal/bridge.m` wrap those kernels; `engine/src/models/mnist.zig` and `engine/src/models/decoder.zig` are fixture/proof oriented; `.safetensors` is intentionally rejected in `engine/src/runtime/llm_assets.zig`; `engine/src/tokenizer.zig` is a small fixture tokenizer.

## Reference file coverage

- Reference Zig files inventoried: 28
- Reference shader files inventoried: 4
- Reference examples inventoried: 8
- Reference tests inventoried: 8

### Module/example/test mapping

| Reference file | Kind | Lines | Current bolt status / target mapping |
| --- | --- | ---: | --- |
| `reference/nnzap/nnmetal/examples/bonsai.zig` | example | 553 | missing: no full Bonsai 1.7B CLI; target `engine/src/runtime/llm_assets.zig`, `engine/src/tokenizer.zig`, transformer runtime/shaders, and `run-bonsai`; owner `model-runtime+transformer`; blocker real Bonsai asset manifest; acceptance `real-model-assets`, `transformer-golden`, `run-bonsai` thresholds |
| `reference/nnzap/nnmetal/examples/bonsai_bench.zig` | example | 648 | missing: no real-model inference benchmark CLI; target benchmark lane plus transformer runtime; owner `benchmark+transformer`; blocker Bonsai runtime parity; acceptance `run-bonsai-bench` and benchmark threshold rows |
| `reference/nnzap/nnmetal/examples/bonsai_golden.zig` | example | 579 | missing: no Bonsai golden-token gate; target golden harness over tokenizer/model pair; owner `model-runtime+tests`; blocker real Bonsai assets and tokenizer parity; acceptance `run-bonsai-golden`, `transformer-golden`, `model-tokenizer-pairing` thresholds |
| `reference/nnzap/nnmetal/examples/bonsai_q4_bench.zig` | example | 645 | missing: no Q4 model path or Q4 benchmark CLI; target Q4 loader, specialized Metal kernels, and benchmark harness; owner `quantized-runtime+benchmark`; blocker Q4 asset manifest and specialized shader parity; acceptance `run-bonsai-q4-bench`, `q4-golden`, benchmark threshold rows |
| `reference/nnzap/nnmetal/examples/bonsai_q4_golden.zig` | example | 1528 | missing: no Q4 golden harness; target Q4 safetensors loader, tokenizer pairing, specialized kernels, and exact-token verifier; owner `quantized-runtime+tests`; blocker Q4 real assets and shader parity; acceptance `run-bonsai-q4-golden`, `q4-golden`, `model-tokenizer-pairing` thresholds |
| `reference/nnzap/nnmetal/examples/inference_bench.zig` | example | 694 | missing: current `engine/src/cli/benchmark_kernels.zig` covers primitive kernels only; target end-to-end inference benchmark over full model/runtime; owner `benchmark+model-runtime`; blocker transformer/model loader parity; acceptance `run-infer` and benchmark threshold rows |
| `reference/nnzap/nnmetal/examples/mnist.zig` | example | 1042 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `reference/nnzap/nnmetal/examples/mnist_1bit.zig` | example | 803 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `reference/nnzap/nnmetal/src/benchmark.zig` | module | 469 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `reference/nnzap/nnmetal/src/benchmark_test.zig` | test | 245 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `reference/nnzap/nnmetal/src/layout.zig` | module | 354 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `reference/nnzap/nnmetal/src/layout_test.zig` | test | 288 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `reference/nnzap/nnmetal/src/metal.zig` | module | 1713 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `reference/nnzap/nnmetal/src/metal_test.zig` | test | 66 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `reference/nnzap/nnmetal/src/mnist.zig` | module | 270 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `reference/nnzap/nnmetal/src/mnist_test.zig` | test | 146 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `reference/nnzap/nnmetal/src/model.zig` | module | 1266 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `reference/nnzap/nnmetal/src/model_test.zig` | test | 315 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `reference/nnzap/nnmetal/src/network.zig` | module | 3308 | missing: no training/backprop Network abstraction |
| `reference/nnzap/nnmetal/src/root.zig` | module | 57 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `reference/nnzap/nnmetal/src/safetensors.zig` | module | 582 | missing: current `.safetensors` path is intentionally unsupported |
| `reference/nnzap/nnmetal/src/safetensors_test.zig` | test | 159 | missing: current `.safetensors` path is intentionally unsupported |
| `reference/nnzap/nnmetal/src/shaders/compute.metal` | shader | 4675 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `reference/nnzap/nnmetal/src/shaders/q4mv_bf16_specialized.metal` | shader | 1181 | missing: no QMV/Q4 specialized shader path |
| `reference/nnzap/nnmetal/src/shaders/qmv_specialized.metal` | shader | 1621 | missing: no QMV/Q4 specialized shader path |
| `reference/nnzap/nnmetal/src/shaders/transformer.metal` | shader | 1764 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `reference/nnzap/nnmetal/src/specialized_q4mv.zig` | module | 154 | missing: no specializer modules |
| `reference/nnzap/nnmetal/src/specialized_qmv.zig` | module | 160 | missing: no specializer modules |
| `reference/nnzap/nnmetal/src/tokenizer.zig` | module | 1139 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `reference/nnzap/nnmetal/src/tokenizer_test.zig` | test | 443 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `reference/nnzap/nnmetal/src/transformer.zig` | module | 4195 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `reference/nnzap/nnmetal/src/transformer_test.zig` | test | 3280 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |

## Reference build steps and current mapping

| Reference build evidence | Current bolt mapping | Milestone 0 status |
| --- | --- | --- |
| `reference/nnzap/nnmetal/build.zig:26` `const mnist_example = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:39` `const run_step = b.step("run", "Run the MNIST example");` | `run_mnist_fixture` exists for inference fixture only; future `run` equivalent must cover MNIST training | inventoried |
| `reference/nnzap/nnmetal/build.zig:48` `const mnist_1bit = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:61` `const run_1bit_step = b.step("run-1bit", "Run the 1-bit MNIST integration test");` | missing; future packed/1-bit path | inventoried |
| `reference/nnzap/nnmetal/build.zig:70` `const infer_bench = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:83` `const run_infer_step = b.step("run-infer", "Run the inference benchmark");` | `benchmark_kernels` exists for primitive benchmark only; future inference benchmark required | inventoried |
| `reference/nnzap/nnmetal/build.zig:92` `const bonsai = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:105` `const run_bonsai_step = b.step("run-bonsai", "Run the Bonsai 1.7B inference CLI");` | `run_llm_fixture` exists for tiny decoder proof only; future real Bonsai path required | inventoried |
| `reference/nnzap/nnmetal/build.zig:114` `const bonsai_bench = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:127` `const run_bonsai_bench_step = b.step(` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:139` `const bonsai_q4_bench = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:152` `const run_bonsai_q4_bench_step = b.step(` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:164` `const bonsai_golden = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:177` `const run_bonsai_golden_step = b.step(` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:189` `const bonsai_q4_golden = b.addExecutable(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:202` `const run_bonsai_q4_golden_step = b.step(` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:214` `const mod_tests = b.addTest(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:219` `const example_tests = b.addTest(.{` | build graph evidence; see module/example mapping | inventoried |
| `reference/nnzap/nnmetal/build.zig:224` `const test_step = b.step("test", "Run tests");` | `cd engine && zig build test` exists for current slice; future parity test suite required | inventoried |

## Shader kernel inventory

### `reference/nnzap/nnmetal/src/shaders/compute.metal` (60 kernels)

| Kernel | Line | Current bolt status |
| --- | ---: | --- |
| `vector_add` | 9 | present with different name/scope |
| `matmul` | 26 | partial: related primitive exists in `engine/src/metal/kernels.metal`, but reference semantics/shape variants still need parity tests |
| `matmul_tiled` | 47 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias` | 115 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias_relu` | 174 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias_relu_infer` | 238 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias_packed` | 306 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias_relu_infer_packed` | 359 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias_v2` | 422 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_bias_relu_v2` | 493 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_transA` | 573 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `matmul_transB` | 631 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `relu_forward` | 689 | partial: related primitive exists in `engine/src/metal/kernels.metal`, but reference semantics/shape variants still need parity tests |
| `tanh_forward` | 698 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `sigmoid_forward` | 707 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `relu_backward` | 720 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `tanh_backward` | 730 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `sigmoid_backward` | 741 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `bias_add` | 758 | partial: related primitive exists in `engine/src/metal/kernels.metal`, but reference semantics/shape variants still need parity tests |
| `bias_grad` | 777 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `sgd_update` | 798 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `bias_grad_sgd` | 817 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `weight_grad_sgd` | 840 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `mse_forward` | 901 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `mse_backward` | 912 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `softmax_forward` | 931 | partial: related primitive exists in `engine/src/metal/kernels.metal`, but reference semantics/shape variants still need parity tests |
| `ce_forward` | 962 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `softmax_ce_backward` | 981 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `argmax_predictions` | 1028 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `forward_fused_infer_3layer` | 1087 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `forward_fused_infer_3layer_v2` | 1167 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `forward_fused_infer_3layer_v3` | 1245 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `forward_fused_infer_batched` | 1332 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `f32_to_f16` | 1410 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `forward_fused_infer_batched_f16` | 1430 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `forward_fused_infer_single_f16` | 1506 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `adam_update` | 1579 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `f32_to_1bit` | 1627 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv` | 1713 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fast` | 1813 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fused_pair` | 1931 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fused_pair_sm` | 2027 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fast_sm` | 2128 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fast_sm2` | 2225 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const` | 2345 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fused_pair_const` | 2492 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fast_multigroup` | 2694 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_multigroup` | 2823 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmm` | 2975 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_f16in` | 3105 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_f16in` | 3200 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fused_pair_const_f16in` | 3351 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_multigroup_f16in` | 3565 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_f16io` | 3712 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_f16io_resadd` | 3806 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_f16io` | 3899 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_f16io_resadd` | 4049 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_fused_pair_const_f16io` | 4200 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_multigroup_f16io` | 4414 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |
| `qmv_const_multigroup_f16io_resadd` | 4550 | partial: current bolt has small combined shader `engine/src/metal/kernels.metal` with 7 primitive kernels; broad compute kernels missing |

### `reference/nnzap/nnmetal/src/shaders/q4mv_bf16_specialized.metal` (9 kernels)

| Kernel | Line | Current bolt status |
| --- | ---: | --- |
| `q4mv_spec_f16io` | 93 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_f16io_resadd` | 185 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_fused_pair_f16io` | 305 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_f16in` | 394 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_mg_f16io_resadd` | 514 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_fused_pair_silu_f16io` | 637 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_fused_norm_pair_silu_f16io` | 726 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_fused_norm_f16io` | 880 | missing: no QMV/Q4 specialized shader path |
| `q4mv_spec_fused_norm_pair_f16io` | 1040 | missing: no QMV/Q4 specialized shader path |

### `reference/nnzap/nnmetal/src/shaders/qmv_specialized.metal` (9 kernels)

| Kernel | Line | Current bolt status |
| --- | ---: | --- |
| `qmv_spec_f16io` | 55 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_f16io_resadd` | 217 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_fused_pair_f16io` | 377 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_f16in` | 597 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_mg_f16io_resadd` | 760 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_fused_pair_silu_f16io` | 955 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_fused_norm_pair_silu_f16io` | 1096 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_fused_norm_f16io` | 1290 | missing: no QMV/Q4 specialized shader path |
| `qmv_spec_fused_norm_pair_f16io` | 1447 | missing: no QMV/Q4 specialized shader path |

### `reference/nnzap/nnmetal/src/shaders/transformer.metal` (23 kernels)

| Kernel | Line | Current bolt status |
| --- | ---: | --- |
| `rms_norm` | 49 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `silu` | 106 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `silu_elementwise_mul` | 128 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `rope` | 170 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `kv_cache_update` | 226 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `gqa_attention` | 294 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `embedding_lookup` | 453 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `embedding_lookup_q4` | 505 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `rms_norm_f16out` | 561 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `rms_norm_bf16_f16out` | 624 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `residual_add` | 690 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `rms_norm_f16` | 722 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `rope_f16` | 783 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `fused_norm_rope_f16` | 840 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `fused_k_norm_rope_kv_cache_f16` | 940 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `fused_rope_k_kv_cache_update_f16` | 1060 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `kv_cache_update_f16in` | 1128 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `gqa_attention_f16io` | 1176 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `gqa_attention_f16io_tg` | 1314 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `gqa_attention_fused_f16io_tg` | 1467 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `silu_elementwise_mul_f16` | 1708 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `residual_add_f16` | 1735 | missing: no full transformer shader library; current decoder is fixture-oriented |
| `set_completion_flag` | 1756 | missing: no full transformer shader library; current decoder is fixture-oriented |

## Public API inventory (`pub` symbols)

### `reference/nnzap/nnmetal/examples/bonsai.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 59 | missing: target `run-bonsai` CLI; owner `model-runtime+transformer`; blocker real Bonsai model/tokenizer/safetensors asset tuple; acceptance `run-bonsai` + `transformer-golden` |

### `reference/nnzap/nnmetal/examples/bonsai_bench.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 38 | missing: target `run-bonsai-bench`; owner `benchmark+transformer`; blocker full Bonsai runtime; acceptance benchmark threshold row |

### `reference/nnzap/nnmetal/examples/bonsai_golden.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 58 | missing: target `run-bonsai-golden`; owner `model-runtime+tests`; blocker paired tokenizer/model assets; acceptance exact golden-output threshold |

### `reference/nnzap/nnmetal/examples/bonsai_q4_bench.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 43 | missing: target `run-bonsai-q4-bench`; owner `quantized-runtime+benchmark`; blocker Q4 loader and specialized Metal kernels; acceptance Q4 benchmark threshold row |

### `reference/nnzap/nnmetal/examples/bonsai_q4_golden.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 63 | missing: target `run-bonsai-q4-golden`; owner `quantized-runtime+tests`; blocker Q4 asset tuple and Q4 shader parity; acceptance exact Q4 golden threshold |

### `reference/nnzap/nnmetal/examples/inference_bench.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 111 | missing: target `run-infer`; owner `benchmark+model-runtime`; blocker full inference runtime; acceptance inference benchmark threshold row |

### `reference/nnzap/nnmetal/examples/mnist.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 174 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |

### `reference/nnzap/nnmetal/examples/mnist_1bit.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `main` | 142 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |

### `reference/nnzap/nnmetal/src/benchmark.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `Optimizer` | 44 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `LossFunction` | 49 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `LayerSpec` | 57 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `Config` | 64 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `EpochResult` | 82 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `TestResult` | 91 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `Benchmark` | 108 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `init` | 131 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `recordEpoch` | 168 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `recordTest` | 180 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `setTrainingTime` | 203 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `save` | 214 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `extractArchitecture` | 358 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `formatTimestamp` | 385 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `sanitiseForPath` | 411 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `computeThroughput` | 445 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |
| `nanosToMs` | 463 | partial: `engine/src/cli/benchmark_kernels.zig` and scripts exist; reference Benchmark infra missing |

### `reference/nnzap/nnmetal/src/layout.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `divCeil` | 17 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `Activation` | 27 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `LayerDesc` | 35 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `NetworkLayout` | 43 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `num_layers:` | 80 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `layers:` | 81 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `weight_counts:` | 85 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `bias_counts:` | 93 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `param_count:` | 102 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `weight_offsets:` | 124 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `bias_offsets:` | 136 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `activation_sizes:` | 148 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `max_activation_size:` | 158 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `input_size:` | 169 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `output_size:` | 172 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `PACK_GROUP_SIZE:` | 178 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `packedWeightBytes` | 183 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `packed_weight_bytes:` | 208 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `packed_weight_offsets:` | 221 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `total_packed_weight_bytes:` | 233 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `packedBitBytes` | 243 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `numScaleGroups` | 255 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `scaleByteOffset` | 268 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `getWeightSlice` | 277 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `getBiasSlice` | 293 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |
| `printSummary` | 308 | partial: `engine/src/tensor/layout.zig` exists, but no reference-style NetworkLayout contract yet |

### `reference/nnzap/nnmetal/src/metal.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `Object` | 13 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `MTLResourceOptions` | 31 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `CPUCacheMode` | 42 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `StorageMode` | 47 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `HazardTrackingMode` | 54 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `storage_shared:` | 60 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `storage_private:` | 63 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `MTLSize` | 68 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `Buffer` | 80 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `init` | 86 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `initWriteCombined` | 123 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `asSlice` | 162 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `metalBuffer` | 172 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `deinit` | 176 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `HalfBuffer` | 186 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `init` | 191 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `deinit` | 217 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `PackedBuffer` | 235 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `packedBytes` | 242 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `numGroups` | 252 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `scaleOffset` | 266 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `init` | 275 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `metalBuffer` | 327 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `deinit` | 331 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `Q4Buffer` | 355 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `nibbleBytes` | 363 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `numGroups` | 373 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `scaleOffset` | 387 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `biasOffset` | 397 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `init` | 407 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `metalBuffer` | 461 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `deinit` | 465 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `MultiBuffered` | 483 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `getCurrent` | 501 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `getCurrentConst` | 508 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `swap` | 516 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `ComputePipeline` | 529 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `init` | 536 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `Device` | 839 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `init` | 959 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `createBuffer` | 1040 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `createMultiBuffered` | 1069 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `beginCommandBuffer` | 1102 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `beginCommandBufferUnretained` | 1118 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `beginCompute` | 1132 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `dispatch1D` | 1154 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `dispatch2D` | 1207 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `dispatchCustom` | 1258 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `commitAndWait` | 1293 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `commit` | 1307 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `commitAndSpinOnFlag` | 1325 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `memoryBarrier` | 1358 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setBuffer` | 1379 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setBufferWithOffset` | 1397 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setBuffersBatchOffsets` | 1425 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setBuffersBatch` | 1470 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setBytes` | 1511 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setPackedBuffer` | 1538 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `setQ4Buffer` | 1573 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |
| `compileLibraryFromSource` | 1664 | partial: `engine/src/metal/context.zig` + `bridge.m` exist for small kernel set; broad Device/Buffer/Packed/Q4 API missing |

### `reference/nnzap/nnmetal/src/mnist.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `image_rows:` | 20 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `image_cols:` | 21 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `image_size:` | 22 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `num_classes:` | 23 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `train_count:` | 24 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `test_count:` | 25 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `Mnist` | 27 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `load` | 46 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `deinit` | 100 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `fillImageBatch` | 113 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `fillLabelBatch` | 131 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `readU32BE` | 153 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |
| `oneHot` | 232 | partial: `engine/src/models/mnist.zig` covers deterministic inference fixture; training/1-bit/data loader missing |

### `reference/nnzap/nnmetal/src/model.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `Model` | 38 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `init` | 123 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `deinit` | 169 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `loadWeights` | 196 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `forwardBlockArgs` | 247 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `forwardDecodeArgs` | 298 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `convertScalesAffineToSymmetric` | 1060 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `convertF32ToF16` | 1073 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `convertBF16ToF32` | 1086 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |
| `convertBF16ToF16` | 1109 | partial: `engine/src/runtime/llm_assets.zig` loads tiny bundles; reference Model loader missing |

### `reference/nnzap/nnmetal/src/network.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `Network` | 70 | missing: no training/backprop Network abstraction |
| `init` | 152 | missing: no training/backprop Network abstraction |
| `deinit` | 237 | missing: no training/backprop Network abstraction |
| `paramSlice` | 270 | missing: no training/backprop Network abstraction |
| `getOutput` | 280 | missing: no training/backprop Network abstraction |
| `gradSlice` | 296 | missing: no training/backprop Network abstraction |
| `forward` | 315 | missing: no training/backprop Network abstraction |
| `forwardInfer` | 413 | missing: no training/backprop Network abstraction |
| `forwardInferFused` | 540 | missing: no training/backprop Network abstraction |
| `forwardInferFusedV3` | 594 | missing: no training/backprop Network abstraction |
| `forwardInferFusedBatched` | 652 | missing: no training/backprop Network abstraction |
| `forwardInferFusedBatchedExt` | 712 | missing: no training/backprop Network abstraction |
| `f16_weight_count:` | 772 | missing: no training/backprop Network abstraction |
| `f16_bias_count:` | 783 | missing: no training/backprop Network abstraction |
| `forwardInferBatchedF16` | 799 | missing: no training/backprop Network abstraction |
| `forwardInferSingleF16` | 887 | missing: no training/backprop Network abstraction |
| `total_bias_count:` | 991 | missing: no training/backprop Network abstraction |
| `quantizeWeightsTo1Bit` | 1021 | missing: no training/backprop Network abstraction |
| `forwardInfer1Bit` | 1128 | missing: no training/backprop Network abstraction |
| `forwardCPU` | 1314 | missing: no training/backprop Network abstraction |
| `backward` | 1374 | missing: no training/backprop Network abstraction |
| `backwardSGD` | 1560 | missing: no training/backprop Network abstraction |
| `update` | 1739 | missing: no training/backprop Network abstraction |
| `updateAdam` | 1779 | missing: no training/backprop Network abstraction |
| `encodeMSELoss` | 1840 | missing: no training/backprop Network abstraction |
| `encodeMSEGrad` | 1870 | missing: no training/backprop Network abstraction |
| `encodeSoftmax` | 1913 | missing: no training/backprop Network abstraction |
| `encodeCELoss` | 1955 | missing: no training/backprop Network abstraction |
| `encodeSoftmaxCEGrad` | 2002 | missing: no training/backprop Network abstraction |
| `encodeArgmax` | 2058 | missing: no training/backprop Network abstraction |

### `reference/nnzap/nnmetal/src/root.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `metal` | 12 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `layout` | 13 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `network` | 14 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `mnist` | 15 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `benchmark` | 16 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `transformer` | 17 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `safetensors` | 18 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `tokenizer` | 19 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `model` | 20 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `specialized_qmv` | 21 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `specialized_q4mv` | 22 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Device` | 25 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Buffer` | 26 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `PackedBuffer` | 27 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Q4Buffer` | 28 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `MultiBuffered` | 29 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `NetworkLayout` | 30 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `LayerDesc` | 31 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Activation` | 32 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `divCeil` | 33 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Network` | 34 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Mnist` | 35 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Benchmark` | 36 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `QuantFormat` | 37 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `TransformerConfig` | 38 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `TransformerPipelines` | 39 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `ForwardBlockArgsT` | 40 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `ForwardDecodeArgsT` | 41 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Bonsai1_7B` | 42 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Bonsai4B` | 43 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Bonsai8B` | 44 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Bonsai1_7B_Q4` | 45 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Bonsai4B_Q4` | 46 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Bonsai8B_Q4` | 47 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `SamplingParams` | 48 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `GenerateResult` | 49 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `GenerateOpts` | 50 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `SafetensorsFile` | 51 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Tokenizer` | 52 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |
| `Model` | 53 | partial: `engine/src/root.zig` exports fixture/runtime modules, not reference public API surface |

### `reference/nnzap/nnmetal/src/safetensors.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `Dtype` | 46 | missing: current `.safetensors` path is intentionally unsupported |
| `sizeBytes` | 54 | missing: current `.safetensors` path is intentionally unsupported |
| `fromString` | 65 | missing: current `.safetensors` path is intentionally unsupported |
| `TensorDescriptor` | 87 | missing: current `.safetensors` path is intentionally unsupported |
| `elementCount` | 99 | missing: current `.safetensors` path is intentionally unsupported |
| `sizeBytes` | 110 | missing: current `.safetensors` path is intentionally unsupported |
| `SafetensorsFile` | 119 | missing: current `.safetensors` path is intentionally unsupported |
| `init` | 134 | missing: current `.safetensors` path is intentionally unsupported |
| `initFromBytes` | 167 | missing: current `.safetensors` path is intentionally unsupported |
| `deinit` | 183 | missing: current `.safetensors` path is intentionally unsupported |
| `getTensor` | 196 | missing: current `.safetensors` path is intentionally unsupported |

### `reference/nnzap/nnmetal/src/specialized_q4mv.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `shaderSource` | 18 | missing: no specializer modules |
| `initOnDevice` | 33 | missing: no specializer modules |

### `reference/nnzap/nnmetal/src/specialized_qmv.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `shaderSource` | 19 | missing: no specializer modules |
| `initOnDevice` | 40 | missing: no specializer modules |

### `reference/nnzap/nnmetal/src/tokenizer.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `byte_map` | 110 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `Tokenizer` | 125 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `init` | 164 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `initFromJson` | 195 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `deinit` | 232 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `encode` | 247 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `decode` | 293 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `applyChatTemplate` | 330 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `bpeMerge` | 761 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `nextPretokenChunk` | 936 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `decodeTokenToBytes` | 1066 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |
| `decodeUtf8Codepoint` | 1124 | partial: `engine/src/tokenizer.zig` is minimal fixture tokenizer; HF/BPE parity missing |

### `reference/nnzap/nnmetal/src/transformer.zig`

| Symbol | Line | Current bolt status |
| --- | ---: | --- |
| `QuantFormat` | 35 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `TransformerDesc` | 62 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `TransformerConfig` | 91 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `vocab_size:` | 96 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `hidden_size:` | 97 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `intermediate_size:` | 98 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `num_layers:` | 99 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `num_query_heads:` | 100 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `num_kv_heads:` | 101 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `head_dim:` | 102 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `max_context_length:` | 103 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `max_prefill_length:` | 105 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `rope_theta:` | 107 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `tie_word_embeddings:` | 108 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `quant_format:` | 110 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `group_size:` | 112 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `WeightBuffer` | 117 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `query_dim:` | 124 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `kv_dim:` | 126 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `heads_per_kv_group:` | 127 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `q_proj_bytes:` | 189 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `k_proj_bytes:` | 191 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `v_proj_bytes:` | 193 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `o_proj_bytes:` | 195 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `gate_proj_bytes:` | 197 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `up_proj_bytes:` | 199 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `down_proj_bytes:` | 201 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `attention_weight_bytes:` | 204 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `mlp_weight_bytes:` | 206 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `layer_weight_bytes:` | 208 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `total_layer_weight_bytes:` | 210 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `embedding_bytes:` | 214 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `lm_head_bytes:` | 216 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `norm_scales_per_layer:` | 223 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `total_norm_scale_count:` | 225 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `total_weight_bytes:` | 230 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `kv_cache_elements_per_layer:` | 236 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `total_kv_cache_elements:` | 238 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `total_kv_cache_bytes:` | 240 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `decode_activation_elements:` | 246 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `prefill_activation_elements:` | 258 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `max_activation_elements:` | 270 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `Bonsai1_7B` | 286 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `Bonsai4B` | 300 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `Bonsai8B` | 314 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `Bonsai1_7B_Q4` | 336 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `Bonsai4B_Q4` | 352 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `Bonsai8B_Q4` | 368 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `RMSNormDims` | 389 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `RoPEDims` | 396 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `KVUpdateDims` | 404 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `FusedRoPEKVDims` | 412 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `FusedNormRoPEDims` | 421 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `FusedKNormRoPEKVDims` | 430 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `GQADims` | 440 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `GQAFusedDims` | 451 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `EmbedDims` | 464 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `QMVDims` | 475 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `TransformerPipelines` | 489 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `init` | 517 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchRMSNorm` | 669 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchSiLU` | 700 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchSiLUElementwiseMul` | 719 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchRoPE` | 740 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchKVCacheUpdate` | 760 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchGQAAttention` | 1030 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchEmbeddingLookup` | 1072 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchResidualAdd` | 1097 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `dispatchQMV` | 1565 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `ForwardBlockArgsT` | 2397 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `ForwardBlockArgs` | 2437 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `forwardBlock` | 2452 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `ForwardDecodeArgsT` | 2495 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `ForwardDecodeArgs` | 2550 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `forwardDecode` | 2566 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `SamplingParams` | 3655 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `GenerateResult` | 3663 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `GenerateOpts` | 3672 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `argmax` | 3694 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `applyTemperature` | 3712 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `softmaxInPlace` | 3727 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `applyTopK` | 3750 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `applyTopP` | 3781 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `sampleToken` | 3853 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `generate` | 3942 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |
| `isEosToken` | 4126 | partial: `engine/src/models/decoder.zig` is tiny proof decoder; full transformer missing |


## Normalized required build-step checklist

This checklist resolves multi-line `b.step(...)` declarations from `reference/nnzap/nnmetal/build.zig` into exact command names that future bolt build aliases must map.

| Reference step | Reference executable/example | Current bolt status |
| --- | --- | --- |
| `run` | `examples/mnist.zig` / `mnist` | partial: current `run_mnist_fixture` is inference fixture only; MNIST training command missing |
| `run-1bit` | `examples/mnist_1bit.zig` / `mnist_1bit` | missing: 1-bit MNIST command and packed path missing |
| `run-infer` | `examples/inference_bench.zig` / `inference_bench` | partial: primitive benchmark exists; reference inference benchmark missing |
| `run-bonsai` | `examples/bonsai.zig` / `bonsai` | partial: tiny `run_llm_fixture`; real Bonsai command missing |
| `run-bonsai-bench` | `examples/bonsai_bench.zig` / `bonsai_bench` | missing: real Bonsai benchmark missing |
| `run-bonsai-q4-bench` | `examples/bonsai_q4_bench.zig` / `bonsai_q4_bench` | missing: Q4 benchmark missing |
| `run-bonsai-golden` | `examples/bonsai_golden.zig` / `bonsai_golden` | missing: unquantized Bonsai golden missing |
| `run-bonsai-q4-golden` | `examples/bonsai_q4_golden.zig` / `bonsai_q4_golden` | missing: Q4 golden missing |
| `test` | module + example tests | partial: current `zig build test` covers current slice only; parity tests missing |
