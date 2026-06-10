const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("bolt", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    mod.addCSourceFile(.{
        .file = b.path("src/metal/bridge.m"),
        .flags = &.{"-fobjc-arc"},
    });
    mod.linkFramework("Metal", .{});
    mod.linkFramework("Foundation", .{});
    mod.linkFramework("Accelerate", .{});
    mod.linkSystemLibrary("objc", .{});
    mod.link_libc = true;

    const fixture_inspect = b.addExecutable(.{
        .name = "fixture_inspect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/fixture_inspect.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(fixture_inspect);

    const dump_llm_runtime = b.addExecutable(.{
        .name = "dump_llm_runtime",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/dump_llm_runtime.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(dump_llm_runtime);

    const trace_llm_fixture = b.addExecutable(.{
        .name = "trace_llm_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/trace_llm_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(trace_llm_fixture);

    const decode_llm_fixture = b.addExecutable(.{
        .name = "decode_llm_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/decode_llm_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(decode_llm_fixture);

    const proof_fixture = b.addExecutable(.{
        .name = "proof_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/proof_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(proof_fixture);

    const run_mnist_fixture = b.addExecutable(.{
        .name = "run_mnist_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/run_mnist_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(run_mnist_fixture);

    const run_llm_fixture = b.addExecutable(.{
        .name = "run_llm_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/run_llm_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(run_llm_fixture);

    const check_mnist_fixture = b.addExecutable(.{
        .name = "check_mnist_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/check_mnist_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(check_mnist_fixture);

    const check_llm_fixture = b.addExecutable(.{
        .name = "check_llm_fixture",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/check_llm_fixture.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(check_llm_fixture);

    const train_mini_network = b.addExecutable(.{
        .name = "train_mini_network",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/train_mini_network.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(train_mini_network);

    const train_mnist = b.addExecutable(.{
        .name = "train_mnist",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/train_mnist.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(train_mnist);

    const mnist_1bit = b.addExecutable(.{
        .name = "mnist_1bit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/mnist_1bit.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(mnist_1bit);

    const inference_bench = b.addExecutable(.{
        .name = "inference_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/inference_bench.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(inference_bench);

    const validate_assets = b.addExecutable(.{
        .name = "validate_assets",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/validate_assets.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(validate_assets);

    const validate_tokenizer = b.addExecutable(.{
        .name = "validate_tokenizer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/validate_tokenizer.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(validate_tokenizer);

    const bonsai = b.addExecutable(.{
        .name = "bonsai",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/bonsai.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai);

    const bonsai_golden = b.addExecutable(.{
        .name = "bonsai_golden",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/bonsai_golden.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_golden);

    const bonsai_bench = b.addExecutable(.{
        .name = "bonsai_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/bonsai_bench.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_bench);

    const bonsai_q4_golden = b.addExecutable(.{
        .name = "bonsai_q4_golden",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/bonsai_q4_golden.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_q4_golden);

    const bonsai_q4_bench = b.addExecutable(.{
        .name = "bonsai_q4_bench",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/bonsai_q4_bench.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(bonsai_q4_bench);

    const benchmark_kernels = b.addExecutable(.{
        .name = "benchmark_kernels",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/benchmark_kernels.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    b.installArtifact(benchmark_kernels);

    const run_step = b.step("run", "Run the MNIST training/example equivalent");
    const run_cmd = b.addRunArtifact(train_mnist);
    run_step.dependOn(&run_cmd.step);
    if (b.args) |args| run_cmd.addArgs(args);

    const run_1bit_step = b.step("run-1bit", "Run the 1-bit MNIST integration equivalent");
    const run_1bit_cmd = b.addRunArtifact(mnist_1bit);
    run_1bit_step.dependOn(&run_1bit_cmd.step);
    if (b.args) |args| run_1bit_cmd.addArgs(args);

    const run_infer_step = b.step("run-infer", "Run the MNIST inference benchmark equivalent");
    const run_infer_cmd = b.addRunArtifact(inference_bench);
    run_infer_step.dependOn(&run_infer_cmd.step);
    if (b.args) |args| run_infer_cmd.addArgs(args);

    const validate_assets_step = b.step("validate-assets", "Validate Bolt real asset manifest");
    const validate_assets_cmd = b.addRunArtifact(validate_assets);
    validate_assets_step.dependOn(&validate_assets_cmd.step);
    if (b.args) |args| validate_assets_cmd.addArgs(args);

    const validate_tokenizer_step = b.step("validate-tokenizer", "Validate Bolt real tokenizer and model pairing");
    const validate_tokenizer_cmd = b.addRunArtifact(validate_tokenizer);
    validate_tokenizer_step.dependOn(&validate_tokenizer_cmd.step);
    if (b.args) |args| validate_tokenizer_cmd.addArgs(args);

    const run_bonsai_step = b.step("run-bonsai", "Run the Bonsai 1.7B inference CLI/gate");
    const run_bonsai_cmd = b.addRunArtifact(bonsai);
    run_bonsai_step.dependOn(&run_bonsai_cmd.step);
    if (b.args) |args| run_bonsai_cmd.addArgs(args);

    const run_bonsai_golden_step = b.step("run-bonsai-golden", "Run the Bonsai 1.7B golden output gate");
    const run_bonsai_golden_cmd = b.addRunArtifact(bonsai_golden);
    run_bonsai_golden_step.dependOn(&run_bonsai_golden_cmd.step);
    if (b.args) |args| run_bonsai_golden_cmd.addArgs(args);

    const run_bonsai_bench_step = b.step("run-bonsai-bench", "Run the Bonsai 1.7B correctness-gated benchmark");
    const run_bonsai_bench_cmd = b.addRunArtifact(bonsai_bench);
    run_bonsai_bench_step.dependOn(&run_bonsai_bench_cmd.step);
    if (b.args) |args| run_bonsai_bench_cmd.addArgs(args);

    const run_bonsai_q4_golden_step = b.step("run-bonsai-q4-golden", "Run the Qwen3/Bonsai Q4 golden output gate");
    const run_bonsai_q4_golden_cmd = b.addRunArtifact(bonsai_q4_golden);
    run_bonsai_q4_golden_step.dependOn(&run_bonsai_q4_golden_cmd.step);
    if (b.args) |args| run_bonsai_q4_golden_cmd.addArgs(args);

    const run_bonsai_q4_bench_step = b.step("run-bonsai-q4-bench", "Run the Qwen3/Bonsai Q4 correctness-gated benchmark");
    const run_bonsai_q4_bench_cmd = b.addRunArtifact(bonsai_q4_bench);
    run_bonsai_q4_bench_step.dependOn(&run_bonsai_q4_bench_cmd.step);
    if (b.args) |args| run_bonsai_q4_bench_cmd.addArgs(args);

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const asset_path_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli/asset_paths.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    const run_asset_path_tests = b.addRunArtifact(asset_path_tests);

    const compile_active_primitives = b.addSystemCommand(&.{
        "sh",
        "-c",
        "mkdir -p .zig-cache/metal && xcrun -sdk macosx metal -c src/metal/kernels.metal -o .zig-cache/metal/active_primitives.air",
    });
    const compile_compute_shaders = b.addSystemCommand(&.{
        "sh",
        "-c",
        "mkdir -p .zig-cache/metal && xcrun -sdk macosx metal -c src/metal/shaders/compute.metal -o .zig-cache/metal/compute.air",
    });
    const compile_transformer_shaders = b.addSystemCommand(&.{
        "sh",
        "-c",
        "mkdir -p .zig-cache/metal && xcrun -sdk macosx metal -c src/metal/shaders/transformer.metal -o .zig-cache/metal/transformer.air",
    });
    const compile_qmv_shaders = b.addSystemCommand(&.{
        "sh",
        "-c",
        "mkdir -p .zig-cache/metal && xcrun -sdk macosx metal -DSPEC_HIDDEN_K=512 -DSPEC_INTER_K=1024 -DSPEC_GS=32 -c src/metal/shaders/qmv_specialized.metal -o .zig-cache/metal/qmv_specialized.air",
    });
    const compile_q4mv_shaders = b.addSystemCommand(&.{
        "sh",
        "-c",
        "mkdir -p .zig-cache/metal && xcrun -sdk macosx metal -DSPEC_HIDDEN_K=512 -DSPEC_INTER_K=1024 -DSPEC_GS=32 -c src/metal/shaders/q4mv_bf16_specialized.metal -o .zig-cache/metal/q4mv_bf16_specialized.air",
    });

    const shader_compile_step = b.step("test-metal-shaders", "Compile Metal shader groups for Bolt engine");
    shader_compile_step.dependOn(&compile_active_primitives.step);
    shader_compile_step.dependOn(&compile_compute_shaders.step);
    shader_compile_step.dependOn(&compile_transformer_shaders.step);
    shader_compile_step.dependOn(&compile_qmv_shaders.step);
    shader_compile_step.dependOn(&compile_q4mv_shaders.step);

    const test_step = b.step("test", "Run engine module tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_asset_path_tests.step);
}
