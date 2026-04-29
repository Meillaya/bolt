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

    const mod_tests = b.addTest(.{
        .root_module = mod,
    });
    const run_mod_tests = b.addRunArtifact(mod_tests);

    const layout_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/unit/layout_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_layout_unit_tests = b.addRunArtifact(layout_unit_tests);

    const tensor_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/unit/tensor_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_tensor_unit_tests = b.addRunArtifact(tensor_unit_tests);

    const kernel_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/unit/kernel_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_kernel_unit_tests = b.addRunArtifact(kernel_unit_tests);

    const tokenizer_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/unit/tokenizer_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_tokenizer_unit_tests = b.addRunArtifact(tokenizer_unit_tests);

    const weights_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/unit/weights_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_weights_unit_tests = b.addRunArtifact(weights_unit_tests);

    const manifest_unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/unit/manifest_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_manifest_unit_tests = b.addRunArtifact(manifest_unit_tests);

    const mnist_integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/integration/mnist_golden_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_mnist_integration_tests = b.addRunArtifact(mnist_integration_tests);

    const llm_integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("../tests/integration/llm_golden_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "bolt", .module = mod },
            },
        }),
    });
    const run_llm_integration_tests = b.addRunArtifact(llm_integration_tests);

    const test_step = b.step("test", "Run engine module tests");
    test_step.dependOn(&run_mod_tests.step);
    test_step.dependOn(&run_layout_unit_tests.step);
    test_step.dependOn(&run_tensor_unit_tests.step);
    test_step.dependOn(&run_kernel_unit_tests.step);
    test_step.dependOn(&run_tokenizer_unit_tests.step);
    test_step.dependOn(&run_weights_unit_tests.step);
    test_step.dependOn(&run_manifest_unit_tests.step);
    test_step.dependOn(&run_mnist_integration_tests.step);
    test_step.dependOn(&run_llm_integration_tests.step);
}
