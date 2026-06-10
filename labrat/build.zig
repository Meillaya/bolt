const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const tools_module = b.addModule("tools.zig", .{
        .root_source_file = b.path("src/tools.zig"),
        .target = target,
        .optimize = optimize,
    });

    const toolbox_module = b.addModule("toolbox.zig", .{
        .root_source_file = b.path("src/toolbox.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "tools.zig", .module = tools_module }},
    });

    const api_client_module = b.addModule("api_client.zig", .{
        .root_source_file = b.path("src/api_client.zig"),
        .target = target,
        .optimize = optimize,
    });

    const agent_core_module = b.addModule("agent_core.zig", .{
        .root_source_file = b.path("src/agent_core.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "api_client.zig", .module = api_client_module }},
    });

    const agent_cli_core_module = b.addModule("agent_cli_core.zig", .{
        .root_source_file = b.path("src/agent_cli_core.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "api_client.zig", .module = api_client_module },
            .{ .name = "agent_core.zig", .module = agent_core_module },
        },
    });

    const mnist_agent = b.addExecutable(.{
        .name = "mnist_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mnist_agent.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "agent_cli_core.zig", .module = agent_cli_core_module }},
        }),
    });
    b.installArtifact(mnist_agent);

    const bonsai_agent = b.addExecutable(.{
        .name = "bonsai_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_agent.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "agent_cli_core.zig", .module = agent_cli_core_module }},
        }),
    });
    b.installArtifact(bonsai_agent);

    const bonsai_q4_agent = b.addExecutable(.{
        .name = "bonsai_q4_agent",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_q4_agent.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "agent_cli_core.zig", .module = agent_cli_core_module }},
        }),
    });
    b.installArtifact(bonsai_q4_agent);

    const mnist_agent_run = b.addRunArtifact(mnist_agent);
    const mnist_agent_live = b.addRunArtifact(mnist_agent);
    mnist_agent_live.setEnvironmentVariable("LABRAT_LIVE", "1");
    mnist_agent_live.setEnvironmentVariable("ANTHROPIC_API_KEY", "");
    mnist_agent_live.step.dependOn(&mnist_agent_run.step);
    const mnist_agent_step = b.step("mnist-agent", "Run MNIST Labrat agent offline and live-block gates");
    mnist_agent_step.dependOn(&mnist_agent_live.step);

    const bonsai_agent_run = b.addRunArtifact(bonsai_agent);
    const bonsai_agent_live = b.addRunArtifact(bonsai_agent);
    bonsai_agent_live.setEnvironmentVariable("LABRAT_LIVE", "1");
    bonsai_agent_live.setEnvironmentVariable("ANTHROPIC_API_KEY", "");
    bonsai_agent_live.step.dependOn(&bonsai_agent_run.step);
    const bonsai_agent_step = b.step("bonsai-agent", "Run Bonsai Labrat agent offline and live-block gates");
    bonsai_agent_step.dependOn(&bonsai_agent_live.step);

    const bonsai_q4_agent_run = b.addRunArtifact(bonsai_q4_agent);
    const bonsai_q4_agent_live = b.addRunArtifact(bonsai_q4_agent);
    bonsai_q4_agent_live.setEnvironmentVariable("LABRAT_LIVE", "1");
    bonsai_q4_agent_live.setEnvironmentVariable("ANTHROPIC_API_KEY", "");
    bonsai_q4_agent_live.step.dependOn(&bonsai_q4_agent_run.step);
    const bonsai_q4_agent_step = b.step("bonsai-q4-agent", "Run Bonsai Q4 Labrat agent offline and live-block gates");
    bonsai_q4_agent_step.dependOn(&bonsai_q4_agent_live.step);

    const researcher_core_module = b.addModule("researcher_core.zig", .{
        .root_source_file = b.path("src/researcher_core.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "tools.zig", .module = tools_module }},
    });

    const mnist_researcher = b.addExecutable(.{
        .name = "mnist_researcher",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/mnist_researcher.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "researcher_core.zig", .module = researcher_core_module }},
        }),
    });
    b.installArtifact(mnist_researcher);

    const bonsai_researcher = b.addExecutable(.{
        .name = "bonsai_researcher",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_researcher.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "researcher_core.zig", .module = researcher_core_module }},
        }),
    });
    b.installArtifact(bonsai_researcher);

    const bonsai_q4_researcher = b.addExecutable(.{
        .name = "bonsai_q4_researcher",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bonsai_q4_researcher.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{.{ .name = "researcher_core.zig", .module = researcher_core_module }},
        }),
    });
    b.installArtifact(bonsai_q4_researcher);

    const mnist_run = b.addSystemCommand(&.{ "zig", "build", "run" });
    mnist_run.setCwd(b.path("../engine"));
    const mnist_1bit = b.addSystemCommand(&.{ "zig", "build", "run-1bit" });
    mnist_1bit.setCwd(b.path("../engine"));
    mnist_1bit.step.dependOn(&mnist_run.step);
    const mnist_infer = b.addSystemCommand(&.{ "zig", "build", "run-infer" });
    mnist_infer.setCwd(b.path("../engine"));
    mnist_infer.step.dependOn(&mnist_1bit.step);
    const mnist_summary = b.addRunArtifact(mnist_researcher);
    mnist_summary.step.dependOn(&mnist_infer.step);
    const mnist_researcher_step = b.step("mnist-researcher", "Run MNIST Labrat researcher gate");
    mnist_researcher_step.dependOn(&mnist_summary.step);

    const bonsai_golden = b.addSystemCommand(&.{ "zig", "build", "run-bonsai-golden" });
    bonsai_golden.setCwd(b.path("../engine"));
    const bonsai_bench = b.addSystemCommand(&.{ "zig", "build", "run-bonsai-bench" });
    bonsai_bench.setCwd(b.path("../engine"));
    bonsai_bench.step.dependOn(&bonsai_golden.step);
    const bonsai_summary = b.addRunArtifact(bonsai_researcher);
    bonsai_summary.step.dependOn(&bonsai_bench.step);
    const bonsai_researcher_step = b.step("bonsai-researcher", "Run Bonsai Labrat researcher gate");
    bonsai_researcher_step.dependOn(&bonsai_summary.step);

    const q4_golden = b.addSystemCommand(&.{ "zig", "build", "run-bonsai-q4-golden" });
    q4_golden.setCwd(b.path("../engine"));
    const q4_bench = b.addSystemCommand(&.{ "zig", "build", "run-bonsai-q4-bench" });
    q4_bench.setCwd(b.path("../engine"));
    q4_bench.step.dependOn(&q4_golden.step);
    const q4_summary = b.addRunArtifact(bonsai_q4_researcher);
    q4_summary.step.dependOn(&q4_bench.step);
    const bonsai_q4_researcher_step = b.step("bonsai-q4-researcher", "Run Bonsai Q4 Labrat researcher gate");
    bonsai_q4_researcher_step.dependOn(&q4_summary.step);

    const toolbox_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/toolbox.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "tools.zig", .module = tools_module }},
        }),
    });

    const tools_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "tools.zig", .module = tools_module }},
        }),
    });

    const toolbox_extra_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/toolbox_test.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "tools.zig", .module = tools_module },
                .{ .name = "toolbox.zig", .module = toolbox_module },
            },
        }),
    });

    const api_client_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/api_client.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const agent_core_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/agent_core.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "api_client.zig", .module = api_client_module }},
        }),
    });

    const api_offline_step = b.step("api-offline-test", "Run Labrat offline API and agent-core tests");
    api_offline_step.dependOn(&b.addRunArtifact(api_client_tests).step);
    api_offline_step.dependOn(&b.addRunArtifact(agent_core_tests).step);

    const test_step = b.step("test", "Run Labrat tests");
    test_step.dependOn(&b.addRunArtifact(toolbox_tests).step);
    test_step.dependOn(&b.addRunArtifact(tools_tests).step);
    test_step.dependOn(&b.addRunArtifact(toolbox_extra_tests).step);
    test_step.dependOn(&b.addRunArtifact(api_client_tests).step);
    test_step.dependOn(&b.addRunArtifact(agent_core_tests).step);
}
