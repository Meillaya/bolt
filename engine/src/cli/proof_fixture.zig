const std = @import("std");
const common = @import("common.zig");
const mnist = @import("bolt").mnist;
const decoder = @import("bolt").decoder;
const debug_reports = @import("bolt").runtime.debug_reports;
const llm_assets = @import("bolt").runtime.llm_assets;
const proof_runs = @import("bolt").runtime.proof_runs;
const proof_checks = @import("bolt").runtime.proof_checks;

const Command = enum {
    run,
    check,
    debug,
};

fn parseCommand(input: []const u8) !Command {
    return std.meta.stringToEnum(Command, input) orelse error.InvalidCommand;
}

fn parseDebugStepCount(input: []const u8) !usize {
    const step_count = try std.fmt.parseUnsigned(usize, input, 10);
    if (step_count == 0) return error.InvalidStepCount;
    return step_count;
}

fn debugLlmFixture(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    context: common.ManifestContext,
    step_count: usize,
) !void {
    var payload = try decoder.loadFixturePayloadFromFile(
        init.io,
        allocator,
        context.payload_path,
    );
    defer payload.deinit();

    var runtime_assets = try llm_assets.loadFromManifestContext(
        init.io,
        allocator,
        context,
    );
    defer runtime_assets.deinit(allocator);

    var trace = try decoder.traceFixtureWithRuntime(
        allocator,
        payload.value,
        runtime_assets.assets.tokenizer,
        runtime_assets.assets.weights,
    );
    defer decoder.freeTrace(allocator, &trace);

    var decode = try decoder.decodeFixtureWithRuntime(
        allocator,
        payload.value,
        runtime_assets.assets.tokenizer,
        runtime_assets.assets.weights,
        step_count,
    );
    defer decoder.freeDecode(allocator, &decode);

    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try debug_reports.writeLlmDebugReport(
        stdout,
        context.family.label(),
        context.manifest.value,
        runtime_assets,
        trace,
        decode,
    );
    try stdout.flush();
}

fn debugMnistFixture(
    init: std.process.Init,
    allocator: std.mem.Allocator,
    context: common.ManifestContext,
) !void {
    var payload = try mnist.loadFixturePayloadFromFile(
        init.io,
        allocator,
        context.payload_path,
    );
    defer payload.deinit();

    var trace = try mnist.traceFixture(allocator, payload.value);
    defer mnist.freeTrace(allocator, &trace);

    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try debug_reports.writeMnistDebugReport(
        stdout,
        context.family.label(),
        context.manifest.value,
        trace,
    );
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 3 and args.len != 4) {
        std.debug.print("usage: proof_fixture <run|check|debug> <manifest-path> [steps]\n", .{});
        return error.InvalidArguments;
    }

    const command = try parseCommand(args[1]);
    const manifest_path = args[2];
    const debug_step_count = if (command == .debug)
        if (args.len == 4) try parseDebugStepCount(args[3]) else 3
    else
        0;

    if ((command == .run or command == .check) and args.len != 3) {
        std.debug.print("usage: proof_fixture <run|check> <manifest-path>\n", .{});
        return error.InvalidArguments;
    }

    var context = try common.loadManifestContextAuto(
        init.io,
        allocator,
        manifest_path,
    );
    defer common.deinitManifestContext(allocator, &context);

    switch (command) {
        .run => {
            const output = switch (context.family) {
                .mnist => try proof_runs.mnistOutput(init.io, allocator, context),
                .llm => try proof_runs.llmOutput(init.io, allocator, context),
            };

            var stdout_buffer: [2048]u8 = undefined;
            var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
            const stdout = &stdout_writer.interface;
            try stdout.writeAll(output);
            try stdout.flush();
        },
        .check => {
            switch (context.family) {
                .mnist => {
                    try proof_checks.mnistCheck(init.io, allocator, context);
                    std.debug.print("mnist-check: PASS\n", .{});
                },
                .llm => {
                    try proof_checks.llmCheck(init.io, allocator, context);
                    std.debug.print("llm-check: PASS\n", .{});
                },
            }
        },
        .debug => switch (context.family) {
            .mnist => try debugMnistFixture(init, allocator, context),
            .llm => try debugLlmFixture(init, allocator, context, debug_step_count),
        },
    }
}

test "parse proof command" {
    try std.testing.expectEqual(.run, try parseCommand("run"));
    try std.testing.expectEqual(.check, try parseCommand("check"));
    try std.testing.expectEqual(.debug, try parseCommand("debug"));
    try std.testing.expectError(error.InvalidCommand, parseCommand("bench"));
}

test "parse debug step count" {
    try std.testing.expectEqual(@as(usize, 3), try parseDebugStepCount("3"));
    try std.testing.expectError(error.InvalidStepCount, parseDebugStepCount("0"));
}
