const std = @import("std");
const common = @import("common.zig");
const fixtures = @import("bolt").fixtures;
const proof_runs = @import("bolt").runtime.proof_runs;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = try common.expectSinglePathArg(
        allocator,
        args,
        "usage: run_llm_fixture <manifest-path>",
    );

    var context = try common.loadManifestContext(
        init.io,
        allocator,
        manifest_path,
        fixtures.Family.llm,
    );
    defer context.manifest.deinit();

    const output = try proof_runs.llmOutput(
        init.io,
        allocator,
        context,
    );

    var stdout_buffer: [1536]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.writeAll(output);
    try stdout.flush();
}
