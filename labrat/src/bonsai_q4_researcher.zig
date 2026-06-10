const std = @import("std");
const core = @import("researcher_core.zig");

const commands = [_]core.CommandSpec{
    .{ .name = "run-bonsai-q4-golden", .argv = &.{ "zig", "build", "run-bonsai-q4-golden" }, .artifact = "../artifacts/bolt-q4-golden.json" },
    .{ .name = "run-bonsai-q4-bench", .argv = &.{ "zig", "build", "run-bonsai-q4-bench" }, .artifact = "../artifacts/bolt-q4-bench.json" },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const mode = core.researcherModeFromArgs(args) catch |err| {
        std.debug.print(
            "unsupported researcher command\nsupported commands: summary, bench-compare, summaries\n",
            .{},
        );
        return err;
    };
    try core.writeResearcherSummaryMode(
        allocator,
        "bonsai_q4",
        "../artifacts/labrat-bonsai-q4-researcher.json",
        &commands,
        mode,
    );
}
