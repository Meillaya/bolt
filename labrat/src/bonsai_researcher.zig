const std = @import("std");
const core = @import("researcher_core.zig");

const commands = [_]core.CommandSpec{
    .{ .name = "run-bonsai-golden", .argv = &.{ "zig", "build", "run-bonsai-golden" }, .artifact = "../artifacts/bolt-bonsai-readiness.json" },
    .{ .name = "run-bonsai-bench", .argv = &.{ "zig", "build", "run-bonsai-bench" }, .artifact = "../artifacts/bolt-bonsai-bench.json" },
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
        "bonsai",
        "../artifacts/labrat-bonsai-researcher.json",
        &commands,
        mode,
    );
}
