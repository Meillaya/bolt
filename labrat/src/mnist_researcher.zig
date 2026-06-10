const std = @import("std");
const core = @import("researcher_core.zig");

const commands = [_]core.CommandSpec{
    .{ .name = "run", .argv = &.{ "zig", "build", "run" }, .artifact = "../artifacts/bolt-mnist-run.json" },
    .{ .name = "run-1bit", .argv = &.{ "zig", "build", "run-1bit" }, .artifact = "../artifacts/bolt-mnist-run-1bit.json" },
    .{ .name = "run-infer", .argv = &.{ "zig", "build", "run-infer" }, .artifact = "../artifacts/bolt-mnist-run-infer.json" },
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
        "mnist",
        "../artifacts/labrat-mnist-researcher.json",
        &commands,
        mode,
    );
}
