const std = @import("std");
const core = @import("researcher_core.zig");

const commands = [_]core.CommandSpec{
    .{ .name = "run", .argv = &.{ "zig", "build", "run" }, .artifact = "../artifacts/nnzap-milestone3-run.json" },
    .{ .name = "run-1bit", .argv = &.{ "zig", "build", "run-1bit" }, .artifact = "../artifacts/nnzap-milestone3-run-1bit.json" },
    .{ .name = "run-infer", .argv = &.{ "zig", "build", "run-infer" }, .artifact = "../artifacts/nnzap-milestone3-run-infer.json" },
};

pub fn main() !void {
    try core.writeResearcherSummary(
        std.heap.page_allocator,
        "mnist",
        "../artifacts/labrat-m3-mnist-researcher.json",
        &commands,
    );
}
