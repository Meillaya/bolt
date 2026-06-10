const std = @import("std");
const core = @import("researcher_core.zig");

const commands = [_]core.CommandSpec{
    .{ .name = "run-bonsai-q4-golden", .argv = &.{ "zig", "build", "run-bonsai-q4-golden" }, .artifact = "../artifacts/nnzap-milestone7-q4-golden.json" },
    .{ .name = "run-bonsai-q4-bench", .argv = &.{ "zig", "build", "run-bonsai-q4-bench" }, .artifact = "../artifacts/nnzap-milestone7-q4-bench.json" },
};

pub fn main() !void {
    try core.writeResearcherSummary(
        std.heap.page_allocator,
        "bonsai_q4",
        "../artifacts/labrat-m3-bonsai-q4-researcher.json",
        &commands,
    );
}
