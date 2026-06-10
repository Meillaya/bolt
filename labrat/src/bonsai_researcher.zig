const std = @import("std");
const core = @import("researcher_core.zig");

const commands = [_]core.CommandSpec{
    .{ .name = "run-bonsai-golden", .argv = &.{ "zig", "build", "run-bonsai-golden" }, .artifact = "../artifacts/nnzap-milestone6-bonsai-readiness.json" },
    .{ .name = "run-bonsai-bench", .argv = &.{ "zig", "build", "run-bonsai-bench" }, .artifact = "../artifacts/nnzap-milestone8-bonsai-bench.json" },
};

pub fn main() !void {
    try core.writeResearcherSummary(
        std.heap.page_allocator,
        "bonsai",
        "../artifacts/labrat-m3-bonsai-researcher.json",
        &commands,
    );
}
