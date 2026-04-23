const std = @import("std");
const common = @import("common.zig");
const fixtures = @import("bolt").fixtures;
const proof_checks = @import("bolt").runtime.proof_checks;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = try common.expectSinglePathArg(
        allocator,
        args,
        "usage: check_mnist_fixture <manifest-path>",
    );

    var context = try common.loadManifestContext(
        init.io,
        allocator,
        manifest_path,
        fixtures.Family.mnist,
    );
    defer context.manifest.deinit();

    try proof_checks.mnistCheck(
        init.io,
        allocator,
        context,
    );

    std.debug.print("mnist-check: PASS\n", .{});
}
