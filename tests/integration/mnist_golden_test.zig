const std = @import("std");
const bolt = @import("bolt");

fn fromEngineCwd(
    allocator: std.mem.Allocator,
    relative_path: []const u8,
) ![]u8 {
    return std.fs.path.join(allocator, &.{ "..", relative_path });
}

test "mnist committed fixture payload matches committed golden summary" {
    const allocator = std.testing.allocator;
    const manifest_path = try fromEngineCwd(
        allocator,
        "python/fixtures/mnist/manifest.json",
    );
    defer allocator.free(manifest_path);

    var manifest = try bolt.fixtures.loadFromFile(
        std.testing.io,
        allocator,
        manifest_path,
    );
    defer manifest.deinit();
    try manifest.value.validate(.mnist);

    const payload_path = try std.fs.path.join(
        allocator,
        &.{ "../python/fixtures/mnist", manifest.value.payload_file },
    );
    defer allocator.free(payload_path);

    const expected_path = try std.fs.path.join(
        allocator,
        &.{ "../python/fixtures/mnist", manifest.value.expected_file },
    );
    defer allocator.free(expected_path);

    const runtime_bundle_path = try std.fs.path.join(
        allocator,
        &.{ "../python/fixtures/mnist", manifest.value.runtime_bundle_file.? },
    );
    defer allocator.free(runtime_bundle_path);

    var payload = try bolt.mnist.loadFixturePayloadFromFile(std.testing.io, allocator, payload_path);
    defer payload.deinit();

    var expected = try bolt.mnist.loadExpectedSummaryFromFile(std.testing.io, allocator, expected_path);
    defer expected.deinit();

    const manifest_context = .{
        .runtime_bundle_path = @as(?[]u8, runtime_bundle_path),
    };
    var runtime_assets = try bolt.runtime.mnist_assets.loadFromManifestContext(
        std.testing.io,
        allocator,
        manifest_context,
    );
    defer runtime_assets.deinit(allocator);

    const summary = try bolt.mnist.summarizeFixtureWithRuntime(payload.value, runtime_assets.assets);
    try bolt.mnist.validateSummary(summary, expected.value);
    try std.testing.expectEqualStrings("metal", summary.backend);
    try std.testing.expect(summary.dispatched_kernels.matmul_f32);
    try std.testing.expect(summary.dispatched_kernels.bias_add_f32);
    try std.testing.expect(summary.dispatched_kernels.softmax_f32);
    try std.testing.expectEqual(@as(usize, 7), summary.predicted_label);
    try std.testing.expect(summary.top_logit_milli > summary.logits_milli[9]);
}
