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

    var payload = try bolt.mnist.loadFixturePayloadFromFile(std.testing.io, allocator, payload_path);
    defer payload.deinit();

    var expected = try bolt.mnist.loadExpectedSummaryFromFile(std.testing.io, allocator, expected_path);
    defer expected.deinit();

    const summary = try bolt.mnist.summarizeFixture(payload.value);
    try bolt.mnist.validateSummary(summary, expected.value);
}
