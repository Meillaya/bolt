const std = @import("std");
const bolt = @import("bolt");

fn fromEngineCwd(
    allocator: std.mem.Allocator,
    relative_path: []const u8,
) ![]u8 {
    return std.fs.path.join(allocator, &.{ "..", relative_path });
}

test "llm manifest parse preserves runtime bundle contract" {
    const allocator = std.testing.allocator;
    const manifest_path = try fromEngineCwd(
        allocator,
        "python/fixtures/llm/manifest.json",
    );
    defer allocator.free(manifest_path);

    var parsed = try bolt.fixtures.loadFromFile(
        std.testing.io,
        allocator,
        manifest_path,
    );
    defer parsed.deinit();

    try parsed.value.validate(.llm);
    try std.testing.expectEqual(@as(usize, bolt.fixtures.current_fixture_version), parsed.value.fixture_version);
    try std.testing.expectEqualStrings("llm-smoke", parsed.value.fixture_name);
    try std.testing.expectEqualStrings("llm-smoke.runtime.json", parsed.value.runtime_bundle_file.?);
}
