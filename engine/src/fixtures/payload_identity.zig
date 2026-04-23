const std = @import("std");
const fixtures = @import("manifest.zig");
const mnist_samples = @import("../testing/mnist_samples.zig");

pub fn validate(
    manifest: fixtures.FixtureManifest,
    payload_family: []const u8,
    payload_fixture_name: []const u8,
) !void {
    if (!std.mem.eql(u8, payload_family, manifest.family)) {
        return error.PayloadFamilyMismatch;
    }
    if (!std.mem.eql(u8, payload_fixture_name, manifest.fixture_name)) {
        return error.PayloadFixtureNameMismatch;
    }
}

test "validate payload identity against manifest" {
    const manifest = mnist_samples.manifest();

    try validate(manifest, manifest.family, manifest.fixture_name);
    try std.testing.expectError(
        error.PayloadFamilyMismatch,
        validate(manifest, "llm", manifest.fixture_name),
    );
    try std.testing.expectError(
        error.PayloadFixtureNameMismatch,
        validate(manifest, manifest.family, "wrong-fixture"),
    );
}
