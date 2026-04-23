const std = @import("std");
const mnist_samples = @import("../testing/mnist_samples.zig");
const llm_samples = @import("../testing/llm_samples.zig");

pub const current_fixture_version: usize = 1;

pub const Family = enum {
    mnist,
    llm,

    pub fn parse(input: []const u8) !Family {
        return std.meta.stringToEnum(Family, input) orelse error.UnsupportedFamily;
    }

    pub fn label(self: Family) []const u8 {
        return @tagName(self);
    }
};

pub const FixtureManifest = struct {
    fixture_version: usize,
    family: []const u8,
    fixture_name: []const u8,
    payload_file: []const u8,
    expected_file: []const u8,
    runtime_bundle_file: ?[]const u8 = null,
    description: []const u8,

    pub fn familyTag(self: FixtureManifest) !Family {
        return Family.parse(self.family);
    }

    pub fn validate(self: FixtureManifest, expected_family: Family) !void {
        if (self.fixture_version != current_fixture_version) {
            return error.UnsupportedFixtureVersion;
        }
        if (self.fixture_name.len == 0) {
            return error.EmptyFixtureName;
        }
        if (self.payload_file.len == 0) {
            return error.EmptyPayloadFile;
        }
        if (self.expected_file.len == 0) {
            return error.EmptyExpectedFile;
        }
        if (self.description.len == 0) {
            return error.EmptyDescription;
        }

        const family = try self.familyTag();
        if (family != expected_family) {
            return error.UnexpectedFamily;
        }
        if (family == .llm) {
            const runtime_bundle_file = self.runtime_bundle_file orelse return error.MissingRuntimeBundleFile;
            if (runtime_bundle_file.len == 0) return error.EmptyRuntimeBundleFile;
        }
    }
};

pub fn parseFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(FixtureManifest) {
    return std.json.parseFromSlice(
        FixtureManifest,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(FixtureManifest) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(64 * 1024),
    );
    defer allocator.free(input);

    return parseFromSlice(allocator, input);
}

test "parse manifest json" {
    const allocator = std.testing.allocator;
    const expected = mnist_samples.manifest();
    const input = std.fmt.comptimePrint(
        \\{{
        \\  "fixture_version": {d},
        \\  "family": "{s}",
        \\  "fixture_name": "{s}",
        \\  "payload_file": "{s}",
        \\  "expected_file": "{s}",
        \\  "description": "{s}"
        \\}}
    , .{
        current_fixture_version,
        mnist_samples.family_name,
        mnist_samples.fixture_name,
        mnist_samples.payload_file_name,
        mnist_samples.expected_file_name,
        mnist_samples.description,
    });

    var parsed = try parseFromSlice(allocator, input);
    defer parsed.deinit();

    try std.testing.expectEqual(@as(usize, expected.fixture_version), parsed.value.fixture_version);
    try std.testing.expectEqualStrings(expected.family, parsed.value.family);
    try std.testing.expectEqualStrings(expected.fixture_name, parsed.value.fixture_name);
    try std.testing.expectEqualStrings(expected.payload_file, parsed.value.payload_file);
    try std.testing.expectEqualStrings(expected.expected_file, parsed.value.expected_file);
}

test "validate manifest family and version" {
    const manifest = llm_samples.manifest();

    try manifest.validate(.llm);
    try std.testing.expectError(error.UnexpectedFamily, manifest.validate(.mnist));
}
