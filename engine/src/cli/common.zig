const std = @import("std");
const fixtures = @import("bolt").fixtures;
const payload_identity = @import("bolt").payload_identity;

pub const ManifestContext = struct {
    manifest: std.json.Parsed(fixtures.FixtureManifest),
    family: fixtures.Family,
    payload_path: []u8,
    expected_path: []u8,
    runtime_bundle_path: ?[]u8,
};

pub fn deinitManifestContext(
    allocator: std.mem.Allocator,
    context: *ManifestContext,
) void {
    context.manifest.deinit();
    allocator.free(context.payload_path);
    allocator.free(context.expected_path);
    if (context.runtime_bundle_path) |runtime_bundle_path| {
        allocator.free(runtime_bundle_path);
    }
}

pub fn expectSinglePathArg(
    allocator: std.mem.Allocator,
    args: []const []const u8,
    usage: []const u8,
) ![]const u8 {
    _ = allocator;
    if (args.len != 2) {
        std.debug.print("{s}\n", .{usage});
        return error.InvalidArguments;
    }
    return args[1];
}

fn loadManifestContextInternal(
    io: std.Io,
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    expected_family: ?fixtures.Family,
) !ManifestContext {
    var manifest = try fixtures.loadFromFile(io, allocator, manifest_path);
    errdefer manifest.deinit();

    const family = try manifest.value.familyTag();
    try manifest.value.validate(expected_family orelse family);

    const manifest_dir = std.fs.path.dirname(manifest_path) orelse ".";
    const payload_path = try std.fs.path.join(
        allocator,
        &.{ manifest_dir, manifest.value.payload_file },
    );
    const expected_path = try std.fs.path.join(
        allocator,
        &.{ manifest_dir, manifest.value.expected_file },
    );
    const runtime_bundle_path = if (manifest.value.runtime_bundle_file) |runtime_bundle_file|
        try std.fs.path.join(allocator, &.{ manifest_dir, runtime_bundle_file })
    else
        null;

    return .{
        .manifest = manifest,
        .family = family,
        .payload_path = payload_path,
        .expected_path = expected_path,
        .runtime_bundle_path = runtime_bundle_path,
    };
}

pub fn loadManifestContext(
    io: std.Io,
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
    expected_family: fixtures.Family,
) !ManifestContext {
    return loadManifestContextInternal(
        io,
        allocator,
        manifest_path,
        expected_family,
    );
}

pub fn loadManifestContextAuto(
    io: std.Io,
    allocator: std.mem.Allocator,
    manifest_path: []const u8,
) !ManifestContext {
    return loadManifestContextInternal(
        io,
        allocator,
        manifest_path,
        null,
    );
}

pub fn requireRuntimePath(path: ?[]u8, what: []const u8) ![]u8 {
    return path orelse {
        std.debug.print("missing runtime path: {s}\n", .{what});
        return error.MissingRuntimePath;
    };
}

pub fn validatePayloadIdentity(
    manifest: fixtures.FixtureManifest,
    payload_family: []const u8,
    payload_fixture_name: []const u8,
) !void {
    try payload_identity.validate(manifest, payload_family, payload_fixture_name);
}

test "expect single path arg" {
    const args = [_][]const u8{
        "runner",
        "manifest.json",
    };
    const path = try expectSinglePathArg(
        std.testing.allocator,
        &args,
        "usage",
    );
    try std.testing.expectEqualStrings("manifest.json", path);
}
