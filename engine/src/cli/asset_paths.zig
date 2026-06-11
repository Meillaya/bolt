const std = @import("std");

pub fn repoRootForManifest(allocator: std.mem.Allocator, manifest_path: []const u8) ![]const u8 {
    const marker = "artifacts/";
    if (std.mem.indexOf(u8, manifest_path, marker)) |index| {
        if (index == 0) return allocator.dupe(u8, ".");
        var prefix = manifest_path[0..index];
        while (prefix.len > 0 and (prefix[prefix.len - 1] == '/' or prefix[prefix.len - 1] == std.fs.path.sep)) {
            prefix = prefix[0 .. prefix.len - 1];
        }
        if (prefix.len == 0) return allocator.dupe(u8, ".");
        return allocator.dupe(u8, prefix);
    }
    return allocator.dupe(u8, ".");
}

pub fn resolveManifestPath(allocator: std.mem.Allocator, repo_root: []const u8, path: []const u8) ![]const u8 {
    if (!manifestPathIsAllowed(path)) return error.DisallowedManifestPath;
    if (isHomeModelsPath(path)) {
        const home = std.c.getenv("HOME") orelse return error.HomeUnavailable;
        const resolved_path = try std.fs.path.join(allocator, &.{ std.mem.span(home), path[2..] });
        errdefer allocator.free(resolved_path);
        return try canonicalizeWithinAllowedRoot(allocator, resolved_path, .home_models, repo_root);
    }
    const resolved_path = try std.fs.path.join(allocator, &.{ repo_root, path });
    errdefer allocator.free(resolved_path);
    const root_kind: AllowedRootKind = if (std.mem.startsWith(u8, path, "data/")) .repo_data else .repo_artifacts;
    return try canonicalizeWithinAllowedRoot(allocator, resolved_path, root_kind, repo_root);
}

pub fn manifestPathIsAllowed(path: []const u8) bool {
    if (path.len == 0) return false;
    if (std.fs.path.isAbsolute(path)) return false;
    if (std.mem.indexOf(u8, path, "\x00") != null) return false;
    var parts = std.mem.splitScalar(u8, path, '/');
    while (parts.next()) |part| {
        if (part.len == 0) return false;
        if (std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    }
    return isHomeModelsPath(path) or
        std.mem.startsWith(u8, path, "data/") or
        std.mem.startsWith(u8, path, "artifacts/");
}

fn isHomeModelsPath(path: []const u8) bool {
    return std.mem.startsWith(u8, path, "~/models/");
}

const AllowedRootKind = enum { repo_data, repo_artifacts, home_models };

fn canonicalizeWithinAllowedRoot(
    allocator: std.mem.Allocator,
    resolved_path: []const u8,
    root_kind: AllowedRootKind,
    repo_root: []const u8,
) ![]const u8 {
    const root_path = try allowedRootPath(allocator, root_kind, repo_root);
    defer allocator.free(root_path);

    const canonical_root = realpathAlloc(allocator, root_path) catch |err| switch (err) {
        error.FileNotFound => return resolved_path,
        else => return err,
    };
    defer allocator.free(canonical_root);

    const canonical_path = realpathAlloc(allocator, resolved_path) catch |err| switch (err) {
        error.FileNotFound => return resolved_path,
        else => return err,
    };
    errdefer allocator.free(canonical_path);

    if (!pathIsWithinRoot(canonical_path, canonical_root)) return error.PathEscapesAllowedRoot;
    allocator.free(resolved_path);
    return canonical_path;
}

fn allowedRootPath(allocator: std.mem.Allocator, root_kind: AllowedRootKind, repo_root: []const u8) ![]const u8 {
    return switch (root_kind) {
        .repo_data => std.fs.path.join(allocator, &.{ repo_root, "data" }),
        .repo_artifacts => std.fs.path.join(allocator, &.{ repo_root, "artifacts" }),
        .home_models => blk: {
            const home = std.c.getenv("HOME") orelse return error.HomeUnavailable;
            break :blk std.fs.path.join(allocator, &.{ std.mem.span(home), "models" });
        },
    };
}

fn realpathAlloc(allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    var buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const len = try std.Io.Dir.cwd().realPathFile(io_state.io(), path, &buffer);
    return allocator.dupe(u8, buffer[0..len]);
}

fn pathIsWithinRoot(path: []const u8, root: []const u8) bool {
    if (std.mem.eql(u8, path, root)) return true;
    if (!std.mem.startsWith(u8, path, root)) return false;
    return path.len > root.len and (path[root.len] == '/' or path[root.len] == std.fs.path.sep);
}

test "manifest paths stay repo-local under allowed asset roots" {
    try std.testing.expect(manifestPathIsAllowed("data/qwen/config.json"));
    try std.testing.expect(manifestPathIsAllowed("artifacts/assets/bolt-parity/model.safetensors"));
    try std.testing.expect(manifestPathIsAllowed("~/models/bonsai-1.7b/config.json"));
    try std.testing.expect(!manifestPathIsAllowed("/tmp/model.safetensors"));
    try std.testing.expect(!manifestPathIsAllowed("../data/model.safetensors"));
    try std.testing.expect(!manifestPathIsAllowed("~/other/model.safetensors"));
    try std.testing.expect(!manifestPathIsAllowed("engine/src/root.zig"));
    try std.testing.expect(!manifestPathIsAllowed("data//config.json"));
}

test "manifest path resolution rejects absolute and out-of-root paths" {
    try std.testing.expectError(error.DisallowedManifestPath, resolveManifestPath(std.testing.allocator, ".", "/tmp/model.safetensors"));
    try std.testing.expectError(error.DisallowedManifestPath, resolveManifestPath(std.testing.allocator, ".", "engine/src/root.zig"));
    const resolved = try resolveManifestPath(std.testing.allocator, ".", "data/qwen/config.json");
    defer std.testing.allocator.free(resolved);
    try std.testing.expectEqualStrings("./data/qwen/config.json", resolved);
}

test "home model manifest paths resolve through HOME" {
    const home = std.c.getenv("HOME") orelse return;
    const resolved = try resolveManifestPath(std.testing.allocator, ".", "~/models/bonsai-1.7b/config.json");
    defer std.testing.allocator.free(resolved);
    try std.testing.expect(std.mem.startsWith(u8, resolved, std.mem.span(home)));
    try std.testing.expect(std.mem.endsWith(u8, resolved, "/models/bonsai-1.7b/config.json"));
}

test "manifest path resolution rejects symlink escapes from allowed roots" {
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();
    const root = "zig-cache/tmp/asset-path-symlink-red";
    const data_root = root ++ "/data";
    const outside_root = root ++ "/outside";
    const link_path = data_root ++ "/escape";

    try std.Io.Dir.cwd().deleteTree(io, root);
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try std.Io.Dir.cwd().createDirPath(io, data_root);
    try std.Io.Dir.cwd().createDirPath(io, outside_root);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = outside_root ++ "/payload.bin", .data = "escape" });
    try std.Io.Dir.cwd().symLink(io, "../outside", link_path, .{});

    try std.testing.expectError(
        error.PathEscapesAllowedRoot,
        resolveManifestPath(std.testing.allocator, root, "data/escape/payload.bin"),
    );
}
