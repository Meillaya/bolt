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
        return std.fs.path.join(allocator, &.{ std.mem.span(home), path[2..] });
    }
    return std.fs.path.join(allocator, &.{ repo_root, path });
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
