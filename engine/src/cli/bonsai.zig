//! Real Bonsai smoke entry point for Bolt engine.
//! This command is intentionally a fast metadata/tokenizer/tensor-contract
//! smoke gate. Full selected-token parity remains owned by run-bonsai-golden.
const std = @import("std");
const bolt = @import("bolt");
const asset_paths = @import("asset_paths.zig");

const default_manifest_path = "../artifacts/assets/bolt-parity/asset-manifest.json";
const smoke_artifact_path = "../artifacts/bolt-bonsai-smoke.json";

fn readRequiredFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, label: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing {s}: {s}\nRun this command from engine/ after preparing real assets, or pass an explicit manifest path when this command accepts one.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
}

fn requireReferencedAssetFile(io: std.Io, path: []const u8, label: []const u8) !void {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing referenced {s}: {s}\nPrepare the real Bonsai asset listed in the manifest, or update the manifest before rerunning run-bonsai.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
    if (stat.kind != .file or stat.size == 0) {
        std.debug.print("referenced {s} is not a non-empty file: {s}\nUpdate the manifest to point at the prepared real asset before rerunning run-bonsai.\n", .{ label, path });
        return error.MissingRequiredFile;
    }
}

fn missingAssetEntry(asset_id: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest is missing required asset entry: {s}\nPrepare the real asset files and add this asset entry to the manifest before rerunning run-bonsai.\n", .{asset_id});
    return error.MissingRequiredFile;
}

fn missingManifestEntry(asset_id: []const u8, expected_file: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest entry {s} is missing required file path for {s}\nAdd that file entry after preparing real Bonsai assets, then rerun run-bonsai.\n", .{ asset_id, expected_file });
    return error.MissingRequiredFile;
}

const Paths = struct {
    config_path: []const u8,
    tokenizer_path: []const u8,
    safetensors_path: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = if (args.len > 1) args[1] else default_manifest_path;
    const started = std.Io.Clock.awake.now(init.io).nanoseconds;

    const input = try readRequiredFile(init.io, allocator, manifest_path, "asset manifest");
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);
    const paths = try findBonsaiPaths(allocator, repo_root, parsed.value);
    try requireReferencedAssetFile(init.io, paths.config_path, "model config");
    try requireReferencedAssetFile(init.io, paths.tokenizer_path, "tokenizer json");
    try requireReferencedAssetFile(init.io, paths.safetensors_path, "safetensors model");

    const report = bolt.bonsai_model.inspectBonsai1_7BFile(init.io, allocator, paths.safetensors_path) catch |err| {
        std.debug.print("could not inspect referenced safetensors model: {s} ({s})\nVerify that the manifest points at a valid real Bonsai safetensors file before rerunning run-bonsai.\n", .{ paths.safetensors_path, @errorName(err) });
        return error.MissingRequiredFile;
    };
    const pass = report.pass();
    const elapsed_ns: u64 = @intCast(std.Io.Clock.awake.now(init.io).nanoseconds - started);
    try writeArtifact(init, manifest_path, paths, report, pass, elapsed_ns);
    if (!pass) return error.BonsaiSmokeGateFailed;
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        try std.Io.Dir.cwd().createDirPath(io, parent);
    }
}

fn findBonsaiPaths(allocator: std.mem.Allocator, repo_root: []const u8, root: std.json.Value) !Paths {
    if (root != .object) return error.InvalidAssetManifest;
    const assets = root.object.get("assets") orelse return error.MissingAssets;
    if (assets != .array) return error.InvalidAssets;
    for (assets.array.items) |asset| {
        if (asset != .object) continue;
        const idv = asset.object.get("id") orelse continue;
        if (idv != .string or !std.mem.eql(u8, idv.string, "bonsai-1.7b")) continue;
        const files = asset.object.get("files") orelse return error.MissingAssetFiles;
        if (files != .array) return error.InvalidAssetFiles;
        var config_path: ?[]const u8 = null;
        var tokenizer_path: ?[]const u8 = null;
        var safetensors_path: ?[]const u8 = null;
        for (files.array.items) |file| {
            if (file != .object) continue;
            const pv = file.object.get("path") orelse continue;
            if (pv != .string) continue;
            const base = std.fs.path.basename(pv.string);
            if (std.mem.eql(u8, base, "config.json")) config_path = try resolveManifestPath(allocator, repo_root, pv.string);
            if (std.mem.eql(u8, base, "tokenizer.json")) tokenizer_path = try resolveManifestPath(allocator, repo_root, pv.string);
            if (std.mem.eql(u8, base, "model.safetensors")) safetensors_path = try resolveManifestPath(allocator, repo_root, pv.string);
        }
        if (config_path == null) return missingManifestEntry("bonsai-1.7b", "config.json");
        if (tokenizer_path == null) return missingManifestEntry("bonsai-1.7b", "tokenizer.json");
        if (safetensors_path == null) return missingManifestEntry("bonsai-1.7b", "model.safetensors");
        return .{
            .config_path = config_path.?,
            .tokenizer_path = tokenizer_path.?,
            .safetensors_path = safetensors_path.?,
        };
    }
    return missingAssetEntry("bonsai-1.7b");
}

fn repoRootForManifest(allocator: std.mem.Allocator, manifest_path: []const u8) ![]const u8 {
    return asset_paths.repoRootForManifest(allocator, manifest_path);
}

fn resolveManifestPath(allocator: std.mem.Allocator, repo_root: []const u8, path: []const u8) ![]const u8 {
    return asset_paths.resolveManifestPath(allocator, repo_root, path);
}

fn writeArtifact(init: std.process.Init, manifest_path: []const u8, paths: Paths, report: bolt.bonsai_model.BonsaiTensorReport, pass: bool, elapsed_ns: u64) !void {
    try ensureParentDir(init.io, smoke_artifact_path);
    var file = try std.Io.Dir.cwd().createFile(init.io, smoke_artifact_path, .{ .truncate = true });
    defer file.close(init.io);
    var file_buf: [8192]u8 = undefined;
    var fw = file.writer(init.io, &file_buf);
    try writeJson(&fw.interface, manifest_path, paths, report, pass, elapsed_ns);
    try fw.interface.flush();

    var stdout_buf: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    try writeJson(&stdout_writer.interface, manifest_path, paths, report, pass, elapsed_ns);
    try stdout_writer.interface.flush();
}

fn writeJson(w: anytype, manifest_path: []const u8, paths: Paths, report: bolt.bonsai_model.BonsaiTensorReport, pass: bool, elapsed_ns: u64) !void {
    const r = report;
    try w.print(
        "{{\n" ++
            "  \"schema_version\":1,\n" ++
            "  \"gate\":\"bolt-bonsai-smoke\",\n" ++
            "  \"status\":\"{s}\",\n" ++
            "  \"manifest_path\":\"{s}\",\n" ++
            "  \"config_path\":\"{s}\",\n" ++
            "  \"tokenizer_path\":\"{s}\",\n" ++
            "  \"safetensors_path\":\"{s}\",\n" ++
            "  \"tensor_contract\":{{\"expected\":{d},\"matched\":{d},\"missing\":{d},\"invalid\":{d},\"extra\":{d},\"total\":{d},\"passed\":{}}},\n" ++
            "  \"elapsed_ns\":{d},\n" ++
            "  \"golden_policy\":\"run-bonsai is smoke-only; selected-token parity is enforced by run-bonsai-golden\"\n" ++
            "}}\n",
        .{ if (pass) "pass" else "blocked", manifest_path, paths.config_path, paths.tokenizer_path, paths.safetensors_path, r.expected_tensors, r.matched_tensors, r.missing_tensors, r.invalid_tensors, r.extra_tensors, r.tensor_count, r.pass(), elapsed_ns },
    );
}
