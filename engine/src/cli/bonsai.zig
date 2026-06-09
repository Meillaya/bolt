//! Real Bonsai smoke entry point for nnzap Milestone 6/8.
//! This command is intentionally a fast metadata/tokenizer/tensor-contract
//! smoke gate. Full selected-token parity remains owned by run-bonsai-golden.
const std = @import("std");
const bolt = @import("bolt");

const default_manifest_path = "../artifacts/assets/nnzap-parity/asset-manifest.json";
const smoke_artifact_path = "../artifacts/nnzap-milestone6-bonsai-smoke.json";

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

    const input = try std.Io.Dir.cwd().readFileAlloc(init.io, manifest_path, allocator, .limited(16 * 1024 * 1024));
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);
    const paths = try findBonsaiPaths(allocator, repo_root, parsed.value);

    const config_ok = fileExists(init.io, paths.config_path);
    const tokenizer_ok = fileExists(init.io, paths.tokenizer_path);
    const report = bolt.bonsai_model.inspectBonsai1_7BFile(init.io, allocator, paths.safetensors_path) catch null;
    const tensor_contract_ok = if (report) |r| r.pass() else false;
    const pass = config_ok and tokenizer_ok and tensor_contract_ok;
    const elapsed_ns: u64 = @intCast(std.Io.Clock.awake.now(init.io).nanoseconds - started);
    try writeArtifact(init, manifest_path, paths, report, pass, elapsed_ns);
    if (!pass) return error.BonsaiSmokeGateFailed;
}

fn fileExists(io: std.Io, path: []const u8) bool {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return stat.kind == .file and stat.size > 0;
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
        return .{
            .config_path = config_path orelse return error.MissingModelConfig,
            .tokenizer_path = tokenizer_path orelse return error.MissingTokenizerJson,
            .safetensors_path = safetensors_path orelse return error.MissingSafetensors,
        };
    }
    return error.MissingBonsaiAsset;
}

fn repoRootForManifest(allocator: std.mem.Allocator, manifest_path: []const u8) ![]const u8 {
    const marker = "artifacts/";
    if (std.mem.indexOf(u8, manifest_path, marker)) |index| {
        if (index == 0) return allocator.dupe(u8, ".");
        var prefix = manifest_path[0..index];
        while (prefix.len > 0 and prefix[prefix.len - 1] == '/') prefix = prefix[0 .. prefix.len - 1];
        if (prefix.len == 0) return allocator.dupe(u8, ".");
        return allocator.dupe(u8, prefix);
    }
    return allocator.dupe(u8, ".");
}

fn resolveManifestPath(allocator: std.mem.Allocator, repo_root: []const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path)) return allocator.dupe(u8, path);
    return std.fs.path.join(allocator, &.{ repo_root, path });
}

fn writeArtifact(init: std.process.Init, manifest_path: []const u8, paths: Paths, report: ?bolt.bonsai_model.BonsaiTensorReport, pass: bool, elapsed_ns: u64) !void {
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

fn writeJson(w: anytype, manifest_path: []const u8, paths: Paths, report: ?bolt.bonsai_model.BonsaiTensorReport, pass: bool, elapsed_ns: u64) !void {
    const r = report orelse bolt.bonsai_model.BonsaiTensorReport{ .expected_tensors = 310, .matched_tensors = 0, .missing_tensors = 310, .invalid_tensors = 0, .extra_tensors = 0, .tensor_count = 0 };
    try w.print(
        "{{\n" ++
            "  \"schema_version\":1,\n" ++
            "  \"gate\":\"nnzap-bonsai-smoke\",\n" ++
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
