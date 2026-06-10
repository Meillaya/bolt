const std = @import("std");
const bolt = @import("bolt");
const asset_paths = @import("asset_paths.zig");

const default_manifest_path = "../artifacts/assets/bolt-parity/asset-manifest.json";
const GoldenPrompt = struct {
    name: []const u8,
    prompt: []const u8,
    expected: []const u32,
};

const prompt_a_ids = [_]u32{ 151644, 872, 198, 785, 6722, 315, 9625, 374, 151645, 198, 151644, 77091, 198 };
const prompt_b_ids = [_]u32{ 151644, 872, 198, 9707, 0, 151645, 198, 151644, 77091, 198 };

const golden_prompts = [_]GoldenPrompt{
    .{ .name = "capital_france", .prompt = "The capital of France is", .expected = &prompt_a_ids },
    .{ .name = "hello", .prompt = "Hello!", .expected = &prompt_b_ids },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = if (args.len > 1) args[1] else default_manifest_path;
    const input = try readRequiredFile(init.io, allocator, manifest_path, "asset manifest");
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);

    const bonsai_paths = try findModelPaths(allocator, repo_root, parsed.value, "bonsai-1.7b");
    const qwen_paths = try findModelPaths(allocator, repo_root, parsed.value, "qwen3-1.7b-q4-gs64");

    const bonsai_vocab = try readConfigVocabSize(init.io, allocator, bonsai_paths.config_path);
    const qwen_vocab = try readConfigVocabSize(init.io, allocator, qwen_paths.config_path);

    var bonsai: bolt.tokenizer.Tokenizer = undefined;
    try loadTokenizerFromFile(init.io, allocator, &bonsai, bonsai_paths.tokenizer_path);
    defer bonsai.deinit();

    var qwen: bolt.tokenizer.Tokenizer = undefined;
    try loadTokenizerFromFile(init.io, allocator, &qwen, qwen_paths.tokenizer_path);
    defer qwen.deinit();

    const special_match = bonsai.eos_token_id == qwen.eos_token_id and bonsai.im_start_token_id == qwen.im_start_token_id and bonsai.im_end_token_id == qwen.im_end_token_id;
    const fixed_goldens_match = try allFixedGoldensMatch(&bonsai, &qwen);
    const config_pairing = tokenizerFitsConfig(bonsai, bonsai_vocab) and tokenizerFitsConfig(qwen, qwen_vocab);
    const pass = special_match and fixed_goldens_match and config_pairing;

    try writeReport(init, manifest_path, bonsai_paths, qwen_paths, bonsai_vocab, qwen_vocab, bonsai, qwen, pass);
    if (!pass) return error.TokenizerParityGateFailed;
}

fn readRequiredFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, label: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing {s}: {s}\nRun this command from engine/ after preparing real assets, or pass an explicit manifest path when this command accepts one.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
}

fn readReferencedAssetFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, label: []const u8, max_bytes: usize) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_bytes)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing referenced {s}: {s}\nPrepare the real asset listed in the manifest, or update the manifest before rerunning validate-tokenizer.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
}

fn missingManifestEntry(asset_id: []const u8, expected_file: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest entry {s} is missing required file path for {s}\nAdd that file entry to the manifest after preparing real assets, then rerun validate-tokenizer.\n", .{ asset_id, expected_file });
    return error.MissingRequiredFile;
}

fn missingAssetEntry(asset_id: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest is missing required asset entry: {s}\nPrepare the real asset files and add this asset entry to the manifest before rerunning validate-tokenizer.\n", .{asset_id});
    return error.MissingRequiredFile;
}

const ModelPaths = struct {
    config_path: []const u8,
    tokenizer_path: []const u8,
};

fn findModelPaths(allocator: std.mem.Allocator, repo_root: []const u8, root: std.json.Value, id: []const u8) !ModelPaths {
    if (root != .object) return error.InvalidAssetManifest;
    const assets = root.object.get("assets") orelse return error.MissingAssets;
    if (assets != .array) return error.InvalidAssets;
    for (assets.array.items) |asset| {
        if (asset != .object) continue;
        const id_value = asset.object.get("id") orelse continue;
        if (id_value != .string or !std.mem.eql(u8, id_value.string, id)) continue;
        const files = asset.object.get("files") orelse return error.MissingAssetFiles;
        if (files != .array) return error.InvalidAssetFiles;
        var config_path: ?[]const u8 = null;
        var tokenizer_path: ?[]const u8 = null;
        for (files.array.items) |file| {
            if (file != .object) continue;
            const path_value = file.object.get("path") orelse continue;
            if (path_value != .string) continue;
            const base = std.fs.path.basename(path_value.string);
            if (std.mem.eql(u8, base, "config.json")) config_path = try resolveManifestPath(allocator, repo_root, path_value.string);
            if (std.mem.eql(u8, base, "tokenizer.json")) tokenizer_path = try resolveManifestPath(allocator, repo_root, path_value.string);
        }
        if (config_path == null) return missingManifestEntry(id, "config.json");
        if (tokenizer_path == null) return missingManifestEntry(id, "tokenizer.json");
        return .{
            .config_path = config_path.?,
            .tokenizer_path = tokenizer_path.?,
        };
    }
    return missingAssetEntry(id);
}

fn readConfigVocabSize(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !u32 {
    const input = try readReferencedAssetFile(io, allocator, path, "model config", 1024 * 1024);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidModelConfig;
    const value = parsed.value.object.get("vocab_size") orelse return error.MissingVocabSize;
    if (value != .integer or value.integer <= 0) return error.InvalidVocabSize;
    return @intCast(value.integer);
}

fn loadTokenizerFromFile(io: std.Io, allocator: std.mem.Allocator, out: *bolt.tokenizer.Tokenizer, path: []const u8) !void {
    const input = try readReferencedAssetFile(io, allocator, path, "tokenizer json", 32 * 1024 * 1024);
    try out.initFromJson(allocator, input);
}

fn slicesEqual(a: []const u32, b: []const u32) bool {
    if (a.len != b.len) return false;
    for (a, b) |x, y| if (x != y) return false;
    return true;
}

fn tokenizerFitsConfig(tokenizer: bolt.tokenizer.Tokenizer, config_vocab_size: u32) bool {
    return tokenizer.vocab_size <= config_vocab_size and
        tokenizer.eos_token_id < config_vocab_size and
        tokenizer.im_start_token_id < config_vocab_size and
        tokenizer.im_end_token_id < config_vocab_size;
}

fn allFixedGoldensMatch(bonsai: *bolt.tokenizer.Tokenizer, qwen: *bolt.tokenizer.Tokenizer) !bool {
    var b_ids: [256]u32 = undefined;
    var q_ids: [256]u32 = undefined;
    for (golden_prompts) |golden| {
        const b_count = try bonsai.applyChatTemplate(golden.prompt, &b_ids);
        const q_count = try qwen.applyChatTemplate(golden.prompt, &q_ids);
        if (!slicesEqual(b_ids[0..b_count], golden.expected)) return false;
        if (!slicesEqual(q_ids[0..q_count], golden.expected)) return false;
    }
    return true;
}

fn repoRootForManifest(allocator: std.mem.Allocator, manifest_path: []const u8) ![]const u8 {
    return asset_paths.repoRootForManifest(allocator, manifest_path);
}

fn resolveManifestPath(allocator: std.mem.Allocator, repo_root: []const u8, path: []const u8) ![]const u8 {
    return asset_paths.resolveManifestPath(allocator, repo_root, path);
}

fn writeIds(stdout: anytype, ids: []const u32) !void {
    try stdout.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try stdout.writeAll(",");
        try stdout.print("{d}", .{id});
    }
    try stdout.writeByte(']');
}

fn writeReport(
    init: std.process.Init,
    manifest_path: []const u8,
    bonsai_paths: ModelPaths,
    qwen_paths: ModelPaths,
    bonsai_vocab: u32,
    qwen_vocab: u32,
    bonsai: bolt.tokenizer.Tokenizer,
    qwen: bolt.tokenizer.Tokenizer,
    pass: bool,
) !void {
    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "{{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"gate\": \"bolt-tokenizer-validation\",\n" ++
            "  \"manifest_path\": \"{s}\",\n" ++
            "  \"status\": \"{s}\",\n" ++
            "  \"bonsai\": {{\"tokenizer\":\"{s}\",\"config\":\"{s}\",\"config_vocab_size\":{d},\"tokenizer_vocab_size\":{d},\"eos\":{d},\"im_start\":{d},\"im_end\":{d}}},\n" ++
            "  \"qwen_q4_gs64\": {{\"tokenizer\":\"{s}\",\"config\":\"{s}\",\"config_vocab_size\":{d},\"tokenizer_vocab_size\":{d},\"eos\":{d},\"im_start\":{d},\"im_end\":{d}}},\n" ++
            "  \"goldens\": [\n",
        .{
            manifest_path,
            if (pass) "pass" else "fail",
            bonsai_paths.tokenizer_path,
            bonsai_paths.config_path,
            bonsai_vocab,
            bonsai.vocab_size,
            bonsai.eos_token_id,
            bonsai.im_start_token_id,
            bonsai.im_end_token_id,
            qwen_paths.tokenizer_path,
            qwen_paths.config_path,
            qwen_vocab,
            qwen.vocab_size,
            qwen.eos_token_id,
            qwen.im_start_token_id,
            qwen.im_end_token_id,
        },
    );
    for (golden_prompts, 0..) |golden, index| {
        if (index != 0) try stdout.writeAll(",\n");
        try stdout.print("    {{\"name\":\"{s}\",\"prompt\":\"{s}\",\"source\":\"fixed-reference-chat-template-ids\",\"expected_ids\":", .{ golden.name, golden.prompt });
        try writeIds(stdout, golden.expected);
        try stdout.writeAll("}");
    }
    try stdout.writeAll("\n  ],\n  \"comparison_policy\":\"Both Bonsai and Qwen tokenizers must match fixed reference token IDs; local-vs-local equality alone is insufficient.\"\n}\n");
    try stdout.flush();
}
