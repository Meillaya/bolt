const std = @import("std");
const bolt = @import("bolt");
const asset_paths = @import("asset_paths.zig");

const default_manifest_path = "../artifacts/assets/bolt-parity/asset-manifest.json";
const readiness_artifact_path = "../artifacts/bolt-bonsai-readiness.json";
const blocker_artifact_path = "../artifacts/blockers/bonsai-decode-incomplete.json";
const golden_prompt = "The capital of France is";
const hidden_size = 2048;
const intermediate_size = 6144;
const vocab_size = 151669;
const model_layer_count = 28;
const num_query_heads = 16;
const num_kv_heads = 8;
const head_dim = 128;
const kv_dim = num_kv_heads * head_dim;
const partial_probe_layers = 28;
const max_generated_tokens = 11;
const logits_chunk_rows = 2048;
const golden_tokens = [_]u32{ 151667, 198, 151668, 271, 785, 6722, 315, 9625, 374, 12095, 13 };

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
            std.debug.print("missing referenced {s}: {s}\nPrepare the real Bonsai asset listed in the manifest, or update the manifest before rerunning run-bonsai-golden.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
}

fn requireReferencedAssetFile(io: std.Io, path: []const u8, label: []const u8) !void {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing referenced {s}: {s}\nPrepare the real Bonsai asset listed in the manifest, or update the manifest before rerunning run-bonsai-golden.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
    if (stat.kind != .file) {
        std.debug.print("referenced {s} is not a file: {s}\nUpdate the manifest to point at the prepared real asset before rerunning run-bonsai-golden.\n", .{ label, path });
        return error.MissingRequiredFile;
    }
}

fn missingAssetEntry(asset_id: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest is missing required asset entry: {s}\nPrepare the real asset files and add this asset entry to the manifest before rerunning run-bonsai-golden.\n", .{asset_id});
    return error.MissingRequiredFile;
}

fn missingManifestEntry(asset_id: []const u8, expected_file: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest entry {s} is missing required file path for {s}\nAdd that file entry after preparing real Bonsai assets, then rerun run-bonsai-golden.\n", .{ asset_id, expected_file });
    return error.MissingRequiredFile;
}

pub const BonsaiDecodeBenchmark = struct {
    prompt_tokens: u32,
    generated_tokens: u32,
    matches_reference: bool,
    elapsed_ns: u64,
    layers_per_token: u32,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = if (args.len > 1) args[1] else default_manifest_path;

    const manifest_input = try readRequiredFile(init.io, allocator, manifest_path, "asset manifest");
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);

    const paths = try findBonsaiPaths(allocator, repo_root, parsed.value);
    try requireReferencedAssetFile(init.io, paths.config_path, "model config");
    try requireReferencedAssetFile(init.io, paths.safetensors_path, "safetensors model");
    const report = try bolt.bonsai_model.inspectBonsai1_7BFile(init.io, allocator, paths.safetensors_path);

    var tok: bolt.tokenizer.Tokenizer = undefined;
    try loadTokenizerFromFile(init.io, allocator, &tok, paths.tokenizer_path);
    defer tok.deinit();

    var prompt_ids: [256]u32 = undefined;
    const prompt_count = try tok.applyChatTemplate(golden_prompt, &prompt_ids);

    const tensor_pass = report.pass();
    const tokenizer_pass = prompt_count > 0 and tok.eos_token_id == 151643 and tok.im_start_token_id == 151644 and tok.im_end_token_id == 151645;
    const readiness_pass = tensor_pass and tokenizer_pass;
    const probe = try computeWeightProbe(init.io, allocator, paths.safetensors_path, prompt_ids[prompt_count - 1]);
    const partial_probe = try computePartialDecodeProbe(init.io, paths.safetensors_path, prompt_ids[prompt_count - 1], partial_probe_layers);
    const generation_probe = try computeGenerationProbe(init.io, paths.safetensors_path, prompt_ids[0..prompt_count]);
    const gate_pass = readiness_pass and generation_probe.matches_reference;

    try writeReadiness(init, manifest_path, paths, report, tok, prompt_ids[0..prompt_count], probe, partial_probe, generation_probe, readiness_pass, gate_pass);
    try writeBlocker(init, manifest_path, paths, report, prompt_ids[0..prompt_count], probe, partial_probe, generation_probe, readiness_pass, gate_pass);
    try writeStdout(init, manifest_path, paths, report, prompt_ids[0..prompt_count], probe, partial_probe, generation_probe, readiness_pass, gate_pass);

    if (!gate_pass) return error.RealBonsaiDecodeMismatch;
}

pub fn runDecodeBenchmark(io: std.Io, allocator: std.mem.Allocator, manifest_path: []const u8) !BonsaiDecodeBenchmark {
    const manifest_input = try readRequiredFile(io, allocator, manifest_path, "asset manifest");
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);
    const paths = try findBonsaiPaths(allocator, repo_root, parsed.value);
    try requireReferencedAssetFile(io, paths.config_path, "model config");
    try requireReferencedAssetFile(io, paths.safetensors_path, "safetensors model");
    var tok: bolt.tokenizer.Tokenizer = undefined;
    try loadTokenizerFromFile(io, allocator, &tok, paths.tokenizer_path);
    defer tok.deinit();
    var prompt_ids: [256]u32 = undefined;
    const prompt_count = try tok.applyChatTemplate(golden_prompt, &prompt_ids);
    const started = std.Io.Clock.awake.now(io).nanoseconds;
    const generation = try computeGenerationProbe(io, paths.safetensors_path, prompt_ids[0..prompt_count]);
    const elapsed: u64 = @intCast(std.Io.Clock.awake.now(io).nanoseconds - started);
    return .{
        .prompt_tokens = @intCast(prompt_count),
        .generated_tokens = generation.generated_count,
        .matches_reference = generation.matches_reference,
        .elapsed_ns = elapsed,
        .layers_per_token = 28,
    };
}

const BonsaiPaths = struct {
    config_path: []const u8,
    tokenizer_path: []const u8,
    safetensors_path: []const u8,
};

const F16Model = struct {
    embedding: []f32,
    final_norm: []f32,
    layers: []bolt.transformer.CpuLayerWeights,

    fn deinit(self: *F16Model, allocator: std.mem.Allocator) void {
        allocator.free(self.embedding);
        allocator.free(self.final_norm);
        for (self.layers) |layer| freeLayer(allocator, layer);
        allocator.free(self.layers);
    }
};

const WeightProbe = struct {
    source_token: u32,
    best_candidate_token: u32,
    best_candidate_logit: f32,
    candidate_logits: [golden_tokens.len]f32,
};

const PartialDecodeProbe = struct {
    layers_executed: u32,
    last_layer_index: u32,
    source_token: u32,
    output_l2: f32,
    output_first4: [4]f32,
    best_candidate_token: u32,
    best_candidate_logit: f32,
    candidate_logits: [golden_tokens.len]f32,
};

const TokenLogit = struct {
    token: u32,
    logit: f32,
};

const GenerationProbe = struct {
    prompt_tokens: u32,
    max_tokens: u32,
    generated_count: u32,
    generated_tokens: [max_generated_tokens]u32,
    top_logits: [max_generated_tokens]f32,
    matched_tokens: u32,
    first_mismatch_index: i32,
    matches_reference: bool,
};

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        try std.Io.Dir.cwd().createDirPath(io, parent);
    }
}

fn findBonsaiPaths(allocator: std.mem.Allocator, repo_root: []const u8, root: std.json.Value) !BonsaiPaths {
    if (root != .object) return error.InvalidAssetManifest;
    const assets = root.object.get("assets") orelse return error.MissingAssets;
    if (assets != .array) return error.InvalidAssets;
    for (assets.array.items) |asset| {
        if (asset != .object) continue;
        const id_value = asset.object.get("id") orelse continue;
        if (id_value != .string or !std.mem.eql(u8, id_value.string, "bonsai-1.7b")) continue;
        const files = asset.object.get("files") orelse return error.MissingAssetFiles;
        if (files != .array) return error.InvalidAssetFiles;
        var config_path: ?[]const u8 = null;
        var tokenizer_path: ?[]const u8 = null;
        var safetensors_path: ?[]const u8 = null;
        for (files.array.items) |file| {
            if (file != .object) continue;
            const path_value = file.object.get("path") orelse continue;
            if (path_value != .string) continue;
            const base = std.fs.path.basename(path_value.string);
            if (std.mem.eql(u8, base, "config.json")) config_path = try resolveManifestPath(allocator, repo_root, path_value.string);
            if (std.mem.eql(u8, base, "tokenizer.json")) tokenizer_path = try resolveManifestPath(allocator, repo_root, path_value.string);
            if (std.mem.eql(u8, base, "model.safetensors")) safetensors_path = try resolveManifestPath(allocator, repo_root, path_value.string);
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

fn loadTokenizerFromFile(io: std.Io, allocator: std.mem.Allocator, out: *bolt.tokenizer.Tokenizer, path: []const u8) !void {
    const input = try readReferencedAssetFile(io, allocator, path, "tokenizer json", 32 * 1024 * 1024);
    try out.initFromJson(allocator, input);
}

fn repoRootForManifest(allocator: std.mem.Allocator, manifest_path: []const u8) ![]const u8 {
    return asset_paths.repoRootForManifest(allocator, manifest_path);
}

fn resolveManifestPath(allocator: std.mem.Allocator, repo_root: []const u8, path: []const u8) ![]const u8 {
    return asset_paths.resolveManifestPath(allocator, repo_root, path);
}

fn computeWeightProbe(io: std.Io, allocator: std.mem.Allocator, safetensors_path: []const u8, source_token: u32) !WeightProbe {
    const embedding = try allocator.alloc(f32, hidden_size);
    defer allocator.free(embedding);
    const final_norm = try allocator.alloc(f32, hidden_size);
    defer allocator.free(final_norm);
    const normed = try allocator.alloc(f32, hidden_size);
    defer allocator.free(normed);
    if (try bolt.bonsai_model.readF16TensorSlice(io, allocator, safetensors_path, "model.embed_tokens.weight", @as(usize, source_token) * hidden_size, embedding) != hidden_size) return error.ShortEmbeddingRead;
    if (try bolt.bonsai_model.readF16TensorPrefix(io, allocator, safetensors_path, "model.norm.weight", final_norm) != hidden_size) return error.ShortNormRead;
    try bolt.transformer.rmsNorm(embedding, final_norm, normed, 1e-6);

    var logits: [golden_tokens.len]f32 = undefined;
    var best_index: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    const row = try allocator.alloc(f32, hidden_size);
    defer allocator.free(row);
    for (golden_tokens, 0..) |token, index| {
        if (try bolt.bonsai_model.readF16TensorSlice(io, allocator, safetensors_path, "model.embed_tokens.weight", @as(usize, token) * hidden_size, row) != hidden_size) return error.ShortEmbeddingRead;
        var dot: f32 = 0;
        for (row, normed) |w, x| dot += w * x;
        logits[index] = dot;
        if (dot > best_value) {
            best_value = dot;
            best_index = index;
        }
    }
    return .{ .source_token = source_token, .best_candidate_token = golden_tokens[best_index], .best_candidate_logit = best_value, .candidate_logits = logits };
}

fn computePartialDecodeProbe(io: std.Io, safetensors_path: []const u8, source_token: u32, layer_count: u32) !PartialDecodeProbe {
    if (layer_count == 0 or layer_count > 28) return error.InvalidPartialLayerCount;
    const allocator = std.heap.page_allocator;
    const state = try allocator.alloc(f32, hidden_size);
    defer allocator.free(state);
    if (try bolt.bonsai_model.readF16TensorSlice(io, allocator, safetensors_path, "model.embed_tokens.weight", @as(usize, source_token) * hidden_size, state) != hidden_size) return error.ShortEmbeddingRead;

    const k_cache = try allocator.alloc(f32, @as(usize, layer_count) * kv_dim);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, @as(usize, layer_count) * kv_dim);
    defer allocator.free(v_cache);
    @memset(k_cache, 0);
    @memset(v_cache, 0);
    const dims = bolt.transformer.CpuDecodeDims{ .vocab_size = vocab_size, .hidden_size = hidden_size, .intermediate_size = intermediate_size, .num_query_heads = num_query_heads, .num_kv_heads = num_kv_heads, .head_dim = head_dim, .eps = 1e-6, .rope_theta = 1000000.0 };

    for (0..layer_count) |layer_index_usize| {
        const layer_index: u32 = @intCast(layer_index_usize);
        const layer = try loadLayer(allocator, io, safetensors_path, layer_index);
        const cache_start = layer_index_usize * kv_dim;
        try bolt.transformer.cpuForwardOneLayer(allocator, dims, layer, state, 0, k_cache[cache_start..][0..kv_dim], v_cache[cache_start..][0..kv_dim]);
        freeLayer(allocator, layer);
    }

    var l2: f32 = 0;
    for (state) |value| l2 += value * value;
    const logits = try computeCandidateLogits(allocator, io, safetensors_path, state);
    return .{
        .layers_executed = layer_count,
        .last_layer_index = layer_count - 1,
        .source_token = source_token,
        .output_l2 = @sqrt(l2),
        .output_first4 = .{ state[0], state[1], state[2], state[3] },
        .best_candidate_token = logits.best_candidate_token,
        .best_candidate_logit = logits.best_candidate_logit,
        .candidate_logits = logits.candidate_logits,
    };
}

fn computeGenerationProbe(io: std.Io, safetensors_path: []const u8, prompt_ids: []const u32) !GenerationProbe {
    if (prompt_ids.len == 0) return error.InvalidPrompt;
    const allocator = std.heap.page_allocator;
    const max_positions = prompt_ids.len + golden_tokens.len + 1;
    const k_cache = try allocator.alloc(f32, 28 * max_positions * kv_dim);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, 28 * max_positions * kv_dim);
    defer allocator.free(v_cache);
    @memset(k_cache, 0);
    @memset(v_cache, 0);

    const state = try allocator.alloc(f32, hidden_size);
    defer allocator.free(state);
    const dims = bolt.transformer.CpuDecodeDims{ .vocab_size = vocab_size, .hidden_size = hidden_size, .intermediate_size = intermediate_size, .num_query_heads = num_query_heads, .num_kv_heads = num_kv_heads, .head_dim = head_dim, .eps = 1e-6, .rope_theta = 1000000.0 };
    var model = try loadF16Model(allocator, io, safetensors_path);
    defer model.deinit(allocator);

    var top = TokenLogit{ .token = 0, .logit = -std.math.inf(f32) };
    for (prompt_ids, 0..) |token, position| {
        if (position + 1 == prompt_ids.len) {
            top = try forwardTokenAndTopModel(allocator, &model, dims, token, position, max_positions, k_cache, v_cache, state);
        } else {
            try forwardTokenModel(allocator, &model, dims, token, position, max_positions, k_cache, v_cache, state);
        }
    }

    var generated = [_]u32{0} ** max_generated_tokens;
    var top_logits = [_]f32{0} ** max_generated_tokens;
    var matched: u32 = 0;
    var first_mismatch: i32 = -1;
    var next = top.token;
    var position = prompt_ids.len;
    var count: u32 = 0;
    while (count < max_generated_tokens and count < golden_tokens.len) : (count += 1) {
        generated[count] = next;
        top_logits[count] = top.logit;
        if (next == golden_tokens[count]) {
            matched += 1;
        } else if (first_mismatch < 0) {
            first_mismatch = @intCast(count);
            count += 1;
            break;
        }
        if (count + 1 == golden_tokens.len) {
            count += 1;
            break;
        }
        top = try forwardTokenAndTopModel(allocator, &model, dims, next, position, max_positions, k_cache, v_cache, state);
        next = top.token;
        position += 1;
    }
    return .{
        .prompt_tokens = @intCast(prompt_ids.len),
        .max_tokens = max_generated_tokens,
        .generated_count = count,
        .generated_tokens = generated,
        .top_logits = top_logits,
        .matched_tokens = matched,
        .first_mismatch_index = first_mismatch,
        .matches_reference = count == golden_tokens.len and matched == golden_tokens.len,
    };
}

fn loadF16Model(allocator: std.mem.Allocator, io: std.Io, safetensors_path: []const u8) !F16Model {
    var loaded_layers: usize = 0;
    var model = F16Model{
        .embedding = try loadF16Tensor(allocator, io, safetensors_path, "model.embed_tokens.weight", vocab_size * hidden_size),
        .final_norm = try loadF16Tensor(allocator, io, safetensors_path, "model.norm.weight", hidden_size),
        .layers = try allocator.alloc(bolt.transformer.CpuLayerWeights, model_layer_count),
    };
    errdefer {
        allocator.free(model.embedding);
        allocator.free(model.final_norm);
        for (model.layers[0..loaded_layers]) |layer| freeLayer(allocator, layer);
        allocator.free(model.layers);
    }
    for (0..model_layer_count) |layer_index| {
        model.layers[layer_index] = try loadLayer(allocator, io, safetensors_path, @intCast(layer_index));
        loaded_layers += 1;
    }
    return model;
}

fn forwardTokenModel(
    allocator: std.mem.Allocator,
    model: *F16Model,
    dims: bolt.transformer.CpuDecodeDims,
    token: u32,
    position: usize,
    max_positions: usize,
    k_cache: []f32,
    v_cache: []f32,
    state: []f32,
) !void {
    if (token >= vocab_size) return error.InvalidToken;
    const row_start = @as(usize, token) * hidden_size;
    @memcpy(state, model.embedding[row_start..][0..hidden_size]);
    for (model.layers, 0..) |layer, layer_index| {
        const cache_start = layer_index * max_positions * kv_dim;
        try bolt.transformer.cpuForwardOneLayer(
            allocator,
            dims,
            layer,
            state,
            position,
            k_cache[cache_start..][0 .. max_positions * kv_dim],
            v_cache[cache_start..][0 .. max_positions * kv_dim],
        );
    }
}

fn forwardTokenAndTopModel(
    allocator: std.mem.Allocator,
    model: *F16Model,
    dims: bolt.transformer.CpuDecodeDims,
    token: u32,
    position: usize,
    max_positions: usize,
    k_cache: []f32,
    v_cache: []f32,
    state: []f32,
) !TokenLogit {
    try forwardTokenModel(allocator, model, dims, token, position, max_positions, k_cache, v_cache, state);
    return computeTopLogitModel(allocator, model, state);
}

fn computeTopLogitModel(allocator: std.mem.Allocator, model: *F16Model, state: []const f32) !TokenLogit {
    const normed = try allocator.alloc(f32, hidden_size);
    defer allocator.free(normed);
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);
    try bolt.transformer.rmsNorm(state, model.final_norm, normed, 1e-6);
    bolt.transformer.matVecRows(model.embedding, vocab_size, hidden_size, normed, logits);
    var best = TokenLogit{ .token = 0, .logit = -std.math.inf(f32) };
    for (logits, 0..) |value, index| {
        if (value > best.logit) best = .{ .token = @intCast(index), .logit = value };
    }
    return best;
}

fn forwardTokenAndTop(io: std.Io, allocator: std.mem.Allocator, safetensors_path: []const u8, dims: bolt.transformer.CpuDecodeDims, token: u32, position: usize, max_positions: usize, k_cache: []f32, v_cache: []f32, state: []f32) !TokenLogit {
    if (try bolt.bonsai_model.readF16TensorSlice(io, allocator, safetensors_path, "model.embed_tokens.weight", @as(usize, token) * hidden_size, state) != hidden_size) return error.ShortEmbeddingRead;
    for (0..28) |layer_index_usize| {
        const layer = try loadLayer(allocator, io, safetensors_path, @intCast(layer_index_usize));
        defer freeLayer(allocator, layer);
        const cache_start = layer_index_usize * max_positions * kv_dim;
        try bolt.transformer.cpuForwardOneLayer(
            allocator,
            dims,
            layer,
            state,
            position,
            k_cache[cache_start..][0 .. max_positions * kv_dim],
            v_cache[cache_start..][0 .. max_positions * kv_dim],
        );
    }
    return computeTopLogit(allocator, io, safetensors_path, state);
}

fn computeTopLogit(allocator: std.mem.Allocator, io: std.Io, safetensors_path: []const u8, state: []const f32) !TokenLogit {
    const final_norm = try allocator.alloc(f32, hidden_size);
    defer allocator.free(final_norm);
    const normed = try allocator.alloc(f32, hidden_size);
    defer allocator.free(normed);
    if (try bolt.bonsai_model.readF16TensorPrefix(io, allocator, safetensors_path, "model.norm.weight", final_norm) != hidden_size) return error.ShortNormRead;
    try bolt.transformer.rmsNorm(state, final_norm, normed, 1e-6);

    const location = try bolt.bonsai_model.locateTensor(io, allocator, safetensors_path, "model.embed_tokens.weight");
    const rows_total = location.dims[0];
    const matrix = try allocator.alloc(f32, logits_chunk_rows * hidden_size);
    defer allocator.free(matrix);
    const logits = try allocator.alloc(f32, logits_chunk_rows);
    defer allocator.free(logits);

    var best = TokenLogit{ .token = 0, .logit = -std.math.inf(f32) };
    var row_offset: usize = 0;
    while (row_offset < rows_total) {
        const remaining = rows_total - row_offset;
        const rows: usize = @min(@as(usize, logits_chunk_rows), remaining);
        if (rows == 0) break;
        const value_count = try std.math.mul(usize, rows, @as(usize, hidden_size));
        const values = matrix[0..value_count];
        if (try bolt.bonsai_model.readF16LocationSlice(io, allocator, safetensors_path, location, row_offset * hidden_size, values) != values.len) return error.ShortEmbeddingRead;
        bolt.transformer.matVecRows(values, rows, hidden_size, normed, logits[0..rows]);
        for (logits[0..rows], 0..) |value, local_index| {
            if (value > best.logit) {
                best = .{ .token = @intCast(row_offset + local_index), .logit = value };
            }
        }
        row_offset += rows;
    }
    return best;
}

fn computeCandidateLogits(allocator: std.mem.Allocator, io: std.Io, safetensors_path: []const u8, state: []const f32) !WeightProbe {
    if (state.len != hidden_size) return error.InvalidPartialState;
    const final_norm = try allocator.alloc(f32, hidden_size);
    defer allocator.free(final_norm);
    const normed = try allocator.alloc(f32, hidden_size);
    defer allocator.free(normed);
    if (try bolt.bonsai_model.readF16TensorPrefix(io, allocator, safetensors_path, "model.norm.weight", final_norm) != hidden_size) return error.ShortNormRead;
    try bolt.transformer.rmsNorm(state, final_norm, normed, 1e-6);

    var logits: [golden_tokens.len]f32 = undefined;
    var best_index: usize = 0;
    var best_value: f32 = -std.math.inf(f32);
    const row = try allocator.alloc(f32, hidden_size);
    defer allocator.free(row);
    for (golden_tokens, 0..) |token, index| {
        if (try bolt.bonsai_model.readF16TensorSlice(io, allocator, safetensors_path, "model.embed_tokens.weight", @as(usize, token) * hidden_size, row) != hidden_size) return error.ShortEmbeddingRead;
        var dot: f32 = 0;
        for (row, normed) |w, x| dot += w * x;
        logits[index] = dot;
        if (dot > best_value) {
            best_value = dot;
            best_index = index;
        }
    }
    return .{ .source_token = 0, .best_candidate_token = golden_tokens[best_index], .best_candidate_logit = best_value, .candidate_logits = logits };
}

fn loadLayer(allocator: std.mem.Allocator, io: std.Io, safetensors_path: []const u8, layer_index: u32) !bolt.transformer.CpuLayerWeights {
    var input_norm_name: [96]u8 = undefined;
    var post_norm_name: [96]u8 = undefined;
    var q_norm_name: [96]u8 = undefined;
    var k_norm_name: [96]u8 = undefined;
    var q_proj_name: [96]u8 = undefined;
    var k_proj_name: [96]u8 = undefined;
    var v_proj_name: [96]u8 = undefined;
    var o_proj_name: [96]u8 = undefined;
    var gate_proj_name: [96]u8 = undefined;
    var up_proj_name: [96]u8 = undefined;
    var down_proj_name: [96]u8 = undefined;
    return .{
        .input_norm = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&input_norm_name, "model.layers.{d}.input_layernorm.weight", .{layer_index}), hidden_size),
        .post_norm = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&post_norm_name, "model.layers.{d}.post_attention_layernorm.weight", .{layer_index}), hidden_size),
        .q_norm = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&q_norm_name, "model.layers.{d}.self_attn.q_norm.weight", .{layer_index}), head_dim),
        .k_norm = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&k_norm_name, "model.layers.{d}.self_attn.k_norm.weight", .{layer_index}), head_dim),
        .q_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&q_proj_name, "model.layers.{d}.self_attn.q_proj.weight", .{layer_index}), hidden_size * hidden_size),
        .k_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&k_proj_name, "model.layers.{d}.self_attn.k_proj.weight", .{layer_index}), kv_dim * hidden_size),
        .v_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&v_proj_name, "model.layers.{d}.self_attn.v_proj.weight", .{layer_index}), kv_dim * hidden_size),
        .o_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&o_proj_name, "model.layers.{d}.self_attn.o_proj.weight", .{layer_index}), hidden_size * hidden_size),
        .gate_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&gate_proj_name, "model.layers.{d}.mlp.gate_proj.weight", .{layer_index}), intermediate_size * hidden_size),
        .up_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&up_proj_name, "model.layers.{d}.mlp.up_proj.weight", .{layer_index}), intermediate_size * hidden_size),
        .down_proj = try loadF16Tensor(allocator, io, safetensors_path, try std.fmt.bufPrint(&down_proj_name, "model.layers.{d}.mlp.down_proj.weight", .{layer_index}), hidden_size * intermediate_size),
    };
}

fn loadF16Tensor(allocator: std.mem.Allocator, io: std.Io, path: []const u8, name: []const u8, count: usize) ![]f32 {
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);
    if (try bolt.bonsai_model.readF16TensorPrefix(io, allocator, path, name, values) != count) return error.ShortTensorRead;
    return values;
}

fn freeLayer(allocator: std.mem.Allocator, layer: bolt.transformer.CpuLayerWeights) void {
    allocator.free(layer.input_norm);
    allocator.free(layer.post_norm);
    allocator.free(layer.q_norm);
    allocator.free(layer.k_norm);
    allocator.free(layer.q_proj);
    allocator.free(layer.k_proj);
    allocator.free(layer.v_proj);
    allocator.free(layer.o_proj);
    allocator.free(layer.gate_proj);
    allocator.free(layer.up_proj);
    allocator.free(layer.down_proj);
}

fn writePartialDecodeProbe(writer: anytype, probe: PartialDecodeProbe) !void {
    try writer.print("{{\"layers_executed\":{d},\"last_layer_index\":{d},\"source_token\":{d},\"output_l2\":{d},\"output_first4\":[{d},{d},{d},{d}],\"best_candidate_token\":{d},\"best_candidate_logit\":{d},\"candidate_logits\":[", .{ probe.layers_executed, probe.last_layer_index, probe.source_token, probe.output_l2, probe.output_first4[0], probe.output_first4[1], probe.output_first4[2], probe.output_first4[3], probe.best_candidate_token, probe.best_candidate_logit });
    for (golden_tokens, 0..) |token, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{{\"token\":{d},\"logit\":{d}}}", .{ token, probe.candidate_logits[index] });
    }
    try writer.writeAll("]}");
}

fn writeGenerationProbe(writer: anytype, probe: GenerationProbe) !void {
    try writer.print("{{\"prompt_tokens\":{d},\"max_tokens\":{d},\"generated_count\":{d},\"matched_tokens\":{d},\"first_mismatch_index\":{d},\"matches_reference\":{s},\"generated_tokens\":", .{ probe.prompt_tokens, probe.max_tokens, probe.generated_count, probe.matched_tokens, probe.first_mismatch_index, if (probe.matches_reference) "true" else "false" });
    try writeIds(writer, probe.generated_tokens[0..probe.generated_count]);
    try writer.writeAll(",\"top_logits\":[");
    for (0..probe.generated_count) |index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{probe.top_logits[index]});
    }
    try writer.writeAll("]}");
}

fn writeWeightProbe(writer: anytype, probe: WeightProbe) !void {
    try writer.print("{{\"source_token\":{d},\"best_candidate_token\":{d},\"best_candidate_logit\":{d},\"candidate_logits\":[", .{ probe.source_token, probe.best_candidate_token, probe.best_candidate_logit });
    for (golden_tokens, 0..) |token, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{{\"token\":{d},\"logit\":{d}}}", .{ token, probe.candidate_logits[index] });
    }
    try writer.writeAll("]}");
}

fn writeIds(writer: anytype, ids: []const u32) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeReportBody(writer: anytype, meta: bolt.runtime.artifact_metadata.Metadata, manifest_digest: []const u8, artifact_type: []const u8, manifest_path: []const u8, paths: BonsaiPaths, report: bolt.bonsai_model.BonsaiTensorReport, prompt_ids: []const u32, probe: WeightProbe, layer_probe: PartialDecodeProbe, generation_probe: GenerationProbe, readiness_pass: bool, gate_pass: bool, gate: []const u8, status: []const u8) !void {
    try writer.print(
        "{{\n" ++
            "  \"schema_version\": \"{s}\",\n" ++
            "  \"artifact_type\": \"{s}\",\n" ++
            "  \"status\": \"{s}\",\n" ++
            "  \"command\": \"{s}\",\n" ++
            "  \"cwd\": \"{s}\",\n" ++
            "  \"git_commit\": \"{s}\",\n" ++
            "  \"timestamp_utc\": \"{s}\",\n" ++
            "  \"toolchain\": {{\"zig\":\"{s}\"}},\n" ++
            "  \"manifest_digest\": \"{s}\",\n" ++
            "  \"gate\": \"{s}\",\n" ++
            "  \"acceptance\": \"real_unquantized_transformer_golden_{s}\",\n" ++
            "  \"manifest_path\": \"{s}\",\n" ++
            "  \"config_path\": \"{s}\",\n" ++
            "  \"tokenizer_path\": \"{s}\",\n" ++
            "  \"safetensors_path\": \"{s}\",\n" ++
            "  \"readiness_pass\": {s},\n" ++
            "  \"readiness\": {{\"passed\":{s}}},\n" ++
            "  \"smoke\": {{\"passed\":{s}}},\n" ++
            "  \"tensor_contract\": {{\"expected\":{d},\"matched\":{d},\"missing\":{d},\"invalid\":{d},\"extra\":{d},\"tensor_count\":{d}}},\n" ++
            "  \"prompt\": \"{s}\",\n" ++
            "  \"prompt_token_ids\": ",
        .{
            bolt.runtime.artifact_metadata.schema_version,
            artifact_type,
            status,
            meta.command,
            meta.cwd,
            meta.git_commit,
            meta.timestamp_utc,
            meta.zig_version,
            manifest_digest,
            gate,
            if (gate_pass) "passed" else "blocked",
            manifest_path,
            paths.config_path,
            paths.tokenizer_path,
            paths.safetensors_path,
            if (readiness_pass) "true" else "false",
            if (gate_pass) "true" else "false",
            if (gate_pass) "true" else "false",
            report.expected_tensors,
            report.matched_tensors,
            report.missing_tensors,
            report.invalid_tensors,
            report.extra_tensors,
            report.tensor_count,
            golden_prompt,
        },
    );
    try writeIds(writer, prompt_ids);
    try writer.writeAll(",\n  \"real_weight_probe\": ");
    try writeWeightProbe(writer, probe);
    try writer.writeAll(",\n  \"real_partial_decode_probe\": ");
    try writePartialDecodeProbe(writer, layer_probe);
    try writer.writeAll(",\n  \"real_generation_probe\": ");
    try writeGenerationProbe(writer, generation_probe);
    try writer.writeAll(",\n  \"reference_golden_tokens\": ");
    try writeIds(writer, &golden_tokens);
    if (gate_pass) {
        try writer.writeAll(",\n  \"missing_required_surface\": []\n}\n");
    } else {
        try writer.writeAll(",\n  \"missing_required_surface\": [\"selected-token/logit parity\"]\n}\n");
    }
}

fn writeReadiness(init: std.process.Init, manifest_path: []const u8, paths: BonsaiPaths, report: bolt.bonsai_model.BonsaiTensorReport, tok: bolt.tokenizer.Tokenizer, prompt_ids: []const u32, probe: WeightProbe, layer_probe: PartialDecodeProbe, generation_probe: GenerationProbe, readiness_pass: bool, gate_pass: bool) !void {
    _ = tok;
    const allocator = init.arena.allocator();
    const meta = try bolt.runtime.artifact_metadata.capture(allocator, init.io, "zig build run-bonsai-golden --summary all");
    const manifest_digest = try bolt.runtime.artifact_metadata.fileSha256Hex(init.io, allocator, manifest_path);
    try ensureParentDir(init.io, readiness_artifact_path);
    var file = try std.Io.Dir.cwd().createFile(init.io, readiness_artifact_path, .{ .truncate = true });
    defer file.close(init.io);
    var buf: [16384]u8 = undefined;
    var fw = file.writer(init.io, &buf);
    try writeReportBody(&fw.interface, meta, manifest_digest, "engine.bonsai.readiness", manifest_path, paths, report, prompt_ids, probe, layer_probe, generation_probe, readiness_pass, gate_pass, "bolt-bonsai-readiness", if (gate_pass) "pass" else "blocked");
    try fw.interface.flush();
}

fn writeBlocker(init: std.process.Init, manifest_path: []const u8, paths: BonsaiPaths, report: bolt.bonsai_model.BonsaiTensorReport, prompt_ids: []const u32, probe: WeightProbe, layer_probe: PartialDecodeProbe, generation_probe: GenerationProbe, readiness_pass: bool, gate_pass: bool) !void {
    if (gate_pass) {
        std.Io.Dir.cwd().deleteFile(init.io, blocker_artifact_path) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
        return;
    }
    const allocator = init.arena.allocator();
    const meta = try bolt.runtime.artifact_metadata.capture(allocator, init.io, "zig build run-bonsai-golden --summary all");
    const manifest_digest = try bolt.runtime.artifact_metadata.fileSha256Hex(init.io, allocator, manifest_path);
    try ensureParentDir(init.io, blocker_artifact_path);
    var file = try std.Io.Dir.cwd().createFile(init.io, blocker_artifact_path, .{ .truncate = true });
    defer file.close(init.io);
    var buf: [16384]u8 = undefined;
    var fw = file.writer(init.io, &buf);
    try writeReportBody(&fw.interface, meta, manifest_digest, "engine.bonsai.smoke", manifest_path, paths, report, prompt_ids, probe, layer_probe, generation_probe, readiness_pass, gate_pass, "bolt-bonsai-golden", if (gate_pass) "pass" else "blocked");
    try fw.interface.flush();
}

fn writeStdout(init: std.process.Init, manifest_path: []const u8, paths: BonsaiPaths, report: bolt.bonsai_model.BonsaiTensorReport, prompt_ids: []const u32, probe: WeightProbe, layer_probe: PartialDecodeProbe, generation_probe: GenerationProbe, readiness_pass: bool, gate_pass: bool) !void {
    const allocator = init.arena.allocator();
    const meta = try bolt.runtime.artifact_metadata.capture(allocator, init.io, "zig build run-bonsai-golden --summary all");
    const manifest_digest = try bolt.runtime.artifact_metadata.fileSha256Hex(init.io, allocator, manifest_path);
    var stdout_buffer: [16384]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try writeReportBody(stdout, meta, manifest_digest, "engine.bonsai.smoke", manifest_path, paths, report, prompt_ids, probe, layer_probe, generation_probe, readiness_pass, gate_pass, "bolt-bonsai-golden", if (gate_pass) "pass" else "blocked");
    try stdout.flush();
}
