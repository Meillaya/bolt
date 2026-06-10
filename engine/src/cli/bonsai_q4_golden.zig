const std = @import("std");
const bolt = @import("bolt");
const asset_paths = @import("asset_paths.zig");

const default_manifest_path = "../artifacts/assets/bolt-parity/asset-manifest.json";
const artifact_path = "../artifacts/bolt-q4-golden.json";
const golden_prompt = "The capital of France is";
const hidden_size: usize = 2048;
const intermediate_size: usize = 6144;
const num_query_heads: usize = 16;
const num_kv_heads: usize = 8;
const head_dim: usize = 128;
const kv_dim: usize = num_kv_heads * head_dim;
const vocab_size: usize = 151936;
const layer_count: usize = 28;
const max_generated_tokens: usize = 20;
const logits_chunk_rows: usize = 1024;
const golden_tokens = [_]u32{ 198, 279, 198, 279, 6722, 315, 315, 9625, 374, 198, 6722, 315, 315, 9625, 374, 198, 279, 6722, 315, 279 };

const Paths = struct { config_path: []const u8, tokenizer_path: []const u8, safetensors_path: []const u8 };
const TokenLogit = struct { token: u32, logit: f32 };
const GenerationProbe = struct {
    prompt_tokens: u32,
    generated_count: u32,
    generated_tokens: [max_generated_tokens]u32,
    top_logits: [max_generated_tokens]f32,
    matched_tokens: u32,
    first_mismatch_index: i32,
    matches_reference: bool,
    metal_decode_used: bool,
    metal_projection_dispatches: u32,
    metal_logits_dispatches: u32,
};

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
            std.debug.print("missing referenced {s}: {s}\nPrepare the real Q4 asset listed in the manifest, or update the manifest before rerunning run-bonsai-q4-golden.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
}

fn requireReferencedAssetFile(io: std.Io, path: []const u8, label: []const u8) !void {
    const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing referenced {s}: {s}\nPrepare the real Q4 asset listed in the manifest, or update the manifest before rerunning run-bonsai-q4-golden.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
    if (stat.kind != .file) {
        std.debug.print("referenced {s} is not a file: {s}\nUpdate the manifest to point at the prepared real asset before rerunning run-bonsai-q4-golden.\n", .{ label, path });
        return error.MissingRequiredFile;
    }
}

fn missingAssetEntry(asset_id: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest is missing required asset entry: {s}\nPrepare the real asset files and add this asset entry to the manifest before rerunning run-bonsai-q4-golden.\n", .{asset_id});
    return error.MissingRequiredFile;
}

fn missingManifestEntry(asset_id: []const u8, expected_file: []const u8) error{MissingRequiredFile} {
    std.debug.print("asset manifest entry {s} is missing required file path for {s}\nAdd that file entry after preparing real Q4 assets, then rerun run-bonsai-q4-golden.\n", .{ asset_id, expected_file });
    return error.MissingRequiredFile;
}

const MetalQ4Probe = struct {
    available: bool,
    used: bool,
    kernel: []const u8,
    tensor: []const u8,
    tensor_count: u32,
    rows: u32,
    cols: u32,
    group_size: u32,
    dispatches: u32,
    max_abs_error: f32,
    pass: bool,
};

const Q4ProjectionSpec = struct { suffix: []const u8, rows: usize, cols: usize };

pub const Q4DecodeBenchmark = struct {
    prompt_tokens: u32,
    generated_tokens: u32,
    matches_reference: bool,
    elapsed_ns: u64,
    q4_projection_dispatches: u32,
    q4_logits_dispatches: u32,
};

const Q4Tensor = struct {
    weight: bolt.bonsai_model.BonsaiTensorLocation,
    scales: bolt.bonsai_model.BonsaiTensorLocation,
    biases: bolt.bonsai_model.BonsaiTensorLocation,
    rows: usize,
    cols: usize,
    groups: usize,
    words_per_row: usize,
    group_size: usize,
};

const Q4MetalLayer = struct {
    input_norm: []f32,
    post_norm: []f32,
    q_norm: []f32,
    k_norm: []f32,
    q_proj: bolt.metal.context.Q4Buffer,
    k_proj: bolt.metal.context.Q4Buffer,
    v_proj: bolt.metal.context.Q4Buffer,
    o_proj: bolt.metal.context.Q4Buffer,
    gate_proj: bolt.metal.context.Q4Buffer,
    up_proj: bolt.metal.context.Q4Buffer,
    down_proj: bolt.metal.context.Q4Buffer,
};

const Q4Model = struct {
    embedding: bolt.metal.context.Q4Buffer,
    embedding_location: Q4Tensor,
    final_norm: []f32,
    layers: []Q4MetalLayer,

    fn deinit(self: *Q4Model, allocator: std.mem.Allocator) void {
        self.embedding.deinit();
        allocator.free(self.final_norm);
        for (self.layers) |*layer| freeLayer(allocator, layer);
        allocator.free(self.layers);
    }
};

const MetalDecodeStats = struct {
    available: bool = false,
    used: bool = false,
    projection_dispatches: u32 = 0,
    logits_dispatches: u32 = 0,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = if (args.len > 1) args[1] else default_manifest_path;
    const manifest_input = try readRequiredFile(init.io, allocator, manifest_path, "asset manifest");
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);
    const paths = try findQ4Paths(allocator, repo_root, parsed.value);
    try requireReferencedAssetFile(init.io, paths.config_path, "model config");
    try requireReferencedAssetFile(init.io, paths.safetensors_path, "safetensors model");

    var tok: bolt.tokenizer.Tokenizer = undefined;
    try loadTokenizerFromFile(init.io, allocator, &tok, paths.tokenizer_path);
    defer tok.deinit();
    var prompt_ids: [256]u32 = undefined;
    const prompt_count = try tok.applyChatTemplate(golden_prompt, &prompt_ids);
    const tokenizer_pass = prompt_count > 0 and tok.eos_token_id == 151643 and tok.im_start_token_id == 151644 and tok.im_end_token_id == 151645;
    const tensor_pass = try validateQ4TensorSurface(init.io, allocator, paths.safetensors_path);
    const generation = try computeGenerationProbe(init.io, paths.safetensors_path, prompt_ids[0..prompt_count]);
    const metal_probe = try runMetalQ4Probe(init.io, allocator, paths.safetensors_path);
    const gate_pass = tokenizer_pass and tensor_pass and generation.matches_reference and metal_probe.pass;

    try writeArtifact(init, manifest_path, paths, prompt_ids[0..prompt_count], generation, metal_probe, tokenizer_pass, tensor_pass, gate_pass);
    if (!gate_pass) return error.RealQ4DecodeMismatch;
}

pub fn runDecodeBenchmark(io: std.Io, allocator: std.mem.Allocator, manifest_path: []const u8) !Q4DecodeBenchmark {
    const manifest_input = try readRequiredFile(io, allocator, manifest_path, "asset manifest");
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const repo_root = try repoRootForManifest(allocator, manifest_path);
    const paths = try findQ4Paths(allocator, repo_root, parsed.value);
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
        .q4_projection_dispatches = generation.metal_projection_dispatches,
        .q4_logits_dispatches = generation.metal_logits_dispatches,
    };
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        try std.Io.Dir.cwd().createDirPath(io, parent);
    }
}

fn computeGenerationProbe(io: std.Io, safetensors_path: []const u8, prompt_ids: []const u32) !GenerationProbe {
    if (prompt_ids.len == 0) return error.InvalidPrompt;
    if (!bolt.metal.context.Context.isAvailable()) return error.MetalUnavailable;
    const allocator = std.heap.page_allocator;
    const max_positions = prompt_ids.len + golden_tokens.len + 1;
    const k_cache = try allocator.alloc(f32, layer_count * max_positions * kv_dim);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, layer_count * max_positions * kv_dim);
    defer allocator.free(v_cache);
    @memset(k_cache, 0);
    @memset(v_cache, 0);
    const state = try allocator.alloc(f32, hidden_size);
    defer allocator.free(state);
    var context = try bolt.metal.context.Context.init();
    defer context.deinit();
    var model = try loadQ4Model(allocator, io, safetensors_path, &context);
    defer model.deinit(allocator);
    var stats = MetalDecodeStats{ .available = true, .used = true };
    const dims = bolt.transformer.CpuDecodeDims{ .vocab_size = vocab_size, .hidden_size = hidden_size, .intermediate_size = intermediate_size, .num_query_heads = num_query_heads, .num_kv_heads = num_kv_heads, .head_dim = head_dim, .eps = 1e-6, .rope_theta = 1000000.0 };

    var top = TokenLogit{ .token = 0, .logit = -std.math.inf(f32) };
    for (prompt_ids, 0..) |token, pos| top = try forwardTokenAndTop(allocator, io, safetensors_path, &context, &model, dims, token, pos, max_positions, k_cache, v_cache, state, &stats);

    var generated = [_]u32{0} ** max_generated_tokens;
    var top_logits = [_]f32{0} ** max_generated_tokens;
    var matched: u32 = 0;
    var first_mismatch: i32 = -1;
    var next = top.token;
    var pos = prompt_ids.len;
    var count: u32 = 0;
    while (count < max_generated_tokens) : (count += 1) {
        generated[count] = next;
        top_logits[count] = top.logit;
        if (next == golden_tokens[count]) matched += 1 else if (first_mismatch < 0) first_mismatch = @intCast(count);
        top = try forwardTokenAndTop(allocator, io, safetensors_path, &context, &model, dims, next, pos, max_positions, k_cache, v_cache, state, &stats);
        next = top.token;
        pos += 1;
    }
    return .{
        .prompt_tokens = @intCast(prompt_ids.len),
        .generated_count = count,
        .generated_tokens = generated,
        .top_logits = top_logits,
        .matched_tokens = matched,
        .first_mismatch_index = first_mismatch,
        .matches_reference = matched == golden_tokens.len,
        .metal_decode_used = stats.used,
        .metal_projection_dispatches = stats.projection_dispatches,
        .metal_logits_dispatches = stats.logits_dispatches,
    };
}

fn loadQ4Model(allocator: std.mem.Allocator, io: std.Io, path: []const u8, context: *bolt.metal.context.Context) !Q4Model {
    var loaded_layers: usize = 0;
    const embedding_location = try locateQ4Tensor(allocator, io, path, "model.embed_tokens", vocab_size, hidden_size);
    var model = Q4Model{
        .embedding = try loadQ4Buffer(allocator, io, path, context, embedding_location),
        .embedding_location = embedding_location,
        .final_norm = try loadBf16(allocator, io, path, "model.norm.weight", hidden_size),
        .layers = try allocator.alloc(Q4MetalLayer, layer_count),
    };
    errdefer {
        model.embedding.deinit();
        allocator.free(model.final_norm);
        for (model.layers[0..loaded_layers]) |*layer| freeLayer(allocator, layer);
        allocator.free(model.layers);
    }
    for (0..layer_count) |layer_index| {
        model.layers[layer_index] = try loadQ4Layer(allocator, io, path, context, @intCast(layer_index));
        loaded_layers += 1;
    }
    return model;
}

fn forwardTokenAndTop(allocator: std.mem.Allocator, io: std.Io, path: []const u8, context: *bolt.metal.context.Context, model: *Q4Model, dims: bolt.transformer.CpuDecodeDims, token: u32, position: usize, max_positions: usize, k_cache: []f32, v_cache: []f32, state: []f32, stats: *MetalDecodeStats) !TokenLogit {
    try dequantQ4TensorRows(allocator, io, path, model.embedding_location, token, 1, state, true);
    for (model.layers, 0..) |*layer, layer_index| {
        const cache_start = layer_index * max_positions * kv_dim;
        try metalForwardOneLayerQ4Approx(allocator, context, dims, layer, state, position, k_cache[cache_start..][0 .. max_positions * kv_dim], v_cache[cache_start..][0 .. max_positions * kv_dim], stats);
    }
    return computeTopLogit(allocator, context, model, state, stats);
}

fn computeTopLogit(allocator: std.mem.Allocator, context: *bolt.metal.context.Context, model: *Q4Model, state: []const f32, stats: *MetalDecodeStats) !TokenLogit {
    const normed = try allocator.alloc(f32, hidden_size);
    defer allocator.free(normed);
    try bolt.transformer.rmsNorm(state, model.final_norm, normed, 1e-6);
    bolt.transformer.roundSliceBf16ThenF16(normed);
    const logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits);
    try q4MatVec(context, model.embedding, normed, logits);
    stats.logits_dispatches += 1;
    var best = TokenLogit{ .token = 0, .logit = -std.math.inf(f32) };
    for (logits) |*value| value.* = bolt.transformer.bf16Round(value.*);
    for (logits, 0..) |value, i| {
        if (value > best.logit) best = .{ .token = @intCast(i), .logit = value };
    }
    return best;
}

fn metalForwardOneLayerQ4Approx(allocator: std.mem.Allocator, context: *bolt.metal.context.Context, d: bolt.transformer.CpuDecodeDims, layer: *Q4MetalLayer, state: []f32, position: usize, k_cache: []f32, v_cache: []f32, stats: *MetalDecodeStats) !void {
    try d.validate();
    if (state.len != d.hidden_size) return error.InvalidCpuDecodeShape;
    const kvd = d.kvDim();
    const qd = d.queryDim();
    if (k_cache.len < (position + 1) * kvd or v_cache.len < (position + 1) * kvd) return error.InvalidCpuDecodeShape;

    const norm = try allocator.alloc(f32, d.hidden_size);
    defer allocator.free(norm);
    const q = try allocator.alloc(f32, qd);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kvd);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kvd);
    defer allocator.free(v);
    const attn = try allocator.alloc(f32, qd);
    defer allocator.free(attn);
    const proj = try allocator.alloc(f32, d.hidden_size);
    defer allocator.free(proj);
    const gate = try allocator.alloc(f32, d.intermediate_size);
    defer allocator.free(gate);
    const up = try allocator.alloc(f32, d.intermediate_size);
    defer allocator.free(up);
    const mlp = try allocator.alloc(f32, d.hidden_size);
    defer allocator.free(mlp);

    try bolt.transformer.rmsNorm(state, layer.input_norm, norm, d.eps);
    bolt.transformer.roundSliceBf16ThenF16(norm);
    try q4MatVec(context, layer.q_proj, norm, q);
    try q4MatVec(context, layer.k_proj, norm, k);
    try q4MatVec(context, layer.v_proj, norm, v);
    stats.projection_dispatches += 3;
    bolt.transformer.roundSliceBf16ThenF16(q);
    bolt.transformer.roundSliceBf16ThenF16(k);
    bolt.transformer.roundSliceBf16ThenF16(v);
    try bolt.transformer.normalizeHeads(q, d.num_query_heads, d.head_dim, layer.q_norm, d.eps);
    try bolt.transformer.normalizeHeads(k, d.num_kv_heads, d.head_dim, layer.k_norm, d.eps);
    try bolt.transformer.applyRoPEGrouped(q, d.num_query_heads, d.head_dim, @intCast(position), d.rope_theta);
    try bolt.transformer.applyRoPEGrouped(k, d.num_kv_heads, d.head_dim, @intCast(position), d.rope_theta);
    bolt.transformer.roundSliceF16(k);
    @memcpy(k_cache[position * kvd ..][0..kvd], k);
    @memcpy(v_cache[position * kvd ..][0..kvd], v);
    try bolt.transformer.cpuGqaAttention(d, q, k_cache, v_cache, position + 1, attn, allocator);
    bolt.transformer.roundSliceF16(attn);
    try q4MatVec(context, layer.o_proj, attn, proj);
    stats.projection_dispatches += 1;
    bolt.transformer.roundSliceBf16(proj);
    for (state, proj) |*s, value| s.* += value;
    try bolt.transformer.rmsNorm(state, layer.post_norm, norm, d.eps);
    bolt.transformer.roundSliceBf16ThenF16(norm);
    try q4MatVec(context, layer.gate_proj, norm, gate);
    try q4MatVec(context, layer.up_proj, norm, up);
    stats.projection_dispatches += 2;
    for (gate, up) |*g, u| g.* = bolt.transformer.bf16ThenF16Round(bolt.transformer.silu(g.*) * u);
    try q4MatVec(context, layer.down_proj, gate, mlp);
    stats.projection_dispatches += 1;
    bolt.transformer.roundSliceBf16(mlp);
    for (state, mlp) |*s, value| s.* += value;
}

fn q4MatVec(context: *bolt.metal.context.Context, q4: bolt.metal.context.Q4Buffer, input: []const f32, out: []f32) !void {
    var input_buffer = try context.createSharedBufferF32(input.len);
    defer input_buffer.deinit();
    try input_buffer.write(input);
    var output_buffer = try context.createSharedBufferF32(out.len);
    defer output_buffer.deinit();
    try context.q4mvSharedBufferF32(q4, input_buffer, output_buffer);
    @memcpy(out, output_buffer.asSlice());
}

fn loadQ4Layer(allocator: std.mem.Allocator, io: std.Io, path: []const u8, context: *bolt.metal.context.Context, layer_index: u32) !Q4MetalLayer {
    var n0: [128]u8 = undefined;
    var n1: [128]u8 = undefined;
    var n2: [128]u8 = undefined;
    var n3: [128]u8 = undefined;
    var n4: [128]u8 = undefined;
    var n5: [128]u8 = undefined;
    var n6: [128]u8 = undefined;
    var n7: [128]u8 = undefined;
    var n8: [128]u8 = undefined;
    var n9: [128]u8 = undefined;
    var n10: [128]u8 = undefined;
    const layer = Q4MetalLayer{
        .input_norm = try loadBf16(allocator, io, path, try std.fmt.bufPrint(&n0, "model.layers.{d}.input_layernorm.weight", .{layer_index}), hidden_size),
        .post_norm = try loadBf16(allocator, io, path, try std.fmt.bufPrint(&n1, "model.layers.{d}.post_attention_layernorm.weight", .{layer_index}), hidden_size),
        .q_norm = try loadBf16(allocator, io, path, try std.fmt.bufPrint(&n2, "model.layers.{d}.self_attn.q_norm.weight", .{layer_index}), head_dim),
        .k_norm = try loadBf16(allocator, io, path, try std.fmt.bufPrint(&n3, "model.layers.{d}.self_attn.k_norm.weight", .{layer_index}), head_dim),
        .q_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n4, "model.layers.{d}.self_attn.q_proj", .{layer_index}), hidden_size, hidden_size),
        .k_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n5, "model.layers.{d}.self_attn.k_proj", .{layer_index}), kv_dim, hidden_size),
        .v_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n6, "model.layers.{d}.self_attn.v_proj", .{layer_index}), kv_dim, hidden_size),
        .o_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n7, "model.layers.{d}.self_attn.o_proj", .{layer_index}), hidden_size, hidden_size),
        .gate_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n8, "model.layers.{d}.mlp.gate_proj", .{layer_index}), intermediate_size, hidden_size),
        .up_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n9, "model.layers.{d}.mlp.up_proj", .{layer_index}), intermediate_size, hidden_size),
        .down_proj = try loadQ4BufferByName(allocator, io, path, context, try std.fmt.bufPrint(&n10, "model.layers.{d}.mlp.down_proj", .{layer_index}), hidden_size, intermediate_size),
    };
    return layer;
}

fn loadBf16(allocator: std.mem.Allocator, io: std.Io, path: []const u8, name: []const u8, count: usize) ![]f32 {
    const out = try allocator.alloc(f32, count);
    errdefer allocator.free(out);
    if (try bolt.bonsai_model.readBf16TensorPrefix(io, allocator, path, name, out) != count) return error.ShortTensorRead;
    return out;
}

fn loadQ4BufferByName(allocator: std.mem.Allocator, io: std.Io, path: []const u8, context: *bolt.metal.context.Context, base: []const u8, rows: usize, cols: usize) !bolt.metal.context.Q4Buffer {
    const tensor = try locateQ4Tensor(allocator, io, path, base, rows, cols);
    return loadQ4Buffer(allocator, io, path, context, tensor);
}

fn loadQ4Buffer(allocator: std.mem.Allocator, io: std.Io, path: []const u8, context: *bolt.metal.context.Context, tensor: Q4Tensor) !bolt.metal.context.Q4Buffer {
    var q4 = try context.createQ4Buffer(tensor.rows, tensor.cols, tensor.group_size);
    errdefer q4.deinit();
    const packed_words = try allocator.alloc(u32, tensor.rows * tensor.words_per_row);
    defer allocator.free(packed_words);
    const scales_raw = try allocator.alloc(u16, tensor.rows * tensor.groups);
    defer allocator.free(scales_raw);
    const biases_raw = try allocator.alloc(u16, tensor.rows * tensor.groups);
    defer allocator.free(biases_raw);
    if (try bolt.bonsai_model.readU32LocationSlice(io, allocator, path, tensor.weight, 0, packed_words) != packed_words.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16RawLocationSlice(io, allocator, path, tensor.scales, 0, scales_raw) != scales_raw.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16RawLocationSlice(io, allocator, path, tensor.biases, 0, biases_raw) != biases_raw.len) return error.ShortTensorRead;
    try q4.packed_nibbles.write(std.mem.sliceAsBytes(packed_words));
    try q4.scales_bf16.writeRaw(scales_raw);
    try q4.biases_bf16.writeRaw(biases_raw);
    return q4;
}

fn freeLayer(allocator: std.mem.Allocator, l: *Q4MetalLayer) void {
    allocator.free(l.input_norm);
    allocator.free(l.post_norm);
    allocator.free(l.q_norm);
    allocator.free(l.k_norm);
    l.q_proj.deinit();
    l.k_proj.deinit();
    l.v_proj.deinit();
    l.o_proj.deinit();
    l.gate_proj.deinit();
    l.up_proj.deinit();
    l.down_proj.deinit();
}

fn locateQ4Tensor(allocator: std.mem.Allocator, io: std.Io, path: []const u8, base: []const u8, rows: usize, cols: usize) !Q4Tensor {
    if (cols % 8 != 0) return error.InvalidQ4Shape;
    const words_per_row = cols / 8;
    const weight_name = try std.fmt.allocPrint(allocator, "{s}.weight", .{base});
    defer allocator.free(weight_name);
    const scale_name = try std.fmt.allocPrint(allocator, "{s}.scales", .{base});
    defer allocator.free(scale_name);
    const bias_name = try std.fmt.allocPrint(allocator, "{s}.biases", .{base});
    defer allocator.free(bias_name);
    const w_loc = try bolt.bonsai_model.locateTensor(io, allocator, path, weight_name);
    const s_loc = try bolt.bonsai_model.locateTensor(io, allocator, path, scale_name);
    const b_loc = try bolt.bonsai_model.locateTensor(io, allocator, path, bias_name);
    if (w_loc.dtype != .u32 or s_loc.dtype != .bf16 or b_loc.dtype != .bf16) return error.InvalidQ4Dtype;
    if (w_loc.dims[0] != rows or s_loc.dims[0] != rows or b_loc.dims[0] != rows) return error.InvalidQ4Shape;
    const groups = s_loc.dims[1];
    const group_size = try bolt.q4.inferGroupSize(cols, groups);
    if (w_loc.dims[1] != words_per_row or b_loc.dims[1] != groups) return error.InvalidQ4Shape;
    return .{ .weight = w_loc, .scales = s_loc, .biases = b_loc, .rows = rows, .cols = cols, .groups = groups, .words_per_row = words_per_row, .group_size = group_size };
}

fn dequantQ4TensorRows(allocator: std.mem.Allocator, io: std.Io, path: []const u8, tensor: Q4Tensor, row_offset: u32, row_count: usize, out: []f32, bf16_round_output: bool) !void {
    if (@as(usize, row_offset) + row_count > tensor.rows or out.len != row_count * tensor.cols) return error.InvalidQ4Shape;
    const packed_words = try allocator.alloc(u32, row_count * tensor.words_per_row);
    defer allocator.free(packed_words);
    const scales = try allocator.alloc(f32, row_count * tensor.groups);
    defer allocator.free(scales);
    const biases = try allocator.alloc(f32, row_count * tensor.groups);
    defer allocator.free(biases);
    if (try bolt.bonsai_model.readU32LocationSlice(io, allocator, path, tensor.weight, @as(usize, row_offset) * tensor.words_per_row, packed_words) != packed_words.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16LocationSlice(io, allocator, path, tensor.scales, @as(usize, row_offset) * tensor.groups, scales) != scales.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16LocationSlice(io, allocator, path, tensor.biases, @as(usize, row_offset) * tensor.groups, biases) != biases.len) return error.ShortTensorRead;
    try bolt.q4.dequantizeRows(.{
        .packed_words = packed_words,
        .scales = scales,
        .biases = biases,
        .rows = row_count,
        .cols = tensor.cols,
        .group_size = tensor.group_size,
    }, out, bf16_round_output);
}

fn validateQ4TensorSurface(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !bool {
    _ = try locateQ4Tensor(allocator, io, path, "model.embed_tokens", vocab_size, hidden_size);
    _ = try locateQ4Tensor(allocator, io, path, "model.layers.0.self_attn.q_proj", hidden_size, hidden_size);
    _ = try locateQ4Tensor(allocator, io, path, "model.layers.27.mlp.down_proj", hidden_size, intermediate_size);
    const norm = try bolt.bonsai_model.locateTensor(io, allocator, path, "model.norm.weight");
    return norm.dtype == .bf16 and norm.dims[0] == hidden_size;
}

fn runMetalQ4Probe(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !MetalQ4Probe {
    if (!bolt.metal.context.Context.isAvailable()) {
        return .{ .available = false, .used = false, .kernel = "q4mv_f32", .tensor = "all_q4_projection_tensors_sampled", .tensor_count = 0, .rows = 0, .cols = 0, .group_size = 0, .dispatches = 0, .max_abs_error = std.math.inf(f32), .pass = false };
    }

    var context = try bolt.metal.context.Context.init();
    defer context.deinit();

    const sample_rows: usize = 8;
    const specs = [_]Q4ProjectionSpec{
        .{ .suffix = "self_attn.q_proj", .rows = hidden_size, .cols = hidden_size },
        .{ .suffix = "self_attn.k_proj", .rows = kv_dim, .cols = hidden_size },
        .{ .suffix = "self_attn.v_proj", .rows = kv_dim, .cols = hidden_size },
        .{ .suffix = "self_attn.o_proj", .rows = hidden_size, .cols = hidden_size },
        .{ .suffix = "mlp.gate_proj", .rows = intermediate_size, .cols = hidden_size },
        .{ .suffix = "mlp.up_proj", .rows = intermediate_size, .cols = hidden_size },
        .{ .suffix = "mlp.down_proj", .rows = hidden_size, .cols = intermediate_size },
    };

    var max_abs_error: f32 = 0.0;
    var dispatches: u32 = 0;
    var max_cols: usize = 0;
    for (0..layer_count) |layer_index| {
        for (specs) |spec| {
            var name_buf: [128]u8 = undefined;
            const tensor_name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.{s}", .{ layer_index, spec.suffix });
            const err = try runMetalQ4ProbeOne(&context, io, allocator, path, tensor_name, spec.rows, spec.cols, sample_rows);
            if (err > max_abs_error) max_abs_error = err;
            if (spec.cols > max_cols) max_cols = spec.cols;
            dispatches += 1;
        }
    }

    return .{
        .available = true,
        .used = true,
        .kernel = "q4mv_f32",
        .tensor = "all_q4_projection_tensors_sampled",
        .tensor_count = dispatches,
        .rows = @intCast(sample_rows),
        .cols = @intCast(max_cols),
        .group_size = 64,
        .dispatches = dispatches,
        .max_abs_error = max_abs_error,
        .pass = dispatches == layer_count * specs.len and max_abs_error <= 0.05,
    };
}

fn runMetalQ4ProbeOne(context: *bolt.metal.context.Context, io: std.Io, allocator: std.mem.Allocator, path: []const u8, tensor_name: []const u8, expected_rows: usize, expected_cols: usize, sample_rows: usize) !f32 {
    const tensor = try locateQ4Tensor(allocator, io, path, tensor_name, expected_rows, expected_cols);
    const rows = @min(sample_rows, tensor.rows);
    const packed_words = try allocator.alloc(u32, rows * tensor.words_per_row);
    defer allocator.free(packed_words);
    const scales_f32 = try allocator.alloc(f32, rows * tensor.groups);
    defer allocator.free(scales_f32);
    const biases_f32 = try allocator.alloc(f32, rows * tensor.groups);
    defer allocator.free(biases_f32);
    const scales_raw = try allocator.alloc(u16, rows * tensor.groups);
    defer allocator.free(scales_raw);
    const biases_raw = try allocator.alloc(u16, rows * tensor.groups);
    defer allocator.free(biases_raw);

    if (try bolt.bonsai_model.readU32LocationSlice(io, allocator, path, tensor.weight, 0, packed_words) != packed_words.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16LocationSlice(io, allocator, path, tensor.scales, 0, scales_f32) != scales_f32.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16LocationSlice(io, allocator, path, tensor.biases, 0, biases_f32) != biases_f32.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16RawLocationSlice(io, allocator, path, tensor.scales, 0, scales_raw) != scales_raw.len) return error.ShortTensorRead;
    if (try bolt.bonsai_model.readBf16RawLocationSlice(io, allocator, path, tensor.biases, 0, biases_raw) != biases_raw.len) return error.ShortTensorRead;

    const input_values = try allocator.alloc(f32, tensor.cols);
    defer allocator.free(input_values);
    for (input_values, 0..) |*value, index| {
        const centered: i32 = @as(i32, @intCast(index % 17)) - 8;
        value.* = @as(f32, @floatFromInt(centered)) / 16.0;
    }
    const expected = try allocator.alloc(f32, rows);
    defer allocator.free(expected);
    try bolt.q4.matVecRows(.{
        .packed_words = packed_words,
        .scales = scales_f32,
        .biases = biases_f32,
        .rows = rows,
        .cols = tensor.cols,
        .group_size = tensor.group_size,
    }, input_values, expected, false);

    var q4 = try context.createQ4Buffer(rows, tensor.cols, tensor.group_size);
    defer q4.deinit();
    try q4.packed_nibbles.write(std.mem.sliceAsBytes(packed_words));
    try q4.scales_bf16.writeRaw(scales_raw);
    try q4.biases_bf16.writeRaw(biases_raw);
    var input = try context.createSharedBufferF32(tensor.cols);
    defer input.deinit();
    try input.write(input_values);
    var output = try context.createSharedBufferF32(rows);
    defer output.deinit();
    try context.q4mvSharedBufferF32(q4, input, output);

    var max_abs_error: f32 = 0.0;
    for (output.asSlice(), expected) |actual, want| {
        const err = @abs(actual - want);
        if (err > max_abs_error) max_abs_error = err;
    }
    return max_abs_error;
}

fn findQ4Paths(allocator: std.mem.Allocator, repo_root: []const u8, root: std.json.Value) !Paths {
    if (root != .object) return error.InvalidAssetManifest;
    const assets = root.object.get("assets") orelse return error.MissingAssets;
    if (assets != .array) return error.InvalidAssets;
    for (assets.array.items) |asset| {
        if (asset != .object) continue;
        const idv = asset.object.get("id") orelse continue;
        if (idv != .string or !std.mem.eql(u8, idv.string, "qwen3-1.7b-q4-gs64")) continue;
        const files = asset.object.get("files") orelse return error.MissingAssetFiles;
        var c: ?[]const u8 = null;
        var t: ?[]const u8 = null;
        var s: ?[]const u8 = null;
        for (files.array.items) |file| {
            if (file != .object) continue;
            const pv = file.object.get("path") orelse continue;
            if (pv != .string) continue;
            const base = std.fs.path.basename(pv.string);
            if (std.mem.eql(u8, base, "config.json")) c = try resolveManifestPath(allocator, repo_root, pv.string);
            if (std.mem.eql(u8, base, "tokenizer.json")) t = try resolveManifestPath(allocator, repo_root, pv.string);
            if (std.mem.eql(u8, base, "model.safetensors")) s = try resolveManifestPath(allocator, repo_root, pv.string);
        }
        if (c == null) return missingManifestEntry("qwen3-1.7b-q4-gs64", "config.json");
        if (t == null) return missingManifestEntry("qwen3-1.7b-q4-gs64", "tokenizer.json");
        if (s == null) return missingManifestEntry("qwen3-1.7b-q4-gs64", "model.safetensors");
        return .{ .config_path = c.?, .tokenizer_path = t.?, .safetensors_path = s.? };
    }
    return missingAssetEntry("qwen3-1.7b-q4-gs64");
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

fn writeIds(writer: anytype, ids: []const u32) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, i| {
        if (i != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id});
    }
    try writer.writeByte(']');
}

fn writeArtifact(init: std.process.Init, manifest_path: []const u8, paths: Paths, prompt_ids: []const u32, generation: GenerationProbe, metal_probe: MetalQ4Probe, tokenizer_pass: bool, tensor_pass: bool, gate_pass: bool) !void {
    try ensureParentDir(init.io, artifact_path);
    var file = try std.Io.Dir.cwd().createFile(init.io, artifact_path, .{ .truncate = true });
    defer file.close(init.io);
    var buf: [32768]u8 = undefined;
    var fw = file.writer(init.io, &buf);
    const w = &fw.interface;
    try writeJson(w, manifest_path, paths, prompt_ids, generation, metal_probe, tokenizer_pass, tensor_pass, gate_pass);
    try fw.interface.flush();
    var sbuf: [32768]u8 = undefined;
    var sw = std.Io.File.stdout().writer(init.io, &sbuf);
    try writeJson(&sw.interface, manifest_path, paths, prompt_ids, generation, metal_probe, tokenizer_pass, tensor_pass, gate_pass);
    try sw.interface.flush();
}

fn writeJson(w: anytype, manifest_path: []const u8, paths: Paths, prompt_ids: []const u32, g: GenerationProbe, metal_probe: MetalQ4Probe, tokenizer_pass: bool, tensor_pass: bool, gate_pass: bool) !void {
    try w.print("{{\n  \"schema_version\":1,\n  \"gate\":\"bolt-bonsai-q4-golden\",\n  \"status\":\"{s}\",\n  \"acceptance\":\"real_q4_transformer_golden_{s}\",\n  \"manifest_path\":\"{s}\",\n  \"config_path\":\"{s}\",\n  \"tokenizer_path\":\"{s}\",\n  \"safetensors_path\":\"{s}\",\n  \"tokenizer_pass\":{s},\n  \"tensor_pass\":{s},\n  \"prompt\":\"{s}\",\n  \"prompt_token_ids\":", .{ if (gate_pass) "pass" else "blocked", if (gate_pass) "passed" else "blocked", manifest_path, paths.config_path, paths.tokenizer_path, paths.safetensors_path, if (tokenizer_pass) "true" else "false", if (tensor_pass) "true" else "false", golden_prompt });
    try writeIds(w, prompt_ids);
    try w.writeAll(",\n  \"generated_tokens\":");
    try writeIds(w, g.generated_tokens[0..g.generated_count]);
    try w.print(",\n  \"matched_tokens\":{d},\n  \"first_mismatch_index\":{d},\n  \"matches_reference\":{s},\n  \"integrated_metal_decode\":{{\"used\":{s},\"q4_projection_dispatches\":{d},\"q4_logits_dispatches\":{d}}},\n  \"reference_golden_tokens\":", .{
        g.matched_tokens,
        g.first_mismatch_index,
        if (g.matches_reference) "true" else "false",
        if (g.metal_decode_used) "true" else "false",
        g.metal_projection_dispatches,
        g.metal_logits_dispatches,
    });
    try writeIds(w, &golden_tokens);
    try w.writeAll(",\n  \"top_logits\":[");
    for (0..g.generated_count) |i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{d}", .{g.top_logits[i]});
    }
    try w.print("],\n  \"metal_q4mv_probe\":{{\"available\":{s},\"used\":{s},\"kernel\":\"{s}\",\"tensor\":\"{s}\",\"tensor_count\":{d},\"sampled_rows_per_tensor\":{d},\"max_cols\":{d},\"group_size\":{d},\"dispatches\":{d},\"max_abs_error\":{d},\"pass\":{s}}},\n  \"quantized_tolerances\":{{\"selected_token_parity\":\"exact\",\"logit_mode\":\"Q4 affine BF16 diagnostic, no F16 equality claim\",\"metal_q4mv_max_abs_error\":0.05}}\n}}\n", .{ if (metal_probe.available) "true" else "false", if (metal_probe.used) "true" else "false", metal_probe.kernel, metal_probe.tensor, metal_probe.tensor_count, metal_probe.rows, metal_probe.cols, metal_probe.group_size, metal_probe.dispatches, metal_probe.max_abs_error, if (metal_probe.pass) "true" else "false" });
}
