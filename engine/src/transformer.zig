const std = @import("std");

extern fn cblas_sgemv(order: c_int, trans: c_int, m: c_int, n: c_int, alpha: f32, a: [*]const f32, lda: c_int, x: [*]const f32, inc_x: c_int, beta: f32, y: [*]f32, inc_y: c_int) void;
const cblas_row_major: c_int = 101;
const cblas_no_trans: c_int = 111;

pub const QuantFormat = enum { q1_0, q4_mlx, fp16 };

pub const TransformerDesc = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_layers: u32,
    num_query_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_context_length: u32,
    max_prefill_length: u32,
    rope_theta: f32,
    tie_word_embeddings: bool,
    quant_format: QuantFormat = .q1_0,
    group_size: u32 = 128,
};

fn divCeil(comptime T: type, a: T, b: T) T {
    return (a + b - 1) / b;
}

pub fn TransformerConfig(comptime cfg: TransformerDesc) type {
    return struct {
        pub const vocab_size = cfg.vocab_size;
        pub const hidden_size = cfg.hidden_size;
        pub const intermediate_size = cfg.intermediate_size;
        pub const num_layers = cfg.num_layers;
        pub const num_query_heads = cfg.num_query_heads;
        pub const num_kv_heads = cfg.num_kv_heads;
        pub const head_dim = cfg.head_dim;
        pub const max_context_length = cfg.max_context_length;
        pub const max_prefill_length = cfg.max_prefill_length;
        pub const rope_theta = cfg.rope_theta;
        pub const tie_word_embeddings = cfg.tie_word_embeddings;
        pub const quant_format = cfg.quant_format;
        pub const group_size = cfg.group_size;

        pub const query_dim = num_query_heads * head_dim;
        pub const kv_dim = num_kv_heads * head_dim;
        pub const heads_per_kv_group = num_query_heads / num_kv_heads;

        comptime {
            std.debug.assert(num_query_heads % num_kv_heads == 0);
            std.debug.assert(head_dim % 2 == 0);
            std.debug.assert(vocab_size > 0 and hidden_size > 0 and intermediate_size > 0);
            std.debug.assert(num_layers > 0 and max_context_length > 0);
            std.debug.assert(max_prefill_length <= max_context_length);
        }

        fn packedBytes(comptime rows: u32, comptime cols: u32) u32 {
            const total = rows * cols;
            const groups = divCeil(u32, total, group_size);
            return switch (quant_format) {
                .q1_0 => divCeil(u32, total, 8) + groups * 2,
                .q4_mlx => divCeil(u32, total, 2) + groups * 4,
                .fp16 => total * 2,
            };
        }

        pub const q_proj_bytes = packedBytes(hidden_size, query_dim);
        pub const k_proj_bytes = packedBytes(hidden_size, kv_dim);
        pub const v_proj_bytes = packedBytes(hidden_size, kv_dim);
        pub const o_proj_bytes = packedBytes(query_dim, hidden_size);
        pub const gate_proj_bytes = packedBytes(hidden_size, intermediate_size);
        pub const up_proj_bytes = packedBytes(hidden_size, intermediate_size);
        pub const down_proj_bytes = packedBytes(intermediate_size, hidden_size);
        pub const attention_weight_bytes = q_proj_bytes + k_proj_bytes + v_proj_bytes + o_proj_bytes;
        pub const mlp_weight_bytes = gate_proj_bytes + up_proj_bytes + down_proj_bytes;
        pub const layer_weight_bytes = attention_weight_bytes + mlp_weight_bytes;
        pub const total_layer_weight_bytes = layer_weight_bytes * num_layers;
        pub const embedding_bytes = packedBytes(vocab_size, hidden_size);
        pub const lm_head_bytes = if (tie_word_embeddings) 0 else packedBytes(hidden_size, vocab_size);
        pub const norm_scales_per_layer = hidden_size * 2 + head_dim * 2;
        pub const total_norm_scale_count = norm_scales_per_layer * num_layers + hidden_size;
        pub const total_weight_bytes: u64 = @as(u64, embedding_bytes) + total_layer_weight_bytes + lm_head_bytes + total_norm_scale_count * 4;
        pub const kv_cache_elements_per_layer: u64 = @as(u64, max_context_length) * kv_dim * 2;
        pub const total_kv_cache_elements: u64 = kv_cache_elements_per_layer * num_layers;
        pub const total_kv_cache_bytes: u64 = total_kv_cache_elements * 2;
        pub const decode_activation_elements = @max(hidden_size, @max(query_dim, intermediate_size));
    };
}

pub const Bonsai1_7B = TransformerConfig(.{ .vocab_size = 151669, .hidden_size = 2048, .intermediate_size = 6144, .num_layers = 28, .num_query_heads = 16, .num_kv_heads = 8, .head_dim = 128, .max_context_length = 32768, .max_prefill_length = 512, .rope_theta = 1000000.0, .tie_word_embeddings = true });
pub const Bonsai1_7B_Q4 = TransformerConfig(.{ .vocab_size = 151936, .hidden_size = 2048, .intermediate_size = 6144, .num_layers = 28, .num_query_heads = 16, .num_kv_heads = 8, .head_dim = 128, .max_context_length = 32768, .max_prefill_length = 512, .rope_theta = 1000000.0, .tie_word_embeddings = true, .quant_format = .q4_mlx, .group_size = 128 });

pub fn bf16Round(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    return @bitCast(bits & 0xFFFF0000);
}

pub fn f16Round(value: f32) f32 {
    const half: f16 = @floatCast(value);
    return @floatCast(half);
}

pub fn bf16ThenF16Round(value: f32) f32 {
    return f16Round(bf16Round(value));
}

pub fn roundSliceBf16(values: []f32) void {
    for (values) |*value| value.* = bf16Round(value.*);
}

pub fn roundSliceF16(values: []f32) void {
    for (values) |*value| value.* = f16Round(value.*);
}

pub fn roundSliceBf16ThenF16(values: []f32) void {
    for (values) |*value| value.* = bf16ThenF16Round(value.*);
}

pub fn rmsNorm(input: []const f32, scale: []const f32, output: []f32, eps: f32) !void {
    if (input.len != scale.len or input.len != output.len or input.len == 0) return error.InvalidRmsNormShape;
    var sum_sq: f32 = 0;
    for (input) |v| sum_sq += v * v;
    const inv = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(input.len)) + eps);
    for (input, scale, output) |v, s, *out| out.* = v * inv * s;
}

pub fn silu(x: f32) f32 {
    return x / (1.0 + @exp(-x));
}

pub fn applyRoPE(q: []f32, k: []f32, position: u32, theta: f32) !void {
    if (q.len != k.len or q.len % 2 != 0) return error.InvalidRoPEShape;
    const half = q.len / 2;
    for (0..half) |i| {
        const freq = @as(f32, @floatFromInt(position)) / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(q.len)));
        const c = @cos(freq);
        const s = @sin(freq);
        rotatePair(q, i * 2, c, s);
        rotatePair(k, i * 2, c, s);
    }
}

fn rotatePair(values: []f32, index: usize, c: f32, s: f32) void {
    const a = values[index];
    const b = values[index + 1];
    values[index] = a * c - b * s;
    values[index + 1] = a * s + b * c;
}

test "Bonsai config constants match reference" {
    try std.testing.expectEqual(@as(u32, 2048), Bonsai1_7B.hidden_size);
    try std.testing.expectEqual(@as(u32, 28), Bonsai1_7B.num_layers);
    try std.testing.expectEqual(@as(u32, 2048), Bonsai1_7B.query_dim);
    try std.testing.expectEqual(@as(u32, 1024), Bonsai1_7B.kv_dim);
    try std.testing.expectEqual(@as(u32, 151936), Bonsai1_7B_Q4.vocab_size);
    try std.testing.expectEqual(@as(u32, 128), Bonsai1_7B_Q4.group_size);
}

test "CPU transformer primitives are deterministic" {
    const input = [_]f32{ 1, 2, 3, 4 };
    const scale = [_]f32{ 1, 1, 1, 1 };
    var out: [4]f32 = undefined;
    try rmsNorm(&input, &scale, &out, 1e-5);
    try std.testing.expect(out[0] > 0.36 and out[0] < 0.37);
    try std.testing.expect(silu(1.0) > 0.73 and silu(1.0) < 0.74);
    var q = [_]f32{ 1, 0, 0, 1 };
    var k = [_]f32{ 1, 0, 0, 1 };
    try applyRoPE(&q, &k, 1, 10000.0);
    try std.testing.expect(q[0] < 1.0 and q[1] > 0.0);
}

pub const CpuDecodeDims = struct {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f32 = 1e-5,
    rope_theta: f32 = 10000.0,

    pub fn queryDim(self: CpuDecodeDims) usize {
        return self.num_query_heads * self.head_dim;
    }

    pub fn kvDim(self: CpuDecodeDims) usize {
        return self.num_kv_heads * self.head_dim;
    }

    pub fn headsPerKvGroup(self: CpuDecodeDims) usize {
        return self.num_query_heads / self.num_kv_heads;
    }

    pub fn validate(self: CpuDecodeDims) !void {
        if (self.vocab_size == 0 or self.hidden_size == 0 or self.intermediate_size == 0) return error.InvalidCpuDecodeDims;
        if (self.num_query_heads == 0 or self.num_kv_heads == 0 or self.head_dim == 0) return error.InvalidCpuDecodeDims;
        if (self.num_query_heads % self.num_kv_heads != 0) return error.InvalidCpuDecodeDims;
        if (self.head_dim % 2 != 0) return error.InvalidCpuDecodeDims;
        if (self.queryDim() != self.hidden_size) return error.UnsupportedCpuDecodeShape;
    }
};

pub const CpuLayerWeights = struct {
    input_norm: []const f32,
    post_norm: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    q_proj: []const f32,
    k_proj: []const f32,
    v_proj: []const f32,
    o_proj: []const f32,
    gate_proj: []const f32,
    up_proj: []const f32,
    down_proj: []const f32,
};

pub const CpuTransformerWeights = struct {
    dims: CpuDecodeDims,
    embedding: []const f32,
    final_norm: []const f32,
    layers: []const CpuLayerWeights,
};

pub const CpuDecodeTrace = struct {
    layer_count: usize,
    prompt_tokens: usize,
    last_token: u32,
    next_token: u32,
    top_logit: f32,
};

pub fn cpuDecodeGreedy(allocator: std.mem.Allocator, weights: CpuTransformerWeights, prompt_tokens: []const u32) !CpuDecodeTrace {
    try weights.dims.validate();
    if (weights.layers.len == 0 or prompt_tokens.len == 0) return error.InvalidCpuDecodeInput;
    const d = weights.dims;
    if (weights.embedding.len != d.vocab_size * d.hidden_size) return error.InvalidCpuDecodeWeights;
    if (weights.final_norm.len != d.hidden_size) return error.InvalidCpuDecodeWeights;

    const max_seq = prompt_tokens.len;
    const kvd = d.kvDim();
    const qd = d.queryDim();
    const k_cache = try allocator.alloc(f32, weights.layers.len * max_seq * kvd);
    defer allocator.free(k_cache);
    const v_cache = try allocator.alloc(f32, weights.layers.len * max_seq * kvd);
    defer allocator.free(v_cache);
    @memset(k_cache, 0);
    @memset(v_cache, 0);

    const state = try allocator.alloc(f32, d.hidden_size);
    defer allocator.free(state);
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
    const logits = try allocator.alloc(f32, d.vocab_size);
    defer allocator.free(logits);

    for (prompt_tokens, 0..) |token, position| {
        if (token >= d.vocab_size) return error.TokenOutOfRange;
        @memcpy(state, weights.embedding[@as(usize, token) * d.hidden_size ..][0..d.hidden_size]);
        for (weights.layers, 0..) |layer, layer_index| {
            try validateCpuLayer(d, layer);
            try rmsNorm(state, layer.input_norm, norm, d.eps);
            matVec(layer.q_proj, qd, d.hidden_size, norm, q);
            matVec(layer.k_proj, kvd, d.hidden_size, norm, k);
            matVec(layer.v_proj, kvd, d.hidden_size, norm, v);
            try normalizeHeads(q, d.num_query_heads, d.head_dim, layer.q_norm, d.eps);
            try normalizeHeads(k, d.num_kv_heads, d.head_dim, layer.k_norm, d.eps);
            try applyRoPEGrouped(q, d.num_query_heads, d.head_dim, @intCast(position), d.rope_theta);
            try applyRoPEGrouped(k, d.num_kv_heads, d.head_dim, @intCast(position), d.rope_theta);
            const cache_base = layer_index * max_seq * kvd + position * kvd;
            @memcpy(k_cache[cache_base..][0..kvd], k);
            @memcpy(v_cache[cache_base..][0..kvd], v);
            try cpuGqaAttention(d, q, k_cache[layer_index * max_seq * kvd ..][0 .. max_seq * kvd], v_cache[layer_index * max_seq * kvd ..][0 .. max_seq * kvd], position + 1, attn, allocator);
            matVec(layer.o_proj, d.hidden_size, qd, attn, proj);
            for (state, proj) |*s, p| s.* += p;
            try rmsNorm(state, layer.post_norm, norm, d.eps);
            matVec(layer.gate_proj, d.intermediate_size, d.hidden_size, norm, gate);
            matVec(layer.up_proj, d.intermediate_size, d.hidden_size, norm, up);
            for (gate, up) |*g, u| g.* = silu(g.*) * u;
            matVec(layer.down_proj, d.hidden_size, d.intermediate_size, gate, mlp);
            for (state, mlp) |*s, m| s.* += m;
        }
    }

    try rmsNorm(state, weights.final_norm, norm, d.eps);
    matVec(weights.embedding, d.vocab_size, d.hidden_size, norm, logits);
    var best_index: usize = 0;
    var best_value = logits[0];
    for (logits[1..], 1..) |value, index| {
        if (value > best_value) {
            best_value = value;
            best_index = index;
        }
    }
    return .{
        .layer_count = weights.layers.len,
        .prompt_tokens = prompt_tokens.len,
        .last_token = prompt_tokens[prompt_tokens.len - 1],
        .next_token = @intCast(best_index),
        .top_logit = best_value,
    };
}

pub fn cpuForwardOneLayer(allocator: std.mem.Allocator, d: CpuDecodeDims, layer: CpuLayerWeights, state: []f32, position: usize, k_cache: []f32, v_cache: []f32) !void {
    try d.validate();
    try validateCpuLayer(d, layer);
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

    try rmsNorm(state, layer.input_norm, norm, d.eps);
    matVec(layer.q_proj, qd, d.hidden_size, norm, q);
    matVec(layer.k_proj, kvd, d.hidden_size, norm, k);
    matVec(layer.v_proj, kvd, d.hidden_size, norm, v);
    try normalizeHeads(q, d.num_query_heads, d.head_dim, layer.q_norm, d.eps);
    try normalizeHeads(k, d.num_kv_heads, d.head_dim, layer.k_norm, d.eps);
    try applyRoPEGrouped(q, d.num_query_heads, d.head_dim, @intCast(position), d.rope_theta);
    try applyRoPEGrouped(k, d.num_kv_heads, d.head_dim, @intCast(position), d.rope_theta);
    @memcpy(k_cache[position * kvd ..][0..kvd], k);
    @memcpy(v_cache[position * kvd ..][0..kvd], v);
    try cpuGqaAttention(d, q, k_cache, v_cache, position + 1, attn, allocator);
    matVec(layer.o_proj, d.hidden_size, qd, attn, proj);
    for (state, proj) |*s, value| s.* += value;
    try rmsNorm(state, layer.post_norm, norm, d.eps);
    matVec(layer.gate_proj, d.intermediate_size, d.hidden_size, norm, gate);
    matVec(layer.up_proj, d.intermediate_size, d.hidden_size, norm, up);
    for (gate, up) |*g, u| g.* = silu(g.*) * u;
    matVec(layer.down_proj, d.hidden_size, d.intermediate_size, gate, mlp);
    for (state, mlp) |*s, value| s.* += value;
}

pub fn cpuForwardOneLayerQ4Approx(allocator: std.mem.Allocator, d: CpuDecodeDims, layer: CpuLayerWeights, state: []f32, position: usize, k_cache: []f32, v_cache: []f32) !void {
    try d.validate();
    try validateCpuLayer(d, layer);
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

    try rmsNorm(state, layer.input_norm, norm, d.eps);
    roundSliceBf16ThenF16(norm);
    matVec(layer.q_proj, qd, d.hidden_size, norm, q);
    matVec(layer.k_proj, kvd, d.hidden_size, norm, k);
    matVec(layer.v_proj, kvd, d.hidden_size, norm, v);
    roundSliceBf16ThenF16(q);
    roundSliceBf16ThenF16(k);
    roundSliceBf16ThenF16(v);
    try normalizeHeads(q, d.num_query_heads, d.head_dim, layer.q_norm, d.eps);
    try normalizeHeads(k, d.num_kv_heads, d.head_dim, layer.k_norm, d.eps);
    try applyRoPEGrouped(q, d.num_query_heads, d.head_dim, @intCast(position), d.rope_theta);
    try applyRoPEGrouped(k, d.num_kv_heads, d.head_dim, @intCast(position), d.rope_theta);
    roundSliceF16(k);
    @memcpy(k_cache[position * kvd ..][0..kvd], k);
    @memcpy(v_cache[position * kvd ..][0..kvd], v);
    try cpuGqaAttention(d, q, k_cache, v_cache, position + 1, attn, allocator);
    roundSliceF16(attn);
    matVec(layer.o_proj, d.hidden_size, qd, attn, proj);
    roundSliceBf16(proj);
    for (state, proj) |*s, value| s.* += value;
    try rmsNorm(state, layer.post_norm, norm, d.eps);
    roundSliceBf16ThenF16(norm);
    matVec(layer.gate_proj, d.intermediate_size, d.hidden_size, norm, gate);
    matVec(layer.up_proj, d.intermediate_size, d.hidden_size, norm, up);
    for (gate, up) |*g, u| g.* = bf16ThenF16Round(silu(g.*) * u);
    matVec(layer.down_proj, d.hidden_size, d.intermediate_size, gate, mlp);
    roundSliceBf16(mlp);
    for (state, mlp) |*s, value| s.* += value;
}

fn validateCpuLayer(d: CpuDecodeDims, layer: CpuLayerWeights) !void {
    const qd = d.queryDim();
    const kvd = d.kvDim();
    if (layer.input_norm.len != d.hidden_size or layer.post_norm.len != d.hidden_size) return error.InvalidCpuDecodeWeights;
    if (layer.q_norm.len != d.head_dim or layer.k_norm.len != d.head_dim) return error.InvalidCpuDecodeWeights;
    if (layer.q_proj.len != qd * d.hidden_size) return error.InvalidCpuDecodeWeights;
    if (layer.k_proj.len != kvd * d.hidden_size or layer.v_proj.len != kvd * d.hidden_size) return error.InvalidCpuDecodeWeights;
    if (layer.o_proj.len != d.hidden_size * qd) return error.InvalidCpuDecodeWeights;
    if (layer.gate_proj.len != d.intermediate_size * d.hidden_size or layer.up_proj.len != d.intermediate_size * d.hidden_size) return error.InvalidCpuDecodeWeights;
    if (layer.down_proj.len != d.hidden_size * d.intermediate_size) return error.InvalidCpuDecodeWeights;
}

pub fn matVecRows(matrix: []const f32, rows: usize, cols: usize, vector: []const f32, out: []f32) void {
    matVec(matrix, rows, cols, vector, out);
}

fn matVec(matrix: []const f32, rows: usize, cols: usize, vector: []const f32, out: []f32) void {
    std.debug.assert(matrix.len == rows * cols);
    std.debug.assert(vector.len == cols);
    std.debug.assert(out.len == rows);
    if (rows == 0 or cols == 0) return;
    if (rows <= std.math.maxInt(c_int) and cols <= std.math.maxInt(c_int)) {
        cblas_sgemv(
            cblas_row_major,
            cblas_no_trans,
            @intCast(rows),
            @intCast(cols),
            1.0,
            matrix.ptr,
            @intCast(cols),
            vector.ptr,
            1,
            0.0,
            out.ptr,
            1,
        );
        return;
    }
    for (0..rows) |r| {
        var sum: f32 = 0;
        const row = matrix[r * cols ..][0..cols];
        for (row, vector) |w, x| sum += w * x;
        out[r] = sum;
    }
}

pub fn normalizeHeads(values: []f32, heads: usize, head_dim: usize, scale: []const f32, eps: f32) !void {
    if (values.len != heads * head_dim or scale.len != head_dim) return error.InvalidCpuDecodeWeights;
    for (0..heads) |head| {
        const slice = values[head * head_dim ..][0..head_dim];
        try rmsNorm(slice, scale, slice, eps);
    }
}

pub fn applyRoPEGrouped(values: []f32, heads: usize, head_dim: usize, position: u32, theta: f32) !void {
    if (values.len != heads * head_dim or head_dim % 2 != 0) return error.InvalidRoPEShape;
    for (0..heads) |head| {
        const head_values = values[head * head_dim ..][0..head_dim];
        const half = head_dim / 2;
        for (0..half) |i| {
            const freq = @as(f32, @floatFromInt(position)) / std.math.pow(f32, theta, @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(head_dim)));
            rotatePair(head_values, i * 2, @cos(freq), @sin(freq));
        }
    }
}

pub fn cpuGqaAttention(d: CpuDecodeDims, q: []const f32, k_cache: []const f32, v_cache: []const f32, seq_len: usize, out: []f32, allocator: std.mem.Allocator) !void {
    const qd = d.queryDim();
    const kvd = d.kvDim();
    if (q.len != qd or out.len != qd or k_cache.len < seq_len * kvd or v_cache.len < seq_len * kvd) return error.InvalidCpuDecodeShape;
    const scores = try allocator.alloc(f32, seq_len);
    defer allocator.free(scores);
    const inv_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d.head_dim)));
    for (0..d.num_query_heads) |qh| {
        const kvh = qh / d.headsPerKvGroup();
        const q_head = q[qh * d.head_dim ..][0..d.head_dim];
        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |pos| {
            const k_head = k_cache[pos * kvd + kvh * d.head_dim ..][0..d.head_dim];
            var score: f32 = 0;
            for (q_head, k_head) |a, b| score += a * b;
            score *= inv_scale;
            scores[pos] = score;
            if (score > max_score) max_score = score;
        }
        var denom: f32 = 0;
        for (scores) |*score| {
            score.* = @exp(score.* - max_score);
            denom += score.*;
        }
        const out_head = out[qh * d.head_dim ..][0..d.head_dim];
        @memset(out_head, 0);
        for (0..seq_len) |pos| {
            const weight = scores[pos] / denom;
            const v_head = v_cache[pos * kvd + kvh * d.head_dim ..][0..d.head_dim];
            for (out_head, v_head) |*dst, src| dst.* += weight * src;
        }
    }
}

test "CPU tiny transformer decode emits deterministic trace" {
    const allocator = std.testing.allocator;
    const dims = CpuDecodeDims{ .vocab_size = 4, .hidden_size = 2, .intermediate_size = 2, .num_query_heads = 1, .num_kv_heads = 1, .head_dim = 2, .eps = 1e-5, .rope_theta = 10000.0 };
    const embedding = [_]f32{ 1, 0, 0, 1, 0.5, 0.5, -1, 0 };
    const norm2 = [_]f32{ 1, 1 };
    const id2 = [_]f32{ 1, 0, 0, 1 };
    const layer = CpuLayerWeights{
        .input_norm = &norm2,
        .post_norm = &norm2,
        .q_norm = &norm2,
        .k_norm = &norm2,
        .q_proj = &id2,
        .k_proj = &id2,
        .v_proj = &id2,
        .o_proj = &id2,
        .gate_proj = &id2,
        .up_proj = &id2,
        .down_proj = &id2,
    };
    const layers = [_]CpuLayerWeights{layer};
    const weights = CpuTransformerWeights{ .dims = dims, .embedding = &embedding, .final_norm = &norm2, .layers = &layers };
    const prompt = [_]u32{ 0, 1 };
    const trace = try cpuDecodeGreedy(allocator, weights, &prompt);
    try std.testing.expectEqual(@as(usize, 1), trace.layer_count);
    try std.testing.expectEqual(@as(usize, 2), trace.prompt_tokens);
    try std.testing.expect(trace.next_token < dims.vocab_size);
    try std.testing.expect(trace.top_logit > 0.0);
}

test "CPU one-layer forward mutates state deterministically" {
    const allocator = std.testing.allocator;
    const dims = CpuDecodeDims{ .vocab_size = 4, .hidden_size = 2, .intermediate_size = 2, .num_query_heads = 1, .num_kv_heads = 1, .head_dim = 2, .eps = 1e-5, .rope_theta = 10000.0 };
    const norm2 = [_]f32{ 1, 1 };
    const id2 = [_]f32{ 1, 0, 0, 1 };
    const layer = CpuLayerWeights{ .input_norm = &norm2, .post_norm = &norm2, .q_norm = &norm2, .k_norm = &norm2, .q_proj = &id2, .k_proj = &id2, .v_proj = &id2, .o_proj = &id2, .gate_proj = &id2, .up_proj = &id2, .down_proj = &id2 };
    var state = [_]f32{ 1, 0 };
    var k_cache = [_]f32{0} ** 2;
    var v_cache = [_]f32{0} ** 2;
    try cpuForwardOneLayer(allocator, dims, layer, &state, 0, &k_cache, &v_cache);
    try std.testing.expect(state[0] > 1.0);
    try std.testing.expect(state[1] >= 0.0);
}

test "CPU grouped RoPE rotates once" {
    var values = [_]f32{ 1, 0, 0, 1 };
    try applyRoPEGrouped(&values, 1, 4, 1, 10000.0);
    try std.testing.expect(values[0] > 0.53 and values[0] < 0.55);
    try std.testing.expect(values[1] > 0.84 and values[1] < 0.85);
}
