const std = @import("std");

pub const default_output_projection = [_]f32{ 1.0, 1.5, 2.0, 0.5 };
pub const default_transition_bias = [_]f32{
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.1, 0.2, 0.0,
    0.0, 0.0, 0.1, 0.2,
    0.0, 0.8, 0.0, 0.1,
};
pub const tiny_decoder_vocab_size: usize = default_output_projection.len;

pub const WeightsConfig = struct {
    output_projection: []const f32,
    transition_bias: ?[]const f32 = null,
};

pub const sample_config_input =
    \\{
    \\  "output_projection": [1.0, 1.5, 2.0, 0.5],
    \\  "transition_bias": [
    \\    0.0, 0.0, 0.0, 0.0,
    \\    0.0, 0.1, 0.2, 0.0,
    \\    0.0, 0.0, 0.1, 0.2,
    \\    0.0, 0.8, 0.0, 0.1
    \\  ]
    \\}
;

pub const TokenScore = struct {
    token_id: usize,
    score_milli: usize,
};

pub const FixtureDecoderWeights = struct {
    output_projection: []const f32,
    transition_bias: []const f32,

    pub fn vocabSize(self: FixtureDecoderWeights) usize {
        return self.output_projection.len;
    }

    pub fn hasValidTransitionBias(self: FixtureDecoderWeights) bool {
        const vocab_size = self.vocabSize();
        return self.transition_bias.len == vocab_size * vocab_size;
    }

    pub fn weightedLogitSumMilli(self: FixtureDecoderWeights, logits: []const f32) !usize {
        if (logits.len != self.output_projection.len) {
            return error.LogitWeightLengthMismatch;
        }

        var total: f32 = 0;
        for (logits, self.output_projection) |logit, weight| {
            total += logit * weight;
        }

        return @as(usize, @intFromFloat((total * 1000.0) + 0.5));
    }

    pub fn topTokenWeightMilli(self: FixtureDecoderWeights, token_id: usize) !usize {
        if (token_id >= self.output_projection.len) {
            return error.InvalidTokenId;
        }
        return @as(usize, @intFromFloat((self.output_projection[token_id] * 1000.0) + 0.5));
    }

    pub fn transitionBiasMilli(
        self: FixtureDecoderWeights,
        source_token_id: usize,
        target_token_id: usize,
    ) !usize {
        if (!self.hasValidTransitionBias()) {
            return error.InvalidTransitionBiasShape;
        }
        const vocab_size = self.vocabSize();
        if (source_token_id >= vocab_size or target_token_id >= vocab_size) {
            return error.InvalidTokenId;
        }
        const index = (source_token_id * vocab_size) + target_token_id;
        return @as(usize, @intFromFloat((self.transition_bias[index] * 1000.0) + 0.5));
    }

    pub fn conditionedTopToken(
        self: FixtureDecoderWeights,
        logits: []const f32,
        source_token_id: usize,
    ) !TokenScore {
        if (logits.len != self.output_projection.len) {
            return error.LogitWeightLengthMismatch;
        }
        if (!self.hasValidTransitionBias()) {
            return error.InvalidTransitionBiasShape;
        }
        if (source_token_id >= self.vocabSize()) {
            return error.InvalidTokenId;
        }

        var token_id: usize = 0;
        var best_score = logits[0] + self.transition_bias[source_token_id * self.vocabSize()];
        for (logits[1..], 1..) |logit, index| {
            const score = logit + self.transition_bias[(source_token_id * self.vocabSize()) + index];
            if (score > best_score) {
                best_score = score;
                token_id = index;
            }
        }

        return .{
            .token_id = token_id,
            .score_milli = @as(usize, @intFromFloat((best_score * 1000.0) + 0.5)),
        };
    }

    pub fn modelScoreMilli(
        self: FixtureDecoderWeights,
        source_token_id: usize,
        target_token_id: usize,
    ) !usize {
        const projection_milli = try self.topTokenWeightMilli(target_token_id);
        const transition_bias_milli = try self.transitionBiasMilli(
            source_token_id,
            target_token_id,
        );
        return projection_milli + transition_bias_milli;
    }

    pub fn modelTopToken(
        self: FixtureDecoderWeights,
        source_token_id: usize,
    ) !TokenScore {
        if (!self.hasValidTransitionBias()) {
            return error.InvalidTransitionBiasShape;
        }
        if (source_token_id >= self.vocabSize()) {
            return error.InvalidTokenId;
        }

        var token_id: usize = 0;
        var best_score = try self.modelScoreMilli(source_token_id, 0);
        for (1..self.vocabSize()) |index| {
            const score = try self.modelScoreMilli(source_token_id, index);
            if (score > best_score) {
                best_score = score;
                token_id = index;
            }
        }

        return .{
            .token_id = token_id,
            .score_milli = best_score,
        };
    }

    pub fn promptContextScoreMilli(
        self: FixtureDecoderWeights,
        prompt_token_ids: []const usize,
        target_token_id: usize,
    ) !usize {
        if (prompt_token_ids.len == 0) {
            return error.EmptyPrompt;
        }

        var total = try self.topTokenWeightMilli(target_token_id);
        total += try self.promptContextBiasMilli(prompt_token_ids, target_token_id);
        return total;
    }

    pub fn promptContextBiasMilli(
        self: FixtureDecoderWeights,
        prompt_token_ids: []const usize,
        target_token_id: usize,
    ) !usize {
        if (prompt_token_ids.len == 0) {
            return error.EmptyPrompt;
        }

        var total: usize = 0;
        for (prompt_token_ids) |source_token_id| {
            total += try self.transitionBiasMilli(source_token_id, target_token_id);
        }
        return total;
    }

    pub fn promptContextTopToken(
        self: FixtureDecoderWeights,
        prompt_token_ids: []const usize,
    ) !TokenScore {
        if (prompt_token_ids.len == 0) {
            return error.EmptyPrompt;
        }
        if (!self.hasValidTransitionBias()) {
            return error.InvalidTransitionBiasShape;
        }

        var token_id: usize = 0;
        var best_score = try self.promptContextScoreMilli(prompt_token_ids, 0);
        for (1..self.vocabSize()) |index| {
            const score = try self.promptContextScoreMilli(prompt_token_ids, index);
            if (score > best_score) {
                best_score = score;
                token_id = index;
            }
        }

        return .{
            .token_id = token_id,
            .score_milli = best_score,
        };
    }
};

pub fn defaultWeights() FixtureDecoderWeights {
    return .{
        .output_projection = default_output_projection[0..],
        .transition_bias = default_transition_bias[0..],
    };
}

pub fn weightsFromConfig(config: WeightsConfig) FixtureDecoderWeights {
    return .{
        .output_projection = config.output_projection,
        .transition_bias = config.transition_bias orelse &.{},
    };
}

pub fn weightsFromSlices(
    projection: []const f32,
    transition_bias: []const f32,
) FixtureDecoderWeights {
    return .{
        .output_projection = projection,
        .transition_bias = transition_bias,
    };
}

pub fn parseConfigFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(WeightsConfig) {
    return std.json.parseFromSlice(
        WeightsConfig,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadConfigFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(WeightsConfig) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(input);

    return parseConfigFromSlice(allocator, input);
}

pub fn decodeValuesFromBytes(
    allocator: std.mem.Allocator,
    bytes: []const u8,
) ![]f32 {
    if (bytes.len == 0 or bytes.len % @sizeOf(f32) != 0) {
        return error.InvalidBinaryWeightsSize;
    }

    const count = bytes.len / @sizeOf(f32);
    const values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);

    for (0..count) |index| {
        const start = index * @sizeOf(f32);
        const word: *const [4]u8 = @ptrCast(bytes[start .. start + @sizeOf(f32)].ptr);
        const bits = std.mem.readInt(u32, word, .little);
        values[index] = @bitCast(bits);
    }

    return values;
}

pub fn loadValuesFromBinaryFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(bytes);

    return decodeValuesFromBytes(allocator, bytes);
}

test "weighted logit sum is deterministic" {
    const weight_set = defaultWeights();
    try std.testing.expectEqual(@as(usize, 2400), try weight_set.weightedLogitSumMilli(&.{ 0.1, 0.2, 0.9, 0.4 }));
}

test "top token weight is deterministic" {
    const weight_set = defaultWeights();
    try std.testing.expectEqual(@as(usize, 2000), try weight_set.topTokenWeightMilli(2));
}

test "conditioned top token is deterministic" {
    const weight_set = defaultWeights();
    const conditioned = try weight_set.conditionedTopToken(&.{ 0.1, 0.2, 0.9, 0.4 }, 3);
    try std.testing.expectEqual(@as(usize, 1), conditioned.token_id);
    try std.testing.expectEqual(@as(usize, 1000), conditioned.score_milli);
}

test "model top token is deterministic" {
    const weight_set = defaultWeights();
    const model_top = try weight_set.modelTopToken(3);
    try std.testing.expectEqual(@as(usize, 1), model_top.token_id);
    try std.testing.expectEqual(@as(usize, 2300), model_top.score_milli);
}

test "prompt context top token is deterministic" {
    const weight_set = defaultWeights();
    const context_top = try weight_set.promptContextTopToken(&.{ 1, 2, 3 });
    try std.testing.expectEqual(@as(usize, 1), context_top.token_id);
    try std.testing.expectEqual(@as(usize, 2400), context_top.score_milli);
}

test "parse weights config" {
    const allocator = std.testing.allocator;

    var parsed = try parseConfigFromSlice(allocator, sample_config_input);
    defer parsed.deinit();

    const weight_set = weightsFromConfig(parsed.value);
    try std.testing.expectEqual(@as(usize, 4), weight_set.vocabSize());
    try std.testing.expectEqual(@as(usize, 2000), try weight_set.topTokenWeightMilli(2));
    try std.testing.expect(weight_set.hasValidTransitionBias());
}

test "decode binary projection bytes" {
    const allocator = std.testing.allocator;
    const bytes = [_]u8{
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0xc0, 0x3f,
        0x00, 0x00, 0x00, 0x40,
        0x00, 0x00, 0x00, 0x3f,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0xcd, 0xcc, 0xcc, 0x3d,
        0xcd, 0xcc, 0x4c, 0x3e,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0xcd, 0xcc, 0xcc, 0x3d,
        0xcd, 0xcc, 0x4c, 0x3e,
        0x00, 0x00, 0x00, 0x00,
        0xcd, 0xcc, 0x4c, 0x3f,
        0x00, 0x00, 0x00, 0x00,
        0xcd, 0xcc, 0xcc, 0x3d,
    };

    const values = try decodeValuesFromBytes(allocator, &bytes);
    defer allocator.free(values);

    try std.testing.expectEqual(@as(usize, 20), values.len);
    try std.testing.expectEqual(@as(f32, 1.5), values[1]);
    try std.testing.expectEqual(@as(f32, 2.0), values[2]);
    try std.testing.expectEqual(@as(f32, 0.8), values[17]);
}
