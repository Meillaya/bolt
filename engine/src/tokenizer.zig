const std = @import("std");

pub const default_vocab = [_][]const u8{
    "<pad>",
    "zig",
    "metal",
    "bolt",
};

pub const TokenizerConfig = struct {
    vocab: []const []const u8,
};

pub const sample_config_input =
    \\{
    \\  "vocab": ["<pad>", "zig", "metal", "bolt"]
    \\}
;

pub const FixtureTokenizer = struct {
    vocab: []const []const u8,

    pub fn vocabSize(self: FixtureTokenizer) usize {
        return self.vocab.len;
    }

    pub fn decodeToken(self: FixtureTokenizer, token_id: usize) ![]const u8 {
        if (token_id >= self.vocab.len) {
            return error.InvalidTokenId;
        }
        return self.vocab[token_id];
    }

    pub fn encodeToken(self: FixtureTokenizer, token: []const u8) !usize {
        for (self.vocab, 0..) |candidate, index| {
            if (std.mem.eql(u8, candidate, token)) return index;
        }
        return error.UnknownToken;
    }

    pub fn encodeText(self: FixtureTokenizer, allocator: std.mem.Allocator, text: []const u8) ![]usize {
        var count_iter = std.mem.tokenizeScalar(u8, text, ' ');
        var count: usize = 0;
        while (count_iter.next()) |part| {
            if (part.len == 0) continue;
            count += 1;
        }

        const tokens = try allocator.alloc(usize, count);
        errdefer allocator.free(tokens);

        var parts = std.mem.tokenizeScalar(u8, text, ' ');
        var index: usize = 0;
        while (parts.next()) |part| {
            if (part.len == 0) continue;
            tokens[index] = try self.encodeToken(part);
            index += 1;
        }

        return tokens;
    }
};

pub fn defaultTokenizer() FixtureTokenizer {
    return .{ .vocab = default_vocab[0..] };
}

pub fn tokenizerFromConfig(config: TokenizerConfig) FixtureTokenizer {
    return .{ .vocab = config.vocab };
}

pub fn parseConfigFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(TokenizerConfig) {
    return std.json.parseFromSlice(
        TokenizerConfig,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadConfigFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(TokenizerConfig) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(input);

    return parseConfigFromSlice(allocator, input);
}

test "decode known token" {
    const tokenizer = defaultTokenizer();
    try std.testing.expectEqualStrings("metal", try tokenizer.decodeToken(2));
}

test "reject unknown token" {
    const tokenizer = defaultTokenizer();
    try std.testing.expectError(error.InvalidTokenId, tokenizer.decodeToken(99));
}

test "parse tokenizer config" {
    const allocator = std.testing.allocator;

    var parsed = try parseConfigFromSlice(allocator, sample_config_input);
    defer parsed.deinit();

    const tokenizer = tokenizerFromConfig(parsed.value);
    try std.testing.expectEqual(@as(usize, 4), tokenizer.vocabSize());
    try std.testing.expectEqualStrings("bolt", try tokenizer.decodeToken(3));
}

test "encode known token" {
    const tokenizer = defaultTokenizer();
    try std.testing.expectEqual(@as(usize, 1), try tokenizer.encodeToken("zig"));
}

test "encode text" {
    const tokenizer = defaultTokenizer();
    const allocator = std.testing.allocator;
    const encoded = try tokenizer.encodeText(allocator, "zig metal bolt");
    defer allocator.free(encoded);
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, encoded);
}
