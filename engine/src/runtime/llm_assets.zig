const std = @import("std");
const tokenizer = @import("../tokenizer.zig");
const weights = @import("../weights.zig");

pub const RuntimeBundle = struct {
    tokenizer_file: []const u8,
    weights_file: []const u8,
};

pub const sample_tokenizer_file_name = "llm-smoke.tokenizer.json";
pub const sample_weights_file_name = "llm-smoke.weights.bin";

pub const sample_bundle_input =
    std.fmt.comptimePrint(
        \\{{
        \\  "tokenizer_file": "{s}",
        \\  "weights_file": "{s}"
        \\}}
    , .{ sample_tokenizer_file_name, sample_weights_file_name });

pub const LlmRuntimeAssets = struct {
    tokenizer: tokenizer.FixtureTokenizer,
    weights: weights.FixtureDecoderWeights,
};

pub const LoadedLlmRuntimeAssets = struct {
    bundle: std.json.Parsed(RuntimeBundle),
    tokenizer_config: std.json.Parsed(tokenizer.TokenizerConfig),
    weights_config: ?std.json.Parsed(weights.WeightsConfig),
    owned_binary_values: ?[]f32,
    assets: LlmRuntimeAssets,

    pub fn deinit(self: *LoadedLlmRuntimeAssets, allocator: std.mem.Allocator) void {
        self.bundle.deinit();
        self.tokenizer_config.deinit();
        if (self.weights_config) |*weights_config| {
            weights_config.deinit();
        }
        if (self.owned_binary_values) |owned_binary_values| {
            allocator.free(owned_binary_values);
        }
    }
};

pub fn validate(assets: LlmRuntimeAssets) !void {
    if (assets.tokenizer.vocabSize() == 0) {
        return error.EmptyTokenizerVocab;
    }
    if (assets.weights.vocabSize() == 0) {
        return error.EmptyWeights;
    }
    if (assets.tokenizer.vocabSize() != assets.weights.vocabSize()) {
        return error.RuntimeVocabSizeMismatch;
    }
    if (!assets.weights.hasValidTransitionBias()) {
        return error.InvalidTransitionBiasShape;
    }
}

pub fn parseBundleFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(RuntimeBundle) {
    return std.json.parseFromSlice(
        RuntimeBundle,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

fn loadBundleFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(RuntimeBundle) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(input);

    return parseBundleFromSlice(allocator, input);
}

pub fn loadFromManifestContext(
    io: std.Io,
    allocator: std.mem.Allocator,
    context: anytype,
) !LoadedLlmRuntimeAssets {
    const runtime_bundle_path = context.runtime_bundle_path orelse return error.MissingRuntimeBundlePath;
    var bundle = try loadBundleFromFile(
        io,
        allocator,
        runtime_bundle_path,
    );
    errdefer bundle.deinit();

    const bundle_dir = std.fs.path.dirname(runtime_bundle_path) orelse ".";
    const tokenizer_path = try std.fs.path.join(
        allocator,
        &.{ bundle_dir, bundle.value.tokenizer_file },
    );
    defer allocator.free(tokenizer_path);
    const weights_path = try std.fs.path.join(
        allocator,
        &.{ bundle_dir, bundle.value.weights_file },
    );
    defer allocator.free(weights_path);

    var tokenizer_config = try tokenizer.loadConfigFromFile(
        io,
        allocator,
        tokenizer_path,
    );
    errdefer tokenizer_config.deinit();

    const weights_extension = std.fs.path.extension(weights_path);

    var loaded = LoadedLlmRuntimeAssets{
        .bundle = bundle,
        .tokenizer_config = tokenizer_config,
        .weights_config = null,
        .owned_binary_values = null,
        .assets = undefined,
    };
    errdefer loaded.deinit(allocator);

    const token_decoder = tokenizer.tokenizerFromConfig(loaded.tokenizer_config.value);

    if (std.mem.eql(u8, weights_extension, ".json")) {
        const weights_config = try weights.loadConfigFromFile(
            io,
            allocator,
            weights_path,
        );
        loaded.weights_config = weights_config;
        loaded.assets = .{
            .tokenizer = token_decoder,
            .weights = weights.weightsFromConfig(weights_config.value),
        };
    } else if (std.mem.eql(u8, weights_extension, ".bin")) {
        const values = try weights.loadValuesFromBinaryFile(
            io,
            allocator,
            weights_path,
        );
        const vocab_size = token_decoder.vocabSize();
        const expected_value_count = vocab_size + (vocab_size * vocab_size);
        if (values.len != expected_value_count) {
            return error.InvalidBinaryWeightsShape;
        }
        loaded.owned_binary_values = values;
        loaded.assets = .{
            .tokenizer = token_decoder,
            .weights = weights.weightsFromSlices(
                values[0..vocab_size],
                values[vocab_size..],
            ),
        };
    } else {
        return error.UnsupportedWeightsFormat;
    }

    try validate(loaded.assets);
    return loaded;
}

fn sampleRuntimeAssets() LlmRuntimeAssets {
    return .{
        .tokenizer = tokenizer.defaultTokenizer(),
        .weights = weights.defaultWeights(),
    };
}

test "validate matching llm runtime assets" {
    try validate(sampleRuntimeAssets());
}

test "reject runtime vocab mismatch" {
    var assets = sampleRuntimeAssets();
    assets.tokenizer = .{ .vocab = &.{ "<pad>", "zig" } };
    try std.testing.expectError(error.RuntimeVocabSizeMismatch, validate(assets));
}

test "parse runtime bundle" {
    const allocator = std.testing.allocator;

    var parsed = try parseBundleFromSlice(allocator, sample_bundle_input);
    defer parsed.deinit();

    try std.testing.expectEqualStrings(sample_tokenizer_file_name, parsed.value.tokenizer_file);
    try std.testing.expectEqualStrings(sample_weights_file_name, parsed.value.weights_file);
}
