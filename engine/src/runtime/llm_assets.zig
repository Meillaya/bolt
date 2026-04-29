const std = @import("std");
const tokenizer = @import("../tokenizer.zig");
const weights = @import("../weights.zig");

pub const RuntimeBundle = struct {
    model_file: []const u8,
    tokenizer_file: []const u8,
    weights_file: []const u8,
};

pub const ModelConfig = struct {
    loader: []const u8,
    model_name: []const u8,
    architecture: []const u8,
    vocab_size: usize,
    hidden_size: usize,
    context_length: usize,
};

pub const sample_model_file_name = "llm-smoke.model.json";
pub const sample_tokenizer_file_name = "llm-smoke.tokenizer.json";
pub const sample_weights_file_name = "llm-smoke.weights.bin";
pub const sample_loader_name = "bolt-runtime-bundle-v1";
pub const sample_model_name = "llm-smoke-decoder";
pub const sample_architecture_name = "tiny-transition-decoder";

pub const sample_bundle_input =
    std.fmt.comptimePrint(
        \\{{
        \\  "model_file": "{s}",
        \\  "tokenizer_file": "{s}",
        \\  "weights_file": "{s}"
        \\}}
    , .{ sample_model_file_name, sample_tokenizer_file_name, sample_weights_file_name });

pub const sample_model_input =
    std.fmt.comptimePrint(
        \\{{
        \\  "loader": "{s}",
        \\  "model_name": "{s}",
        \\  "architecture": "{s}",
        \\  "vocab_size": 4,
        \\  "hidden_size": 4,
        \\  "context_length": 8
        \\}}
    , .{ sample_loader_name, sample_model_name, sample_architecture_name });

pub const LlmRuntimeAssets = struct {
    tokenizer: tokenizer.FixtureTokenizer,
    weights: weights.FixtureDecoderWeights,
    model: ModelConfig,
    weights_format: []const u8,
};

pub const LoadedLlmRuntimeAssets = struct {
    bundle: std.json.Parsed(RuntimeBundle),
    model_config: std.json.Parsed(ModelConfig),
    tokenizer_config: std.json.Parsed(tokenizer.TokenizerConfig),
    weights_config: ?std.json.Parsed(weights.WeightsConfig),
    owned_binary_values: ?[]f32,
    assets: LlmRuntimeAssets,

    pub fn deinit(self: *LoadedLlmRuntimeAssets, allocator: std.mem.Allocator) void {
        self.bundle.deinit();
        self.model_config.deinit();
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
    if (assets.model.loader.len == 0) return error.EmptyModelLoader;
    if (assets.model.model_name.len == 0) return error.EmptyModelName;
    if (assets.model.architecture.len == 0) return error.EmptyModelArchitecture;
    if (assets.model.vocab_size == 0) return error.EmptyModelVocab;
    if (assets.model.hidden_size == 0) return error.EmptyModelHiddenSize;
    if (assets.model.context_length == 0) return error.EmptyModelContextLength;
    if (assets.weights_format.len == 0) return error.EmptyWeightsFormat;
    if (assets.tokenizer.vocabSize() == 0) {
        return error.EmptyTokenizerVocab;
    }
    if (assets.weights.vocabSize() == 0) {
        return error.EmptyWeights;
    }
    if (assets.tokenizer.vocabSize() != assets.weights.vocabSize()) {
        return error.RuntimeVocabSizeMismatch;
    }
    if (assets.model.vocab_size != assets.tokenizer.vocabSize()) {
        return error.ModelTokenizerVocabSizeMismatch;
    }
    if (!assets.weights.hasValidTransitionBias()) {
        return error.InvalidTransitionBiasShape;
    }
}

pub fn parseModelFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(ModelConfig) {
    return std.json.parseFromSlice(
        ModelConfig,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
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

pub fn weightsFormatLabelForPath(path: []const u8) ![]const u8 {
    const extension = std.fs.path.extension(path);
    if (std.mem.eql(u8, extension, ".json")) return "json-config";
    if (std.mem.eql(u8, extension, ".bin")) return "binary-f32-le";
    return error.UnsupportedWeightsFormat;
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

fn loadModelFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(ModelConfig) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(input);

    return parseModelFromSlice(allocator, input);
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
    const model_path = try std.fs.path.join(
        allocator,
        &.{ bundle_dir, bundle.value.model_file },
    );
    defer allocator.free(model_path);
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

    var model_config = try loadModelFromFile(
        io,
        allocator,
        model_path,
    );
    errdefer model_config.deinit();

    var tokenizer_config = try tokenizer.loadConfigFromFile(
        io,
        allocator,
        tokenizer_path,
    );
    errdefer tokenizer_config.deinit();

    const weights_format = try weightsFormatLabelForPath(weights_path);

    var loaded = LoadedLlmRuntimeAssets{
        .bundle = bundle,
        .model_config = model_config,
        .tokenizer_config = tokenizer_config,
        .weights_config = null,
        .owned_binary_values = null,
        .assets = undefined,
    };
    errdefer loaded.deinit(allocator);

    const token_decoder = tokenizer.tokenizerFromConfig(loaded.tokenizer_config.value);

    if (std.mem.eql(u8, weights_format, "json-config")) {
        const weights_config = try weights.loadConfigFromFile(
            io,
            allocator,
            weights_path,
        );
        loaded.weights_config = weights_config;
        loaded.assets = .{
            .tokenizer = token_decoder,
            .weights = weights.weightsFromConfig(weights_config.value),
            .model = loaded.model_config.value,
            .weights_format = weights_format,
        };
    } else {
        std.debug.assert(std.mem.eql(u8, weights_format, "binary-f32-le"));
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
            .model = loaded.model_config.value,
            .weights_format = weights_format,
        };
    }

    try validate(loaded.assets);
    return loaded;
}

fn sampleRuntimeAssets() LlmRuntimeAssets {
    return .{
        .tokenizer = tokenizer.defaultTokenizer(),
        .weights = weights.defaultWeights(),
        .model = .{
            .loader = sample_loader_name,
            .model_name = sample_model_name,
            .architecture = sample_architecture_name,
            .vocab_size = 4,
            .hidden_size = 4,
            .context_length = 8,
        },
        .weights_format = "binary-f32-le",
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

    try std.testing.expectEqualStrings(sample_model_file_name, parsed.value.model_file);
    try std.testing.expectEqualStrings(sample_tokenizer_file_name, parsed.value.tokenizer_file);
    try std.testing.expectEqualStrings(sample_weights_file_name, parsed.value.weights_file);
}

test "parse model config" {
    const allocator = std.testing.allocator;

    var parsed = try parseModelFromSlice(allocator, sample_model_input);
    defer parsed.deinit();

    try std.testing.expectEqualStrings(sample_loader_name, parsed.value.loader);
    try std.testing.expectEqualStrings(sample_model_name, parsed.value.model_name);
    try std.testing.expectEqual(@as(usize, 4), parsed.value.vocab_size);
}

test "reject unsupported llm weights format label" {
    try std.testing.expectEqualStrings("json-config", try weightsFormatLabelForPath("fixture.weights.json"));
    try std.testing.expectEqualStrings("binary-f32-le", try weightsFormatLabelForPath("fixture.weights.bin"));
    try std.testing.expectError(error.UnsupportedWeightsFormat, weightsFormatLabelForPath("fixture.weights.safetensors"));
}
