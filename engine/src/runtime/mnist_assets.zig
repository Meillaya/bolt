const std = @import("std");
const weights = @import("../weights.zig");

pub const input_size: usize = 784;
pub const output_size: usize = 10;

pub const RuntimeBundle = struct {
    weights_file: []const u8,
};

pub const FixtureMnistWeights = struct {
    dense_weights: []const f32,
    bias: []const f32,

    pub fn validate(self: FixtureMnistWeights) !void {
        if (self.dense_weights.len != input_size * output_size) {
            return error.InvalidMnistDenseWeightShape;
        }
        if (self.bias.len != output_size) {
            return error.InvalidMnistBiasShape;
        }
    }
};

pub const LoadedMnistRuntimeAssets = struct {
    bundle: std.json.Parsed(RuntimeBundle),
    owned_binary_values: []f32,
    assets: FixtureMnistWeights,

    pub fn deinit(self: *LoadedMnistRuntimeAssets, allocator: std.mem.Allocator) void {
        self.bundle.deinit();
        allocator.free(self.owned_binary_values);
    }
};

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

pub fn weightsFromSlices(values: []const f32) !FixtureMnistWeights {
    const dense_len = input_size * output_size;
    const expected_len = dense_len + output_size;
    if (values.len != expected_len) {
        return error.InvalidBinaryWeightsShape;
    }

    const assets = FixtureMnistWeights{
        .dense_weights = values[0..dense_len],
        .bias = values[dense_len..expected_len],
    };
    try assets.validate();
    return assets;
}

pub fn loadFromManifestContext(
    io: std.Io,
    allocator: std.mem.Allocator,
    context: anytype,
) !LoadedMnistRuntimeAssets {
    const runtime_bundle_path = context.runtime_bundle_path orelse return error.MissingRuntimeBundlePath;
    var bundle = try loadBundleFromFile(
        io,
        allocator,
        runtime_bundle_path,
    );
    errdefer bundle.deinit();

    const bundle_dir = std.fs.path.dirname(runtime_bundle_path) orelse ".";
    const weights_path = try std.fs.path.join(
        allocator,
        &.{ bundle_dir, bundle.value.weights_file },
    );
    defer allocator.free(weights_path);

    const values = try weights.loadValuesFromBinaryFile(
        io,
        allocator,
        weights_path,
    );
    errdefer allocator.free(values);

    return .{
        .bundle = bundle,
        .owned_binary_values = values,
        .assets = try weightsFromSlices(values),
    };
}

test "parse mnist runtime bundle" {
    const allocator = std.testing.allocator;
    var parsed = try parseBundleFromSlice(allocator,
        \\{ "weights_file": "mnist-smoke.weights.bin" }
    );
    defer parsed.deinit();

    try std.testing.expectEqualStrings("mnist-smoke.weights.bin", parsed.value.weights_file);
}
