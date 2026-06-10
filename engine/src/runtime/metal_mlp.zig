//! Minimal dense MLP inference runtime built on Bolt's Zig/Metal primitives.
//!
//! This module deliberately stays small: it proves that Bolt can compose its
//! Metal matmul, bias-add, ReLU, and softmax kernels into a model-level runtime
//! with CPU parity tests before larger model runtimes reuse the same boundary.

const std = @import("std");
const metal = @import("../metal/context.zig");
const network = @import("../network.zig");
const layout = @import("../tensor/layout.zig");

pub const Activation = layout.Activation;

pub const DenseLayer = struct {
    input_size: usize,
    output_size: usize,
    weights: []const f32,
    bias: []const f32,
    activation: Activation = .relu,

    pub fn validate(self: DenseLayer) !void {
        if (self.input_size == 0 or self.output_size == 0) return error.EmptyLayer;
        if (self.weights.len != self.input_size * self.output_size) return error.LengthMismatch;
        if (self.bias.len != self.output_size) return error.LengthMismatch;
    }
};

pub const InferenceResult = struct {
    allocator: std.mem.Allocator,
    backend: []const u8,
    logits: []f32,
    probabilities: []f32,
    predicted_class: usize,

    pub fn deinit(self: *InferenceResult) void {
        self.allocator.free(self.probabilities);
        self.allocator.free(self.logits);
        self.* = undefined;
    }
};

pub const DemoModel = struct {
    input: [2]f32,
    layers: [2]DenseLayer,
};

const demo_layer0_weights = [_]f32{
    1.0,  -0.5, 0.75,
    -1.0, 1.0,  0.5,
};
const demo_layer0_bias = [_]f32{ 0.0, 0.25, -0.1 };
const demo_layer1_weights = [_]f32{
    1.0,   -1.0,
    -0.25, 0.75,
    0.5,   0.25,
};
const demo_layer1_bias = [_]f32{ -0.1, 0.2 };

pub fn demoModel() DemoModel {
    return .{
        .input = .{ 0.75, 0.25 },
        .layers = .{
            .{
                .input_size = 2,
                .output_size = 3,
                .weights = &demo_layer0_weights,
                .bias = &demo_layer0_bias,
                .activation = .relu,
            },
            .{
                .input_size = 3,
                .output_size = 2,
                .weights = &demo_layer1_weights,
                .bias = &demo_layer1_bias,
                .activation = .none,
            },
        },
    };
}

fn validateNetwork(layers: []const DenseLayer, input_len: usize) !usize {
    if (layers.len == 0) return error.EmptyNetwork;
    if (input_len != layers[0].input_size) return error.LengthMismatch;

    var expected_input = input_len;
    for (layers) |layer| {
        try layer.validate();
        if (layer.input_size != expected_input) return error.LayerShapeMismatch;
        expected_input = layer.output_size;
    }
    return layers[layers.len - 1].output_size;
}

fn applyCpuActivation(kind: Activation, values: []f32) !void {
    switch (kind) {
        .relu, .none, .sigmoid, .tanh_act => {
            for (values) |*value| value.* = network.activationForward(kind, value.*);
        },
    }
}

fn applyMetalActivation(context: metal.Context, kind: Activation, input: []const f32, output: []f32) !void {
    switch (kind) {
        .relu => try context.reluF32(input, output),
        .sigmoid => try context.sigmoidF32(input, output),
        .tanh_act => try context.tanhF32(input, output),
        .none => @memcpy(output, input),
    }
}

pub fn forwardCpu(
    allocator: std.mem.Allocator,
    layers: []const DenseLayer,
    input: []const f32,
) ![]f32 {
    _ = try validateNetwork(layers, input.len);

    var current = try allocator.dupe(f32, input);
    errdefer allocator.free(current);

    for (layers) |layer| {
        const next = try allocator.alloc(f32, layer.output_size);
        errdefer allocator.free(next);

        for (0..layer.output_size) |out_index| {
            var sum = layer.bias[out_index];
            for (0..layer.input_size) |in_index| {
                sum += current[in_index] * layer.weights[in_index * layer.output_size + out_index];
            }
            next[out_index] = sum;
        }
        try applyCpuActivation(layer.activation, next);

        allocator.free(current);
        current = next;
    }

    return current;
}

pub fn forwardMetal(
    allocator: std.mem.Allocator,
    layers: []const DenseLayer,
    input: []const f32,
) ![]f32 {
    if (!metal.Context.isAvailable()) return error.MetalUnavailable;
    var context = try metal.Context.init();
    defer context.deinit();
    return forwardMetalWithContext(allocator, context, layers, input);
}

pub fn forwardMetalWithContext(
    allocator: std.mem.Allocator,
    context: metal.Context,
    layers: []const DenseLayer,
    input: []const f32,
) ![]f32 {
    _ = try validateNetwork(layers, input.len);

    var current = try allocator.dupe(f32, input);
    errdefer allocator.free(current);

    for (layers) |layer| {
        const linear = try allocator.alloc(f32, layer.output_size);
        errdefer allocator.free(linear);
        const biased = try allocator.alloc(f32, layer.output_size);
        errdefer allocator.free(biased);
        const next = try allocator.alloc(f32, layer.output_size);
        errdefer allocator.free(next);

        try context.matMulF32(current, layer.weights, linear, 1, layer.output_size, layer.input_size);
        try context.biasAddF32(linear, layer.bias, biased, 1, layer.output_size);
        try applyMetalActivation(context, layer.activation, biased, next);

        allocator.free(linear);
        allocator.free(biased);
        allocator.free(current);
        current = next;
    }

    return current;
}

pub fn classifyCpu(
    allocator: std.mem.Allocator,
    layers: []const DenseLayer,
    input: []const f32,
) !InferenceResult {
    const logits = try forwardCpu(allocator, layers, input);
    errdefer allocator.free(logits);
    return finishClassification(allocator, "cpu-reference", logits);
}

pub fn classifyMetal(
    allocator: std.mem.Allocator,
    layers: []const DenseLayer,
    input: []const f32,
) !InferenceResult {
    const logits = try forwardMetal(allocator, layers, input);
    errdefer allocator.free(logits);
    return finishClassification(allocator, "metal", logits);
}

fn finishClassification(
    allocator: std.mem.Allocator,
    backend: []const u8,
    logits: []f32,
) !InferenceResult {
    if (logits.len == 0) return error.EmptyBuffer;
    const probabilities = try allocator.dupe(f32, logits);
    errdefer allocator.free(probabilities);
    try network.softmaxInPlace(probabilities);

    var predicted_class: usize = 0;
    for (probabilities[1..], 1..) |probability, index| {
        if (probability > probabilities[predicted_class]) predicted_class = index;
    }

    return .{
        .allocator = allocator,
        .backend = backend,
        .logits = logits,
        .probabilities = probabilities,
        .predicted_class = predicted_class,
    };
}

pub fn maxAbsDiff(left: []const f32, right: []const f32) !f32 {
    if (left.len != right.len) return error.LengthMismatch;
    var max_diff: f32 = 0.0;
    for (left, right) |a, b| max_diff = @max(max_diff, @abs(a - b));
    return max_diff;
}

pub fn runDemoCpu(allocator: std.mem.Allocator) !InferenceResult {
    const model = demoModel();
    return classifyCpu(allocator, &model.layers, &model.input);
}

pub fn runDemoMetal(allocator: std.mem.Allocator) !InferenceResult {
    const model = demoModel();
    return classifyMetal(allocator, &model.layers, &model.input);
}

test "dense layer validation catches shape contract violations" {
    const bad = DenseLayer{
        .input_size = 2,
        .output_size = 3,
        .weights = &.{ 1.0, 2.0 },
        .bias = &.{ 0.0, 0.0, 0.0 },
    };
    try std.testing.expectError(error.LengthMismatch, bad.validate());

    const model = demoModel();
    try std.testing.expectError(error.LengthMismatch, validateNetwork(&model.layers, 3));
}

test "CPU MLP runtime produces deterministic logits and classification" {
    const allocator = std.testing.allocator;
    const model = demoModel();
    var result = try classifyCpu(allocator, &model.layers, &model.input);
    defer result.deinit();

    try std.testing.expectEqualStrings("cpu-reference", result.backend);
    try std.testing.expectEqual(@as(usize, 0), result.predicted_class);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6625), result.logits[0], 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.059375), result.logits[1], 0.00001);

    var probability_sum: f32 = 0.0;
    for (result.probabilities) |probability| probability_sum += probability;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), probability_sum, 0.00001);
    try std.testing.expect(result.probabilities[0] > result.probabilities[1]);
}

test "Metal MLP runtime matches CPU logits" {
    if (!metal.Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const model = demoModel();
    var cpu = try classifyCpu(allocator, &model.layers, &model.input);
    defer cpu.deinit();
    var gpu = try classifyMetal(allocator, &model.layers, &model.input);
    defer gpu.deinit();

    try std.testing.expectEqualStrings("metal", gpu.backend);
    try std.testing.expectEqual(cpu.predicted_class, gpu.predicted_class);
    try std.testing.expect((try maxAbsDiff(cpu.logits, gpu.logits)) < 0.00001);
    try std.testing.expect((try maxAbsDiff(cpu.probabilities, gpu.probabilities)) < 0.00001);
}

test "Metal MLP runtime matches CPU for sigmoid and tanh activations" {
    if (!metal.Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const layers = [_]DenseLayer{
        .{
            .input_size = 2,
            .output_size = 3,
            .weights = &.{ 1.0, -0.5, 0.25, -0.75, 0.5, 1.25 },
            .bias = &.{ 0.1, -0.2, 0.3 },
            .activation = .sigmoid,
        },
        .{
            .input_size = 3,
            .output_size = 2,
            .weights = &.{ 0.5, -0.25, 1.0, 0.75, -0.5, 0.25 },
            .bias = &.{ -0.1, 0.2 },
            .activation = .tanh_act,
        },
    };
    const input = [_]f32{ -0.5, 1.25 };

    const cpu = try forwardCpu(allocator, &layers, &input);
    defer allocator.free(cpu);
    const gpu = try forwardMetal(allocator, &layers, &input);
    defer allocator.free(gpu);

    try std.testing.expect((try maxAbsDiff(cpu, gpu)) < 0.00001);
}
