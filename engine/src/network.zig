const std = @import("std");
const layout = @import("tensor/layout.zig");

pub const Activation = layout.Activation;
pub const LayerDesc = layout.LayerDesc;
pub const NetworkLayout = layout.NetworkLayout;

pub const TrainingMetrics = struct {
    schema_version: u32 = 1,
    backend: []const u8 = "cpu-reference",
    epochs: usize,
    samples: usize,
    final_loss: f32,
    accuracy_milli: usize,
};

pub fn activationForward(kind: Activation, x: f32) f32 {
    return switch (kind) {
        .relu => @max(x, 0.0),
        .tanh_act => std.math.tanh(x),
        .sigmoid => 1.0 / (1.0 + @exp(-x)),
        .none => x,
    };
}

pub fn activationBackward(kind: Activation, pre_activation: f32, activated: f32) f32 {
    return switch (kind) {
        .relu => if (pre_activation > 0.0) 1.0 else 0.0,
        .tanh_act => 1.0 - activated * activated,
        .sigmoid => activated * (1.0 - activated),
        .none => 1.0,
    };
}

pub fn softmaxInPlace(values: []f32) !void {
    if (values.len == 0) return error.EmptyBuffer;
    var max_value = values[0];
    for (values[1..]) |value| max_value = @max(max_value, value);
    var sum: f32 = 0.0;
    for (values) |*value| {
        value.* = @exp(value.* - max_value);
        sum += value.*;
    }
    if (sum == 0.0) return error.InvalidLoss;
    for (values) |*value| value.* /= sum;
}

pub fn crossEntropy(probs: []const f32, label: usize) !f32 {
    if (label >= probs.len) return error.LabelOutOfRange;
    return -@log(@max(probs[label], 1.0e-9));
}

pub fn CpuNetwork(comptime Layout: type) type {
    return struct {
        const Self = @This();

        allocator: std.mem.Allocator,
        params: []f32,
        grads: []f32,
        pre_activations: []f32,
        post_activations: []f32,
        deltas: []f32,

        pub fn init(allocator: std.mem.Allocator) !Self {
            const activation_slots = @as(usize, Layout.num_layers) * @as(usize, Layout.max_activation_size);
            const params = try allocator.alloc(f32, Layout.param_count);
            errdefer allocator.free(params);
            const grads = try allocator.alloc(f32, Layout.param_count);
            errdefer allocator.free(grads);
            const pre = try allocator.alloc(f32, activation_slots);
            errdefer allocator.free(pre);
            const post = try allocator.alloc(f32, activation_slots);
            errdefer allocator.free(post);
            const deltas = try allocator.alloc(f32, activation_slots);
            errdefer allocator.free(deltas);
            @memset(params, 0.0);
            @memset(grads, 0.0);
            @memset(pre, 0.0);
            @memset(post, 0.0);
            @memset(deltas, 0.0);
            return .{
                .allocator = allocator,
                .params = params,
                .grads = grads,
                .pre_activations = pre,
                .post_activations = post,
                .deltas = deltas,
            };
        }

        pub fn deinit(self: *Self) void {
            self.allocator.free(self.deltas);
            self.allocator.free(self.post_activations);
            self.allocator.free(self.pre_activations);
            self.allocator.free(self.grads);
            self.allocator.free(self.params);
            self.* = undefined;
        }

        pub fn initializeDeterministic(self: *Self) void {
            for (self.params, 0..) |*value, index| {
                const bucket = @as(f32, @floatFromInt((index % 7) + 1));
                value.* = (bucket - 4.0) * 0.05;
            }
        }

        fn actSlice(storage: []f32, layer_index: usize) []f32 {
            const base = layer_index * @as(usize, Layout.max_activation_size);
            const len = @as(usize, Layout.activation_sizes[layer_index]);
            return storage[base..][0..len];
        }

        fn weightSlice(params: []f32, layer_index: usize) []f32 {
            const offset = @as(usize, Layout.weight_offsets[layer_index]);
            const len = @as(usize, Layout.weight_counts[layer_index]);
            return params[offset..][0..len];
        }

        fn biasSlice(params: []f32, layer_index: usize) []f32 {
            const offset = @as(usize, Layout.bias_offsets[layer_index]);
            const len = @as(usize, Layout.bias_counts[layer_index]);
            return params[offset..][0..len];
        }

        pub fn forward(self: *Self, input: []const f32, output: []f32) !void {
            if (input.len != Layout.input_size) return error.LengthMismatch;
            if (output.len != Layout.output_size) return error.LengthMismatch;

            inline for (0..Layout.num_layers) |layer_index| {
                const layer = Layout.layers[layer_index];
                const weights = weightSlice(self.params, layer_index);
                const biases = biasSlice(self.params, layer_index);
                const pre = actSlice(self.pre_activations, layer_index);
                const post = actSlice(self.post_activations, layer_index);
                const layer_input = if (layer_index == 0) input else actSlice(self.post_activations, layer_index - 1);

                for (0..layer.out) |out_index| {
                    var sum = biases[out_index];
                    for (0..layer.in) |in_index| {
                        sum += layer_input[in_index] * weights[in_index * layer.out + out_index];
                    }
                    pre[out_index] = sum;
                    post[out_index] = activationForward(layer.act, sum);
                }
            }

            const last = actSlice(self.post_activations, Layout.num_layers - 1);
            @memcpy(output, last[0..Layout.output_size]);
        }

        pub fn zeroGrad(self: *Self) void {
            @memset(self.grads, 0.0);
        }

        pub fn trainClassSample(self: *Self, input: []const f32, label: usize, learning_rate: f32) !f32 {
            var logits: [Layout.output_size]f32 = undefined;
            try self.forward(input, &logits);
            try softmaxInPlace(&logits);
            const loss = try crossEntropy(&logits, label);
            self.zeroGrad();

            const last_index = Layout.num_layers - 1;
            const last_delta = actSlice(self.deltas, last_index);
            for (0..Layout.output_size) |i| {
                last_delta[i] = logits[i] - (if (i == label) @as(f32, 1.0) else @as(f32, 0.0));
                const pre = actSlice(self.pre_activations, last_index)[i];
                const post = actSlice(self.post_activations, last_index)[i];
                last_delta[i] *= activationBackward(Layout.layers[last_index].act, pre, post);
            }

            var layer_rev: usize = Layout.num_layers;
            while (layer_rev > 0) {
                layer_rev -= 1;
                const layer = Layout.layers[layer_rev];
                const delta = actSlice(self.deltas, layer_rev);
                const layer_input = if (layer_rev == 0) input else actSlice(self.post_activations, layer_rev - 1);
                const weights = weightSlice(self.params, layer_rev);
                const grad_w = weightSlice(self.grads, layer_rev);
                const grad_b = biasSlice(self.grads, layer_rev);

                if (layer_rev > 0) {
                    const prev_layer = Layout.layers[layer_rev - 1];
                    const prev_delta = actSlice(self.deltas, layer_rev - 1);
                    @memset(prev_delta, 0.0);
                    for (0..prev_layer.out) |in_index| {
                        var upstream: f32 = 0.0;
                        for (0..layer.out) |out_index| {
                            upstream += delta[out_index] * weights[in_index * layer.out + out_index];
                        }
                        const pre = actSlice(self.pre_activations, layer_rev - 1)[in_index];
                        const post = actSlice(self.post_activations, layer_rev - 1)[in_index];
                        prev_delta[in_index] = upstream * activationBackward(prev_layer.act, pre, post);
                    }
                }

                for (0..layer.in) |in_index| {
                    for (0..layer.out) |out_index| {
                        grad_w[in_index * layer.out + out_index] += layer_input[in_index] * delta[out_index];
                    }
                }
                for (0..layer.out) |out_index| grad_b[out_index] += delta[out_index];
            }

            for (self.params, self.grads) |*param, grad| {
                param.* -= learning_rate * grad;
            }
            return loss;
        }

        pub fn predictClass(self: *Self, input: []const f32) !usize {
            var logits: [Layout.output_size]f32 = undefined;
            try self.forward(input, &logits);
            var best: usize = 0;
            for (logits[1..], 1..) |value, index| {
                if (value > logits[best]) best = index;
            }
            return best;
        }
    };
}

pub fn trainSyntheticMiniClassifier(allocator: std.mem.Allocator, epochs: usize, learning_rate: f32) !TrainingMetrics {
    const arch = [_]LayerDesc{
        .{ .in = 2, .out = 4, .act = .tanh_act },
        .{ .in = 4, .out = 2, .act = .none },
    };
    const Layout = NetworkLayout(&arch);
    var net = try CpuNetwork(Layout).init(allocator);
    defer net.deinit();
    net.initializeDeterministic();

    const samples = [_][2]f32{
        .{ 0.0, 0.0 },
        .{ 0.0, 1.0 },
        .{ 1.0, 0.0 },
        .{ 1.0, 1.0 },
    };
    const labels = [_]usize{ 0, 1, 1, 1 };

    var final_loss: f32 = 0.0;
    for (0..epochs) |_| {
        final_loss = 0.0;
        for (samples, labels) |sample, label| {
            final_loss += try net.trainClassSample(&sample, label, learning_rate);
        }
        final_loss /= @as(f32, @floatFromInt(samples.len));
    }

    var correct: usize = 0;
    for (samples, labels) |sample, label| {
        if (try net.predictClass(&sample) == label) correct += 1;
    }

    return .{
        .epochs = epochs,
        .samples = samples.len,
        .final_loss = final_loss,
        .accuracy_milli = correct * 1000 / samples.len,
    };
}

test "activation forward and backward paths match CPU reference values" {
    try std.testing.expectEqual(@as(f32, 0.0), activationForward(.relu, -1.0));
    try std.testing.expectEqual(@as(f32, 2.0), activationForward(.relu, 2.0));
    try std.testing.expectEqual(@as(f32, 0.0), activationBackward(.relu, -1.0, 0.0));
    try std.testing.expectEqual(@as(f32, 1.0), activationBackward(.relu, 1.0, 1.0));
    const sig = activationForward(.sigmoid, 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sig, 0.00001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), activationBackward(.sigmoid, 0.0, sig), 0.00001);
}

test "softmax and cross entropy are stable" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    try softmaxInPlace(&logits);
    var sum: f32 = 0.0;
    for (logits) |value| sum += value;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 0.00001);
    try std.testing.expect((try crossEntropy(&logits, 2)) < 0.5);
}

test "synthetic mini-network training golden converges" {
    const metrics = try trainSyntheticMiniClassifier(std.testing.allocator, 500, 0.1);
    try std.testing.expectEqual(@as(usize, 4), metrics.samples);
    try std.testing.expect(metrics.final_loss < 0.20);
    try std.testing.expect(metrics.accuracy_milli >= 1000);
}
