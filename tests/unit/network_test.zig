const std = @import("std");
const bolt = @import("bolt");

test "reference-style NetworkLayout offsets and sizes match M2 contract" {
    const arch = [_]bolt.network.LayerDesc{
        .{ .in = 3, .out = 4, .act = .relu },
        .{ .in = 4, .out = 2, .act = .none },
    };
    const Layout = bolt.network.NetworkLayout(&arch);
    try std.testing.expectEqual(@as(u32, 2), Layout.num_layers);
    try std.testing.expectEqual(@as(u32, 12), Layout.weight_counts[0]);
    try std.testing.expectEqual(@as(u32, 4), Layout.bias_counts[0]);
    try std.testing.expectEqual(@as(u32, 8), Layout.weight_counts[1]);
    try std.testing.expectEqual(@as(u32, 2), Layout.bias_counts[1]);
    try std.testing.expectEqual(@as(u32, 0), Layout.weight_offsets[0]);
    try std.testing.expectEqual(@as(u32, 12), Layout.bias_offsets[0]);
    try std.testing.expectEqual(@as(u32, 16), Layout.weight_offsets[1]);
    try std.testing.expectEqual(@as(u32, 24), Layout.bias_offsets[1]);
    try std.testing.expectEqual(@as(u32, 26), Layout.param_count);
    try std.testing.expectEqual(@as(u32, 4), Layout.max_activation_size);
    try std.testing.expect(Layout.packed_weight_bytes[0] > 0);
}

test "CPU network forward backward update learns a class sample" {
    const arch = [_]bolt.network.LayerDesc{
        .{ .in = 2, .out = 3, .act = .tanh_act },
        .{ .in = 3, .out = 2, .act = .none },
    };
    const Layout = bolt.network.NetworkLayout(&arch);
    var net = try bolt.network.CpuNetwork(Layout).init(std.testing.allocator);
    defer net.deinit();
    net.initializeDeterministic();

    const sample = [_]f32{ 1.0, 0.0 };
    const before = try net.trainClassSample(&sample, 1, 0.1);
    var after = before;
    for (0..200) |_| after = try net.trainClassSample(&sample, 1, 0.1);
    try std.testing.expect(after < before);
    try std.testing.expectEqual(@as(usize, 1), try net.predictClass(&sample));
}

test "synthetic mini-network training golden passes" {
    const metrics = try bolt.network.trainSyntheticMiniClassifier(std.testing.allocator, 500, 0.1);
    try std.testing.expectEqualStrings("cpu-reference", metrics.backend);
    try std.testing.expect(metrics.final_loss < 0.20);
    try std.testing.expect(metrics.accuracy_milli >= 1000);
}
