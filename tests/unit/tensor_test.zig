const std = @import("std");
const bolt = @import("bolt");

test "mnist model input size matches tensor layout element count" {
    const shape = bolt.layout.Shape{ .rows = 28, .cols = 28 };
    const model = bolt.mnist.MnistModel{};
    try std.testing.expectEqual(model.input_size, shape.elementCount());
}

test "tensor buffer round-trips through a shared Metal buffer without host output allocation" {
    if (!bolt.context.Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var tensor = try bolt.buffer.OwnedTensorF32.initCopy(
        allocator,
        .{ .rows = 2, .cols = 2 },
        &.{ 1.0, 0.0, 2.0, 4.0 },
        .host,
    );
    defer tensor.deinit();

    var round_tripped = try tensor.roundTripThroughMetal();
    defer round_tripped.deinit();

    try std.testing.expectEqual(bolt.buffer.Storage.metal_shared, round_tripped.storage);
    try std.testing.expectEqual(@as(?[]f32, null), round_tripped.host_values);
    try std.testing.expect(round_tripped.metal_buffer != null);
    try std.testing.expectEqualSlices(f32, tensor.values, round_tripped.values);
}

test "tensor buffer supports explicit host materialization from shared Metal storage" {
    if (!bolt.context.Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var tensor = try bolt.buffer.OwnedTensorF32.initCopy(
        allocator,
        .{ .rows = 1, .cols = 4 },
        &.{ 0.25, 0.5, 0.75, 1.0 },
        .metal_shared,
    );
    defer tensor.deinit();

    var host_tensor = try tensor.materializeToHost();
    defer host_tensor.deinit();

    try std.testing.expectEqual(bolt.buffer.Storage.host, host_tensor.storage);
    try std.testing.expect(host_tensor.host_values != null);
    try std.testing.expectEqualSlices(f32, tensor.values, host_tensor.values);
}
