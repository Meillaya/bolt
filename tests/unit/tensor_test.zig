const std = @import("std");
const bolt = @import("bolt");

test "mnist model input size matches tensor layout element count" {
    const shape = bolt.layout.Shape{ .rows = 28, .cols = 28 };
    const model = bolt.mnist.MnistModel{};
    try std.testing.expectEqual(model.input_size, shape.elementCount());
}
