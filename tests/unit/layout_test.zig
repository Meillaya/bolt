const std = @import("std");
const bolt = @import("bolt");

test "layout shape matches the compact proof image surface" {
    const shape = bolt.layout.Shape{ .rows = 28, .cols = 28 };
    try std.testing.expectEqual(@as(usize, 784), shape.elementCount());
}
