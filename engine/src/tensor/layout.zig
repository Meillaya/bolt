const std = @import("std");

pub const Shape = struct {
    rows: usize,
    cols: usize,

    pub fn elementCount(self: Shape) usize {
        return self.rows * self.cols;
    }
};

test "shape element count" {
    const shape = Shape{ .rows = 28, .cols = 28 };
    try std.testing.expectEqual(@as(usize, 784), shape.elementCount());
}
