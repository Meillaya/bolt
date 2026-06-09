const std = @import("std");

pub fn bitWordCount(bit_count: usize) usize {
    return (bit_count + 63) / 64;
}

pub fn packBinary(allocator: std.mem.Allocator, values: []const f32, threshold: f32) ![]u64 {
    const words = try allocator.alloc(u64, bitWordCount(values.len));
    @memset(words, 0);
    for (values, 0..) |value, index| {
        if (value > threshold) {
            words[index / 64] |= @as(u64, 1) << @as(u6, @intCast(index % 64));
        }
    }
    return words;
}

pub fn hammingDistancePacked(a: []const u64, b: []const u64) usize {
    std.debug.assert(a.len == b.len);
    var distance: usize = 0;
    for (a, b) |wa, wb| distance += @popCount(wa ^ wb);
    return distance;
}

pub fn hammingDistanceThreshold(a: []const f32, b: []const f32, threshold: f32, best_limit: usize) usize {
    std.debug.assert(a.len == b.len);
    var distance: usize = 0;
    for (a, b) |x, y| {
        if ((x > threshold) != (y > threshold)) {
            distance += 1;
            if (distance >= best_limit) break;
        }
    }
    return distance;
}

test "binary packing and hamming distance match threshold reference" {
    const a = [_]f32{ 0.0, 0.7, 0.2, 0.9, 0.51, 0.49 };
    const b = [_]f32{ 0.0, 0.1, 0.8, 0.9, 0.2, 0.6 };
    const packed_a = try packBinary(std.testing.allocator, &a, 0.5);
    defer std.testing.allocator.free(packed_a);
    const packed_b = try packBinary(std.testing.allocator, &b, 0.5);
    defer std.testing.allocator.free(packed_b);
    try std.testing.expectEqual(@as(usize, 3), hammingDistancePacked(packed_a, packed_b));
    try std.testing.expectEqual(@as(usize, 3), hammingDistanceThreshold(&a, &b, 0.5, std.math.maxInt(usize)));
}
