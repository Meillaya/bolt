const std = @import("std");
const bolt = @import("bolt");

fn expectApproxSlices(expected: []const f32, actual: []const f32) !void {
    try std.testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |expected_value, actual_value| {
        try std.testing.expectApproxEqAbs(expected_value, actual_value, 0.0001);
    }
}

test "Metal kernel wrappers multiply matrices" {
    if (!bolt.context.Context.isAvailable()) return error.SkipZigTest;

    var context = try bolt.context.Context.init();
    defer context.deinit();

    const left = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const right = [_]f32{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };

    try context.matMulF32(&left, &right, &output, 2, 2, 3);
    try std.testing.expectEqualSlices(f32, &.{ 58.0, 64.0, 139.0, 154.0 }, &output);
}

test "Metal kernel wrappers apply bias and relu" {
    if (!bolt.context.Context.isAvailable()) return error.SkipZigTest;

    var context = try bolt.context.Context.init();
    defer context.deinit();

    const input = [_]f32{ -1.0, 2.0, -3.0, 4.0, 5.0, -6.0 };
    const bias = [_]f32{ 1.0, -2.0, 3.0 };
    var biased = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    var activated = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    try context.biasAddF32(&input, &bias, &biased, 2, 3);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 0.0, 0.0, 5.0, 3.0, -3.0 }, &biased);

    try context.reluF32(&biased, &activated);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 0.0, 0.0, 5.0, 3.0, 0.0 }, &activated);
}

test "Metal kernel wrappers reduce and normalize with softmax" {
    if (!bolt.context.Context.isAvailable()) return error.SkipZigTest;

    var context = try bolt.context.Context.init();
    defer context.deinit();

    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const sum = try context.reduceSumF32(&input);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), sum, 0.0001);

    var output = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    try context.softmaxF32(&input, &output);
    try expectApproxSlices(&.{ 0.0320586, 0.0871443, 0.236883, 0.643914 }, &output);
}

test "Metal kernel wrappers validate shapes before dispatch" {
    if (!bolt.context.Context.isAvailable()) return error.SkipZigTest;

    var context = try bolt.context.Context.init();
    defer context.deinit();

    const left = [_]f32{ 1.0, 2.0 };
    const right = [_]f32{3.0};
    var output = [_]f32{0.0};
    try std.testing.expectError(error.LengthMismatch, context.matMulF32(&left, &right, &output, 1, 1, 3));

    const bias = [_]f32{ 1.0, 2.0 };
    try std.testing.expectError(error.LengthMismatch, context.biasAddF32(&left, &bias, &output, 1, 1));
}
