const std = @import("std");

pub const MatrixView = struct {
    packed_words: []const u32,
    scales: []const f32,
    biases: []const f32,
    rows: usize,
    cols: usize,
    group_size: usize,

    pub fn groupsPerRow(self: MatrixView) !usize {
        try validateShape(self.rows, self.cols, self.group_size);
        return self.cols / self.group_size;
    }

    pub fn wordsPerRow(self: MatrixView) !usize {
        try validateShape(self.rows, self.cols, self.group_size);
        return self.cols / 8;
    }
};

pub fn validateShape(rows: usize, cols: usize, group_size: usize) !void {
    if (rows == 0 or cols == 0 or group_size == 0) return error.InvalidQ4Shape;
    if (cols % group_size != 0) return error.InvalidQ4Shape;
    if (group_size % 8 != 0) return error.InvalidQ4Shape;
}

pub fn inferGroupSize(cols: usize, groups_per_row: usize) !usize {
    if (cols == 0 or groups_per_row == 0) return error.InvalidQ4Shape;
    if (cols % groups_per_row != 0) return error.InvalidQ4Shape;
    const group_size = cols / groups_per_row;
    try validateShape(1, cols, group_size);
    return group_size;
}

pub fn nibbleAt(word: u32, nibble_index: usize) u4 {
    std.debug.assert(nibble_index < 8);
    const shift: u5 = @intCast(nibble_index * 4);
    return @truncate((word >> shift) & 0xF);
}

pub fn dequantValue(scale: f32, bias: f32, nibble: u4) f32 {
    return scale * @as(f32, @floatFromInt(nibble)) + bias;
}

pub fn bf16Round(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    return @bitCast(bits & 0xFFFF0000);
}

pub fn dequantizeRows(view: MatrixView, out: []f32, bf16_round_output: bool) !void {
    try validateShape(view.rows, view.cols, view.group_size);
    const groups = view.cols / view.group_size;
    const words_per_row = view.cols / 8;
    const words_per_group = view.group_size / 8;
    if (view.packed_words.len != view.rows * words_per_row) return error.InvalidQ4Shape;
    if (view.scales.len != view.rows * groups or view.biases.len != view.rows * groups) return error.InvalidQ4Shape;
    if (out.len != view.rows * view.cols) return error.InvalidQ4Shape;

    for (0..view.rows) |row| {
        for (0..groups) |group| {
            const scale = view.scales[row * groups + group];
            const bias = view.biases[row * groups + group];
            for (0..words_per_group) |word_i| {
                const word = view.packed_words[row * words_per_row + group * words_per_group + word_i];
                for (0..8) |nib| {
                    const col = group * view.group_size + word_i * 8 + nib;
                    const value = dequantValue(scale, bias, nibbleAt(word, nib));
                    out[row * view.cols + col] = if (bf16_round_output) bf16Round(value) else value;
                }
            }
        }
    }
}

pub fn matVecRows(view: MatrixView, vector: []const f32, out: []f32, bf16_round_weights: bool) !void {
    try validateShape(view.rows, view.cols, view.group_size);
    if (vector.len != view.cols or out.len != view.rows) return error.InvalidQ4Shape;
    const groups = view.cols / view.group_size;
    const words_per_row = view.cols / 8;
    const words_per_group = view.group_size / 8;
    if (view.packed_words.len != view.rows * words_per_row) return error.InvalidQ4Shape;
    if (view.scales.len != view.rows * groups or view.biases.len != view.rows * groups) return error.InvalidQ4Shape;

    for (0..view.rows) |row| {
        var sum: f32 = 0;
        for (0..groups) |group| {
            const scale = view.scales[row * groups + group];
            const bias = view.biases[row * groups + group];
            for (0..words_per_group) |word_i| {
                const word = view.packed_words[row * words_per_row + group * words_per_group + word_i];
                for (0..8) |nib| {
                    const col = group * view.group_size + word_i * 8 + nib;
                    var weight = dequantValue(scale, bias, nibbleAt(word, nib));
                    if (bf16_round_weights) weight = bf16Round(weight);
                    sum += weight * vector[col];
                }
            }
        }
        out[row] = sum;
    }
}

fn pack8(n0: u4, n1: u4, n2: u4, n3: u4, n4: u4, n5: u4, n6: u4, n7: u4) u32 {
    const vals = [_]u4{ n0, n1, n2, n3, n4, n5, n6, n7 };
    var word: u32 = 0;
    for (vals, 0..) |v, i| word |= @as(u32, v) << @intCast(i * 4);
    return word;
}

test "Q4 MLX dequantizes affine groups deterministically" {
    const words = [_]u32{
        pack8(0, 1, 2, 3, 4, 5, 6, 7),
        pack8(8, 9, 10, 11, 12, 13, 14, 15),
        pack8(15, 14, 13, 12, 11, 10, 9, 8),
        pack8(7, 6, 5, 4, 3, 2, 1, 0),
    };
    const scales = [_]f32{ 1.0, 0.5, -1.0, 2.0 };
    const biases = [_]f32{ 0.0, 10.0, 1.0, -4.0 };
    var out: [32]f32 = undefined;
    try dequantizeRows(.{ .packed_words = &words, .scales = &scales, .biases = &biases, .rows = 2, .cols = 16, .group_size = 8 }, &out, false);
    try std.testing.expectEqual(@as(f32, 0.0), out[0]);
    try std.testing.expectEqual(@as(f32, 7.0), out[7]);
    try std.testing.expectEqual(@as(f32, 14.0), out[8]);
    try std.testing.expectEqual(@as(f32, 17.5), out[15]);
    try std.testing.expectEqual(@as(f32, -13.0), out[17]);
    try std.testing.expectEqual(@as(f32, -4.0), out[31]);
}

test "Q4 matvec matches explicit dequantized matrix" {
    const words = [_]u32{
        pack8(0, 1, 2, 3, 4, 5, 6, 7),
        pack8(8, 9, 10, 11, 12, 13, 14, 15),
    };
    const scales = [_]f32{ 0.25, 0.5 };
    const biases = [_]f32{ -1.0, 2.0 };
    const vector = [_]f32{ 1, -1, 2, -2, 3, -3, 4, -4, 0.5, 1.5, -0.5, -1.5, 2.5, -2.5, 3.5, -3.5 };
    const view = MatrixView{ .packed_words = &words, .scales = &scales, .biases = &biases, .rows = 1, .cols = 16, .group_size = 8 };
    var matrix: [16]f32 = undefined;
    try dequantizeRows(view, &matrix, false);
    var expected: f32 = 0;
    for (matrix, vector) |w, x| expected += w * x;
    var actual: [1]f32 = undefined;
    try matVecRows(view, &vector, &actual, false);
    try std.testing.expectApproxEqAbs(expected, actual[0], 0.00001);
}

test "Q4 group size inference validates safetensors scale columns" {
    try std.testing.expectEqual(@as(usize, 128), try inferGroupSize(2048, 16));
    try std.testing.expectEqual(@as(usize, 64), try inferGroupSize(2048, 32));
    try std.testing.expectError(error.InvalidQ4Shape, inferGroupSize(2048, 0));
    try std.testing.expectError(error.InvalidQ4Shape, inferGroupSize(2048, 127));
}
