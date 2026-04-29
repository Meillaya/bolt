const std = @import("std");
const bolt = @import("bolt");

const Iterations = 24;

fn fillSequence(values: []f32, modulus: usize, offset: f32, scale: f32) void {
    for (values, 0..) |*value, index| {
        const bucket = @as(f32, @floatFromInt(index % modulus));
        value.* = offset + bucket * scale;
    }
}

fn benchTime(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).nanoseconds;
}

fn elapsedNs(init: std.process.Init, start: i96) u64 {
    return @as(u64, @intCast(benchTime(init.io) - start));
}

fn writeUnavailable(init: std.process.Init) !void {
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.writeAll(
        "{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"backend\": \"metal\",\n" ++
            "  \"metal_available\": false,\n" ++
            "  \"status\": \"skipped\",\n" ++
            "  \"reason\": \"Metal backend unavailable\"\n" ++
            "}\n",
    );
    try stdout.flush();
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    if (!bolt.context.Context.isAvailable()) {
        try writeUnavailable(init);
        return;
    }

    var context = try bolt.context.Context.init();
    defer context.deinit();

    const vector_count: usize = 1024;
    const rows: usize = 16;
    const cols: usize = 16;
    const inner: usize = 16;
    const matrix_left_count = rows * inner;
    const matrix_right_count = inner * cols;
    const matrix_output_count = rows * cols;

    const left = try allocator.alloc(f32, vector_count);
    const right = try allocator.alloc(f32, vector_count);
    const vector_output = try allocator.alloc(f32, vector_count);
    const matrix_left = try allocator.alloc(f32, matrix_left_count);
    const matrix_right = try allocator.alloc(f32, matrix_right_count);
    const matrix_output = try allocator.alloc(f32, matrix_output_count);
    const bias = try allocator.alloc(f32, cols);
    const softmax_output = try allocator.alloc(f32, cols);

    fillSequence(left, 17, -4.0, 0.5);
    fillSequence(right, 19, 1.0, 0.25);
    fillSequence(matrix_left, 11, -2.0, 0.2);
    fillSequence(matrix_right, 13, 0.5, 0.125);
    fillSequence(bias, 7, -0.75, 0.25);

    var start = benchTime(init.io);
    for (0..Iterations) |_| {
        try context.addF32(left, right, vector_output);
    }
    const add_ns = elapsedNs(init, start);

    start = benchTime(init.io);
    for (0..Iterations) |_| {
        try context.reluF32(left, vector_output);
    }
    const relu_ns = elapsedNs(init, start);

    start = benchTime(init.io);
    for (0..Iterations) |_| {
        try context.matMulF32(matrix_left, matrix_right, matrix_output, rows, cols, inner);
    }
    const matmul_ns = elapsedNs(init, start);

    start = benchTime(init.io);
    for (0..Iterations) |_| {
        try context.biasAddF32(matrix_output, bias, matrix_output, rows, cols);
    }
    const bias_add_ns = elapsedNs(init, start);

    start = benchTime(init.io);
    var reduce_sum: f32 = 0.0;
    for (0..Iterations) |_| {
        reduce_sum = try context.reduceSumF32(left);
    }
    const reduce_sum_ns = elapsedNs(init, start);

    start = benchTime(init.io);
    for (0..Iterations) |_| {
        try context.softmaxF32(bias, softmax_output);
    }
    const softmax_ns = elapsedNs(init, start);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "{{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"backend\": \"metal\",\n" ++
            "  \"metal_available\": true,\n" ++
            "  \"status\": \"ok\",\n" ++
            "  \"iterations\": {d},\n" ++
            "  \"primitive_results\": [\n" ++
            "    {{ \"name\": \"add_f32\", \"elements\": {d}, \"elapsed_ns\": {d}, \"checksum\": {d:.6} }},\n" ++
            "    {{ \"name\": \"relu_f32\", \"elements\": {d}, \"elapsed_ns\": {d}, \"checksum\": {d:.6} }},\n" ++
            "    {{ \"name\": \"matmul_f32\", \"rows\": {d}, \"cols\": {d}, \"inner\": {d}, \"elapsed_ns\": {d}, \"checksum\": {d:.6} }},\n" ++
            "    {{ \"name\": \"bias_add_f32\", \"rows\": {d}, \"cols\": {d}, \"elapsed_ns\": {d}, \"checksum\": {d:.6} }},\n" ++
            "    {{ \"name\": \"reduce_sum_f32\", \"elements\": {d}, \"elapsed_ns\": {d}, \"checksum\": {d:.6} }},\n" ++
            "    {{ \"name\": \"softmax_f32\", \"elements\": {d}, \"elapsed_ns\": {d}, \"checksum\": {d:.6} }}\n" ++
            "  ]\n" ++
            "}}\n",
        .{
            Iterations,
            vector_count,
            add_ns,
            vector_output[vector_count - 1],
            vector_count,
            relu_ns,
            vector_output[vector_count - 1],
            rows,
            cols,
            inner,
            matmul_ns,
            matrix_output[matrix_output_count - 1],
            rows,
            cols,
            bias_add_ns,
            matrix_output[matrix_output_count - 1],
            vector_count,
            reduce_sum_ns,
            reduce_sum,
            cols,
            softmax_ns,
            softmax_output[cols - 1],
        },
    );
    try stdout.flush();
}
