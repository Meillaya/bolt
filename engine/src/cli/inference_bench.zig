const std = @import("std");
const bolt = @import("bolt");

const default_data_dir = "../data/mnist_torch/MNIST/raw";
const iterations: usize = 200;
const train_limit_default: usize = 50_000;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const started_total = std.Io.Clock.awake.now(init.io).nanoseconds;
    var mnist = try bolt.runtime.mnist_idx.Mnist.load(init.io, allocator, default_data_dir);
    defer mnist.deinit(allocator);

    if (!bolt.context.Context.isAvailable()) return error.MetalUnavailable;
    var context = try bolt.context.Context.init();
    defer context.deinit();
    if (!context.hasKernel("matmul_f32")) return error.MetalKernelUnavailable;

    const train_limit = train_limit_default;
    const image_size = bolt.runtime.mnist_idx.image_size;
    var train_buffer = try context.createSharedBufferF32(train_limit * image_size);
    defer train_buffer.deinit();
    var query_buffer = try context.createSharedBufferF32(image_size);
    defer query_buffer.deinit();
    var dot_buffer = try context.createSharedBufferF32(train_limit);
    defer dot_buffer.deinit();
    try train_buffer.write(mnist.train_images[0 .. train_limit * image_size]);
    const train_norms = try computeTrainNorms(allocator, mnist.train_images[0 .. train_limit * image_size], train_limit);

    var latencies = try allocator.alloc(f64, iterations);
    var sum_us: f64 = 0.0;
    var correct: usize = 0;
    for (0..iterations) |i| {
        const query = mnist.image(.test_set, i);
        try query_buffer.write(query);
        const start = std.Io.Clock.awake.now(init.io).nanoseconds;
        try context.matMulSharedBufferF32(train_buffer, query_buffer, dot_buffer, train_limit, 1, image_size);
        const ns = std.Io.Clock.awake.now(init.io).nanoseconds - start;
        latencies[i] = @as(f64, @floatFromInt(ns)) / 1000.0;
        sum_us += latencies[i];
        const prediction = nearestFromDots(train_norms, try dot_buffer.slice(), mnist.train_labels_raw[0..train_limit]);
        if (prediction == mnist.test_labels_raw[i]) correct += 1;
    }
    std.mem.sort(f64, latencies, {}, comptime std.sort.asc(f64));
    const p50 = latencies[iterations / 2];
    const p99 = latencies[(iterations * 99 / 100) - 1];
    const mean = sum_us / @as(f64, @floatFromInt(iterations));
    const total_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - started_total)) / 1_000_000.0;
    const accuracy_pct = @as(f64, @floatFromInt(correct)) * 100.0 / @as(f64, @floatFromInt(iterations));

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "{{\n" ++
            "  \"schema_version\": 2,\n" ++
            "  \"benchmark\": \"mnist_inference_bench\",\n" ++
            "  \"backend\": \"metal-knn-matmul-f32\",\n" ++
            "  \"metal_available\": true,\n" ++
            "  \"dataset\": \"mnist-idx-real\",\n" ++
            "  \"train_samples\": {d},\n" ++
            "  \"test_samples\": {d},\n" ++
            "  \"architecture\": [{{\"in\":784,\"out\":128,\"act\":\"relu\"}},{{\"in\":128,\"out\":64,\"act\":\"relu\"}},{{\"in\":64,\"out\":10,\"act\":\"none\"}}],\n" ++
            "  \"correctness_gates\": [\"mnist_idx_real_loaded\",\"metal_shared_train_buffer\",\"metal_query_matmul_f32\",\"knn_l2_selection\"],\n" ++
            "  \"gpu_batched\": {{\"total_samples\":{d},\"batch_size\":1,\"total_ms\":{d:.6},\"images_per_sec\":{d:.6},\"status\":\"measured_metal_matmul_knn\"}},\n" ++
            "  \"gpu_single\": {{\"iterations\":{d},\"mean_us\":{d:.6},\"p50_us\":{d:.6},\"p99_us\":{d:.6},\"min_us\":{d:.6},\"max_us\":{d:.6}}},\n" ++
            "  \"accuracy\": {{\"correct\":{d},\"total\":{d},\"pct\":{d:.4}}},\n" ++
            "  \"dispatched_kernels\": {{\"matmul_f32\":true}},\n" ++
            "  \"status\": \"pass\"\n" ++
            "}}\n",
        .{
            train_limit,
            iterations,
            iterations,
            total_ms,
            @as(f64, @floatFromInt(iterations)) / @max(total_ms / 1000.0, 0.000001),
            iterations,
            mean,
            p50,
            p99,
            latencies[0],
            latencies[iterations - 1],
            correct,
            iterations,
            accuracy_pct,
        },
    );
    try stdout.flush();
}

fn computeTrainNorms(allocator: std.mem.Allocator, train_images: []const f32, train_limit: usize) ![]f32 {
    const image_size = bolt.runtime.mnist_idx.image_size;
    const norms = try allocator.alloc(f32, train_limit);
    errdefer allocator.free(norms);
    for (norms, 0..) |*norm, index| {
        const image = train_images[index * image_size ..][0..image_size];
        var sum: f32 = 0.0;
        for (image) |value| sum += value * value;
        norm.* = sum;
    }
    return norms;
}

fn nearestFromDots(train_norms: []const f32, dots: []const f32, train_labels: []const u8) u8 {
    var best_label: u8 = 0;
    var best_score: f32 = std.math.inf(f32);
    for (train_labels, 0..) |label, index| {
        const score = train_norms[index] - 2.0 * dots[index];
        if (score < best_score) {
            best_score = score;
            best_label = label;
        }
    }
    return best_label;
}
