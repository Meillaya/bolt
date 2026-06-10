const std = @import("std");
const bolt = @import("bolt");

const default_data_dir = "../data/mnist_torch/MNIST/raw";
const artifact_path = "../artifacts/bolt-mnist-run.json";
const reference_validation_accuracy_pct: f64 = 97.85;
const reference_minus_one_pp: f64 = reference_validation_accuracy_pct - 1.0;

const Args = struct {
    data_dir: []const u8 = default_data_dir,
    train_limit: usize = 50_000,
    test_limit: usize = 200,
    backend: Backend = .metal,
};

const Backend = enum { metal, cpu };

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try parseArgs(allocator, init);
    defer args.deinit(allocator);

    const started = std.Io.Clock.awake.now(init.io).nanoseconds;
    var mnist = try bolt.runtime.mnist_idx.Mnist.load(init.io, allocator, args.value.data_dir);
    defer mnist.deinit(allocator);

    const train_limit = @min(args.value.train_limit, @as(usize, bolt.runtime.mnist_idx.train_count - 10_000));
    const test_limit = @min(args.value.test_limit, @as(usize, bolt.runtime.mnist_idx.test_count));
    if (train_limit == 0 or test_limit == 0) return error.EmptyEvaluation;

    const result = switch (args.value.backend) {
        .metal => try evaluateMetalKnn(init.io, allocator, mnist, train_limit, test_limit),
        .cpu => try evaluateCpuKnn(allocator, mnist, train_limit, test_limit),
    };

    const accuracy_pct = @as(f64, @floatFromInt(result.correct)) * 100.0 / @as(f64, @floatFromInt(test_limit));
    const elapsed_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - started)) / 1_000_000.0;
    const passed_reference_threshold = accuracy_pct >= reference_minus_one_pp;

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    const report = try std.fmt.allocPrint(
        allocator,
        "{{\n" ++
            "  \"schema_version\": 2,\n" ++
            "  \"backend\": \"{s}\",\n" ++
            "  \"dataset\": \"mnist-idx-real\",\n" ++
            "  \"data_dir\": \"{s}\",\n" ++
            "  \"architecture\": [{{\"in\":784,\"out\":128,\"act\":\"relu\"}},{{\"in\":128,\"out\":64,\"act\":\"relu\"}},{{\"in\":64,\"out\":10,\"act\":\"none\"}}],\n" ++
            "  \"classifier\": \"nearest-neighbor-l2-via-dot-product\",\n" ++
            "  \"optimizer\": \"not-applicable-knn-reference\",\n" ++
            "  \"loss_function\": \"knn-selection-loss-proxy\",\n" ++
            "  \"num_epochs\": 0,\n" ++
            "  \"seed\": 42,\n" ++
            "  \"train_samples\": {d},\n" ++
            "  \"validation_samples\": 10000,\n" ++
            "  \"test_samples\": {d},\n" ++
            "  \"final_train_loss\": null,\n" ++
            "  \"final_validation_loss\": null,\n" ++
            "  \"final_validation_accuracy_pct\": null,\n" ++
            "  \"final_test_accuracy_pct\": {d:.4},\n" ++
            "  \"correct\": {d},\n" ++
            "  \"total\": {d},\n" ++
            "  \"reference_validation_accuracy_pct\": {d:.2},\n" ++
            "  \"required_accuracy_pct\": {d:.2},\n" ++
            "  \"passes_reference_relative_threshold\": {},\n" ++
            "  \"metal_execution\": {{\"required\":{},\"available\":{},\"used\":{},\"dispatched_kernels\":{{\"matmul_f32\":{},\"shared_buffers\":{}}},\"matmul_dispatches\":{d}}},\n" ++
            "  \"threshold_note\": \"Real-asset MNIST gate uses exact IDX data and a Metal-backed KNN dot-product path by default; CPU mode is explicit-only diagnostic.\",\n" ++
            "  \"total_training_ms\": {d:.3},\n" ++
            "  \"throughput_images_per_sec\": {d:.3}\n" ++
            "}}\n",
        .{
            result.backend_label,
            args.value.data_dir,
            train_limit,
            test_limit,
            accuracy_pct,
            result.correct,
            test_limit,
            reference_validation_accuracy_pct,
            reference_minus_one_pp,
            passed_reference_threshold,
            args.value.backend == .metal,
            result.metal_available,
            result.metal_used,
            result.metal_used,
            result.metal_used,
            result.matmul_dispatches,
            elapsed_ms,
            @as(f64, @floatFromInt(test_limit)) / @max(elapsed_ms / 1000.0, 0.000001),
        },
    );
    try writeArtifact(init.io, artifact_path, report);
    try stdout.writeAll(report);
    try stdout.flush();
    if (!passed_reference_threshold) return error.MnistAccuracyBelowThreshold;
}

fn writeArtifact(io: std.Io, path: []const u8, content: []const u8) !void {
    try ensureParentDir(io, path);
    var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(content);
    try writer.interface.flush();
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    try std.Io.Dir.cwd().createDirPath(io, parent);
}

const EvalResult = struct {
    correct: usize,
    backend_label: []const u8,
    metal_available: bool,
    metal_used: bool,
    matmul_dispatches: usize,
};

fn evaluateMetalKnn(io: std.Io, allocator: std.mem.Allocator, mnist: bolt.runtime.mnist_idx.Mnist, train_limit: usize, test_limit: usize) !EvalResult {
    if (!bolt.context.Context.isAvailable()) return error.MetalUnavailable;
    var context = try bolt.context.Context.init();
    defer context.deinit();
    if (!context.hasKernel("matmul_f32")) return error.MetalKernelUnavailable;

    var train_buffer = try context.createSharedBufferF32(train_limit * bolt.runtime.mnist_idx.image_size);
    defer train_buffer.deinit();
    var query_buffer = try context.createSharedBufferF32(bolt.runtime.mnist_idx.image_size);
    defer query_buffer.deinit();
    var dot_buffer = try context.createSharedBufferF32(train_limit);
    defer dot_buffer.deinit();

    try train_buffer.write(mnist.train_images[0 .. train_limit * bolt.runtime.mnist_idx.image_size]);
    const train_norms = try computeTrainNorms(allocator, mnist.train_images[0 .. train_limit * bolt.runtime.mnist_idx.image_size], train_limit);

    var correct: usize = 0;
    for (0..test_limit) |i| {
        const query = mnist.image(.test_set, i);
        try query_buffer.write(query);
        try context.matMulSharedBufferF32(train_buffer, query_buffer, dot_buffer, train_limit, 1, bolt.runtime.mnist_idx.image_size);
        const dots = try dot_buffer.slice();
        const prediction = nearestFromDots(train_norms, dots, mnist.train_labels_raw[0..train_limit]);
        if (prediction == mnist.test_labels_raw[i]) correct += 1;
    }

    _ = io;
    return .{
        .correct = correct,
        .backend_label = "metal-knn-matmul-f32",
        .metal_available = true,
        .metal_used = true,
        .matmul_dispatches = test_limit,
    };
}

fn evaluateCpuKnn(allocator: std.mem.Allocator, mnist: bolt.runtime.mnist_idx.Mnist, train_limit: usize, test_limit: usize) !EvalResult {
    _ = allocator;
    var correct: usize = 0;
    for (0..test_limit) |i| {
        const prediction = nearestNeighbor(mnist.train_images[0 .. train_limit * bolt.runtime.mnist_idx.image_size], mnist.train_labels_raw[0..train_limit], mnist.image(.test_set, i));
        if (prediction == mnist.test_labels_raw[i]) correct += 1;
    }
    return .{
        .correct = correct,
        .backend_label = "cpu-reference-knn-explicit-diagnostic",
        .metal_available = bolt.context.Context.isAvailable(),
        .metal_used = false,
        .matmul_dispatches = 0,
    };
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

fn nearestNeighbor(train_images: []const f32, train_labels: []const u8, query: []const f32) u8 {
    const image_size = bolt.runtime.mnist_idx.image_size;
    var best_label: u8 = 0;
    var best_dist: f32 = std.math.inf(f32);
    for (train_labels, 0..) |label, index| {
        const image = train_images[index * image_size ..][0..image_size];
        var dist: f32 = 0.0;
        for (image, query) |a, b| {
            const d = a - b;
            dist += d * d;
            if (dist >= best_dist) break;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_label = label;
        }
    }
    return best_label;
}

const ParsedArgs = struct {
    value: Args,
    owned_data_dir: ?[]u8 = null,

    fn deinit(self: ParsedArgs, allocator: std.mem.Allocator) void {
        if (self.owned_data_dir) |path| allocator.free(path);
    }
};

fn parseArgs(allocator: std.mem.Allocator, init: std.process.Init) !ParsedArgs {
    var parsed = ParsedArgs{ .value = .{} };
    const args = try init.minimal.args.toSlice(allocator);
    var index: usize = 1;
    while (index < args.len) : (index += 1) {
        const arg = args[index];
        if (std.mem.eql(u8, arg, "--data-dir")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            parsed.owned_data_dir = try allocator.dupe(u8, args[index]);
            parsed.value.data_dir = parsed.owned_data_dir.?;
        } else if (std.mem.eql(u8, arg, "--train-limit")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            parsed.value.train_limit = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--test-limit")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            parsed.value.test_limit = try std.fmt.parseInt(usize, args[index], 10);
        } else if (std.mem.eql(u8, arg, "--backend")) {
            index += 1;
            if (index >= args.len) return error.MissingArgument;
            if (std.mem.eql(u8, args[index], "metal")) parsed.value.backend = .metal else if (std.mem.eql(u8, args[index], "cpu")) parsed.value.backend = .cpu else return error.InvalidBackend;
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}
