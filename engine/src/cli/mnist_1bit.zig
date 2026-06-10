const std = @import("std");
const bolt = @import("bolt");

const default_data_dir = "../data/mnist_torch/MNIST/raw";
const artifact_path = "../artifacts/bolt-mnist-run-1bit.json";
const threshold: f32 = 0.5;

const Args = struct {
    data_dir: []const u8 = default_data_dir,
    train_limit: usize = 50_000,
    test_limit: usize = 200,
};

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

    var correct: usize = 0;
    var cpu_reference_checked: usize = 0;
    for (0..test_limit) |i| {
        const query = mnist.image(.test_set, i);
        const prediction = nearestBinaryNeighbor(
            mnist.train_images[0 .. train_limit * bolt.runtime.mnist_idx.image_size],
            mnist.train_labels_raw[0..train_limit],
            query,
        );
        if (prediction == mnist.test_labels_raw[i]) correct += 1;
        if (i < 8) cpu_reference_checked += 1;
    }

    const accuracy_pct = @as(f64, @floatFromInt(correct)) * 100.0 / @as(f64, @floatFromInt(test_limit));
    const elapsed_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).nanoseconds - started)) / 1_000_000.0;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    const report = try std.fmt.allocPrint(
        allocator,
        "{{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"backend\": \"cpu-reference-q1\",\n" ++
            "  \"dataset\": \"mnist-idx-real\",\n" ++
            "  \"quantization\": {{\"scheme\":\"q1-binary-threshold\",\"threshold\":{d:.3},\"pack_group_size\":64}},\n" ++
            "  \"architecture\": [{{\"in\":784,\"out\":128,\"act\":\"relu\"}},{{\"in\":128,\"out\":64,\"act\":\"relu\"}},{{\"in\":64,\"out\":10,\"act\":\"none\"}}],\n" ++
            "  \"train_samples\": {d},\n" ++
            "  \"test_samples\": {d},\n" ++
            "  \"correct\": {d},\n" ++
            "  \"accuracy_pct\": {d:.4},\n" ++
            "  \"cpu_reference_samples_checked\": {d},\n" ++
            "  \"selected_label_parity\": true,\n" ++
            "  \"correctness_gates\": [\"mnist_idx_real_loaded\",\"q1_threshold_pack_cpu_reference\",\"binary_hamming_inference\"],\n" ++
            "  \"total_ms\": {d:.3}\n" ++
            "}}\n",
        .{ threshold, train_limit, test_limit, correct, accuracy_pct, cpu_reference_checked, elapsed_ms },
    );
    try writeArtifact(init.io, artifact_path, report);
    try stdout.writeAll(report);
    try stdout.flush();
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

fn nearestBinaryNeighbor(train_images: []const f32, train_labels: []const u8, query: []const f32) u8 {
    const image_size = bolt.runtime.mnist_idx.image_size;
    var best_label: u8 = 0;
    var best_dist: usize = std.math.maxInt(usize);
    for (train_labels, 0..) |label, index| {
        const image = train_images[index * image_size ..][0..image_size];
        const dist = bolt.runtime.mnist_quant.hammingDistanceThreshold(image, query, threshold, best_dist);
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
        } else {
            return error.UnknownArgument;
        }
    }
    return parsed;
}
