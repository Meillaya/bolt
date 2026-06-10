const std = @import("std");
const bolt = @import("bolt");

const artifact_path = "../artifacts/bolt-metal-mlp-runtime.json";

fn writeReport(init: std.process.Init, path: []const u8, report: []const u8) !void {
    try ensureParentDir(init.io, path);
    var file = try std.Io.Dir.cwd().createFile(init.io, path, .{ .truncate = true });
    defer file.close(init.io);
    var file_buffer: [4096]u8 = undefined;
    var file_writer = file.writer(init.io, &file_buffer);
    try file_writer.interface.writeAll(report);
    try file_writer.interface.flush();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.writeAll(report);
    try stdout.flush();
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    try std.Io.Dir.cwd().createDirPath(io, parent);
}

fn formatUnavailable(
    allocator: std.mem.Allocator,
    cpu: *const bolt.runtime.metal_mlp.InferenceResult,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"gate\": \"bolt-metal-mlp-runtime\",\n" ++
            "  \"status\": \"skipped\",\n" ++
            "  \"reason\": \"Metal backend unavailable\",\n" ++
            "  \"artifact_path\": \"{s}\",\n" ++
            "  \"cpu_predicted_class\": {d},\n" ++
            "  \"cpu_logits\": [{d:.6}, {d:.6}]\n" ++
            "}}\n",
        .{ artifact_path, cpu.predicted_class, cpu.logits[0], cpu.logits[1] },
    );
}

fn formatParityReport(
    allocator: std.mem.Allocator,
    cpu: *const bolt.runtime.metal_mlp.InferenceResult,
    gpu: *const bolt.runtime.metal_mlp.InferenceResult,
    max_logit_diff: f32,
    max_probability_diff: f32,
    status: []const u8,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"gate\": \"bolt-metal-mlp-runtime\",\n" ++
            "  \"status\": \"{s}\",\n" ++
            "  \"artifact_path\": \"{s}\",\n" ++
            "  \"model\": \"demo-2x3x2-mlp\",\n" ++
            "  \"cpu_backend\": \"{s}\",\n" ++
            "  \"metal_backend\": \"{s}\",\n" ++
            "  \"cpu_predicted_class\": {d},\n" ++
            "  \"metal_predicted_class\": {d},\n" ++
            "  \"max_logit_diff\": {d:.8},\n" ++
            "  \"max_probability_diff\": {d:.8},\n" ++
            "  \"cpu_logits\": [{d:.6}, {d:.6}],\n" ++
            "  \"metal_logits\": [{d:.6}, {d:.6}],\n" ++
            "  \"probabilities\": [{d:.6}, {d:.6}]\n" ++
            "}}\n",
        .{
            status,
            artifact_path,
            cpu.backend,
            gpu.backend,
            cpu.predicted_class,
            gpu.predicted_class,
            max_logit_diff,
            max_probability_diff,
            cpu.logits[0],
            cpu.logits[1],
            gpu.logits[0],
            gpu.logits[1],
            gpu.probabilities[0],
            gpu.probabilities[1],
        },
    );
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var cpu = try bolt.runtime.metal_mlp.runDemoCpu(allocator);
    defer cpu.deinit();

    if (!bolt.context.Context.isAvailable()) {
        const report = try formatUnavailable(allocator, &cpu);
        try writeReport(init, artifact_path, report);
        return;
    }

    var gpu = try bolt.runtime.metal_mlp.runDemoMetal(allocator);
    defer gpu.deinit();
    const max_logit_diff = try bolt.runtime.metal_mlp.maxAbsDiff(cpu.logits, gpu.logits);
    const max_probability_diff = try bolt.runtime.metal_mlp.maxAbsDiff(cpu.probabilities, gpu.probabilities);
    const status = if (cpu.predicted_class == gpu.predicted_class and max_logit_diff < 0.00001 and max_probability_diff < 0.00001) "pass" else "fail";

    const report = try formatParityReport(allocator, &cpu, &gpu, max_logit_diff, max_probability_diff, status);
    try writeReport(init, artifact_path, report);
    if (!std.mem.eql(u8, status, "pass")) return error.MetalMlpParityFailed;
}
