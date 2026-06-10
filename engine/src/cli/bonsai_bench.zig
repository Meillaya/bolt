const std = @import("std");
const bonsai_golden = @import("bonsai_golden.zig");

const golden_artifact_path = "../artifacts/bolt-bonsai-readiness.json";
const default_manifest_path = "../artifacts/assets/bolt-parity/asset-manifest.json";
const bench_artifact_path = "../artifacts/bolt-bonsai-bench.json";

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = if (args.len > 1) args[1] else default_manifest_path;
    const started = std.Io.Clock.awake.now(init.io).nanoseconds;
    const golden_input = try readRequiredFile(init.io, allocator, golden_artifact_path, "prerequisite golden artifact");
    const stat = try std.Io.Dir.cwd().statFile(init.io, golden_artifact_path, .{});
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, golden_input, .{ .allocate = .alloc_always });
    defer parsed.deinit();

    const golden_pass = artifactHasPassStatus(parsed.value) and artifactHasReferenceMatch(parsed.value);
    const elapsed: u64 = @intCast(std.Io.Clock.awake.now(init.io).nanoseconds - started);
    const digest_hex = try fileSha256Hex(init.io, allocator, golden_artifact_path);
    const bench = if (golden_pass) try bonsai_golden.runDecodeBenchmark(init.io, allocator, manifest_path) else bonsai_golden.BonsaiDecodeBenchmark{ .prompt_tokens = 0, .generated_tokens = 0, .matches_reference = false, .elapsed_ns = 0, .layers_per_token = 0 };
    try writeArtifact(init, golden_pass and bench.matches_reference, elapsed, stat.size, digest_hex, bench);
    if (!golden_pass) return error.BonsaiGoldenGateRequiredBeforeBench;
    if (!bench.matches_reference) return error.BonsaiBenchmarkDecodeMismatch;
}

fn readRequiredFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8, label: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(4 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => {
            std.debug.print("missing {s}: {s}\nRun the corresponding golden gate first, or pass the required artifact path if the command accepts one.\n", .{ label, path });
            return error.MissingRequiredFile;
        },
        else => return err,
    };
}

fn ensureParentDir(io: std.Io, path: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        try std.Io.Dir.cwd().createDirPath(io, parent);
    }
}

fn artifactHasPassStatus(root: std.json.Value) bool {
    if (root != .object) return false;
    const status = root.object.get("status") orelse return false;
    return status == .string and std.mem.eql(u8, status.string, "pass");
}

fn artifactHasReferenceMatch(root: std.json.Value) bool {
    if (root != .object) return false;
    if (root.object.get("real_generation_probe")) |probe| {
        if (probe == .object) {
            const value = probe.object.get("matches_reference") orelse return false;
            return value == .bool and value.bool;
        }
    }
    const value = root.object.get("matches_reference") orelse return false;
    return value == .bool and value.bool;
}

fn fileSha256Hex(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    var hasher = Sha256.init(.{});
    var file = std.Io.Dir.cwd().openFile(io, path, .{}) catch |err| {
        std.debug.print("could not hash prerequisite golden artifact: {s} ({s})\nRerun the corresponding golden gate before rerunning run-bonsai-bench.\n", .{ path, @errorName(err) });
        return error.MissingRequiredFile;
    };
    defer file.close(io);
    var offset: u64 = 0;
    var buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const read_len = file.readPositionalAll(io, &buffer, offset) catch |err| {
            std.debug.print("could not hash prerequisite golden artifact: {s} ({s})\nRerun the corresponding golden gate before rerunning run-bonsai-bench.\n", .{ path, @errorName(err) });
            return error.MissingRequiredFile;
        };
        if (read_len == 0) break;
        hasher.update(buffer[0..read_len]);
        offset += read_len;
    }
    var digest: [Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return try std.fmt.allocPrint(allocator, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
}

fn writeArtifact(init: std.process.Init, pass: bool, elapsed_ns: u64, artifact_size: u64, digest_hex: []const u8, bench: bonsai_golden.BonsaiDecodeBenchmark) !void {
    try ensureParentDir(init.io, bench_artifact_path);
    var file = try std.Io.Dir.cwd().createFile(init.io, bench_artifact_path, .{ .truncate = true });
    defer file.close(init.io);
    var buf: [4096]u8 = undefined;
    var fw = file.writer(init.io, &buf);
    try writeJson(&fw.interface, pass, elapsed_ns, artifact_size, digest_hex, bench);
    try fw.interface.flush();

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &stdout_buf);
    try writeJson(&stdout.interface, pass, elapsed_ns, artifact_size, digest_hex, bench);
    try stdout.interface.flush();
}

fn writeJson(w: anytype, pass: bool, elapsed_ns: u64, artifact_size: u64, digest_hex: []const u8, bench: bonsai_golden.BonsaiDecodeBenchmark) !void {
    try w.print(
        "{{\n" ++
            "  \"schema_version\":1,\n" ++
            "  \"gate\":\"bolt-bonsai-bench\",\n" ++
            "  \"status\":\"{s}\",\n" ++
            "  \"correctness_gate\":{{\"golden_artifact\":\"{s}\",\"required\":true,\"passed\":{s},\"artifact_size_bytes\":{d},\"artifact_sha256\":\"{s}\"}},\n" ++
            "  \"timing\":{{\"golden_artifact_check_ns\":{d},\"real_decode_benchmark_ns\":{d}}},\n" ++
            "  \"workload\":{{\"kind\":\"real_bonsai_f16_decode\",\"prompt_tokens\":{d},\"generated_tokens\":{d},\"matches_reference\":{s},\"layers_per_token\":{d}}},\n" ++
            "  \"benchmark_claim\":\"{s}\"\n" ++
            "}}\n",
        .{
            if (pass) "pass" else "blocked",
            golden_artifact_path,
            if (pass) "true" else "false",
            artifact_size,
            digest_hex,
            elapsed_ns,
            bench.elapsed_ns,
            bench.prompt_tokens,
            bench.generated_tokens,
            if (bench.matches_reference) "true" else "false",
            bench.layers_per_token,
            if (pass) "real Bonsai decode benchmark timing may be consumed" else "blocked until Bonsai golden gate passes",
        },
    );
}

test "bonsai bench status parser requires pass and reference match" {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"status\":\"pass\",\"real_generation_probe\":{\"matches_reference\":true}}", .{});
    defer parsed.deinit();
    try std.testing.expect(artifactHasPassStatus(parsed.value));
    try std.testing.expect(artifactHasReferenceMatch(parsed.value));
}
