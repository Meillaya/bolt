const std = @import("std");
const q4_golden = @import("bonsai_q4_golden.zig");

const default_golden_artifact = "../artifacts/nnzap-milestone7-q4-golden.json";
const default_manifest_path = "../artifacts/assets/nnzap-parity/asset-manifest.json";
const bench_artifact_path = "../artifacts/nnzap-milestone7-q4-bench.json";

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const golden_path = if (args.len > 1) args[1] else default_golden_artifact;
    const started = std.Io.Clock.awake.now(init.io).nanoseconds;
    const input = try std.Io.Dir.cwd().readFileAlloc(init.io, golden_path, allocator, .limited(1024 * 1024));
    const stat = try std.Io.Dir.cwd().statFile(init.io, golden_path, .{});
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, input, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    const pass = goldenPassed(parsed.value);
    const elapsed: u64 = @intCast(std.Io.Clock.awake.now(init.io).nanoseconds - started);
    const digest_hex = fileSha256Hex(init.io, allocator, golden_path) catch "sha256-unavailable";
    const bench = if (pass) try q4_golden.runDecodeBenchmark(init.io, allocator, default_manifest_path) else q4_golden.Q4DecodeBenchmark{ .prompt_tokens = 0, .generated_tokens = 0, .matches_reference = false, .elapsed_ns = 0, .q4_projection_dispatches = 0, .q4_logits_dispatches = 0 };
    try writeBench(init, golden_path, pass and bench.matches_reference, elapsed, stat.size, digest_hex, bench);
    if (!pass) return error.Q4GoldenGateRequiredBeforeBench;
    if (!bench.matches_reference) return error.Q4BenchmarkDecodeMismatch;
}

fn goldenPassed(root: std.json.Value) bool {
    if (root != .object) return false;
    const status = root.object.get("status") orelse return false;
    const matches = root.object.get("matches_reference") orelse return false;
    return status == .string and std.mem.eql(u8, status.string, "pass") and matches == .bool and matches.bool;
}

fn fileSha256Hex(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    var hasher = Sha256.init(.{});
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var offset: u64 = 0;
    var buffer: [64 * 1024]u8 = undefined;
    while (true) {
        const read_len = try file.readPositionalAll(io, &buffer, offset);
        if (read_len == 0) break;
        hasher.update(buffer[0..read_len]);
        offset += read_len;
    }
    var digest: [Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    return try std.fmt.allocPrint(allocator, "{s}", .{std.fmt.bytesToHex(digest, .lower)});
}

fn writeBench(init: std.process.Init, golden_path: []const u8, golden_pass: bool, check_ns: u64, artifact_size: u64, digest_hex: []const u8, bench: q4_golden.Q4DecodeBenchmark) !void {
    var file = try std.Io.Dir.cwd().createFile(init.io, bench_artifact_path, .{ .truncate = true });
    defer file.close(init.io);
    var buf: [8192]u8 = undefined;
    var fw = file.writer(init.io, &buf);
    try writeJson(&fw.interface, golden_path, golden_pass, check_ns, artifact_size, digest_hex, bench);
    try fw.interface.flush();
    var stdout_buf: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buf);
    try writeJson(&stdout_writer.interface, golden_path, golden_pass, check_ns, artifact_size, digest_hex, bench);
    try stdout_writer.interface.flush();
}

fn writeJson(w: anytype, golden_path: []const u8, golden_pass: bool, check_ns: u64, artifact_size: u64, digest_hex: []const u8, bench: q4_golden.Q4DecodeBenchmark) !void {
    try w.print(
        "{{\n" ++
            "  \"schema_version\":1,\n" ++
            "  \"gate\":\"nnzap-bonsai-q4-bench\",\n" ++
            "  \"status\":\"{s}\",\n" ++
            "  \"correctness_gate\":{{\"golden_artifact\":\"{s}\",\"required\":true,\"passed\":{s},\"artifact_size_bytes\":{d},\"artifact_sha256\":\"{s}\"}},\n" ++
            "  \"timing\":{{\"golden_artifact_check_ns\":{d},\"real_decode_benchmark_ns\":{d}}},\n" ++
            "  \"workload\":{{\"kind\":\"real_q4_integrated_metal_decode\",\"prompt_tokens\":{d},\"generated_tokens\":{d},\"matches_reference\":{s},\"q4_projection_dispatches\":{d},\"q4_logits_dispatches\":{d}}},\n" ++
            "  \"benchmark_claim\":\"{s}\"\n" ++
            "}}\n",
        .{
            if (golden_pass) "pass" else "blocked",
            golden_path,
            if (golden_pass) "true" else "false",
            artifact_size,
            digest_hex,
            check_ns,
            bench.elapsed_ns,
            bench.prompt_tokens,
            bench.generated_tokens,
            if (bench.matches_reference) "true" else "false",
            bench.q4_projection_dispatches,
            bench.q4_logits_dispatches,
            if (golden_pass) "real integrated Metal Q4 decode benchmark timing may be consumed" else "blocked until q4 golden selected-token parity passes",
        },
    );
}

test "q4 bench refuses to pass without golden" {
    const root = std.json.Value{ .object = std.json.ObjectMap.init(std.testing.allocator) };
    var mutable = root;
    defer mutable.object.deinit();
    try std.testing.expect(!goldenPassed(mutable));
}
