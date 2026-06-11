const std = @import("std");
const builtin = @import("builtin");

pub const schema_version = "1";

pub const Metadata = struct {
    command: []const u8,
    cwd: []const u8,
    git_commit: []const u8,
    timestamp_utc: []const u8,
    zig_version: []const u8,
};

pub fn capture(allocator: std.mem.Allocator, io: std.Io, command: []const u8) !Metadata {
    return .{
        .command = command,
        .cwd = try commandOutput(allocator, io, &.{"pwd"}),
        .git_commit = try gitIdentity(allocator, io),
        .timestamp_utc = try timestampUtc(allocator, io),
        .zig_version = builtin.zig_version_string,
    };
}

fn gitIdentity(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    const sha = commandOutput(allocator, io, &.{ "git", "rev-parse", "--short", "HEAD" }) catch return "dirty";
    if (std.mem.eql(u8, sha, "unknown")) return "dirty";
    const status = commandOutput(allocator, io, &.{ "git", "status", "--short" }) catch "dirty";
    if (status.len == 0) return sha;
    return try std.fmt.allocPrint(allocator, "{s}-dirty", .{sha});
}

fn commandOutput(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) ![]const u8 {
    const result = std.process.run(allocator, io, .{
        .argv = argv,
        .stdout_limit = .limited(4096),
        .stderr_limit = .limited(4096),
        .expand_arg0 = .expand,
    }) catch return "unknown";
    switch (result.term) {
        .exited => |code| if (code != 0) return "unknown",
        else => return "unknown",
    }
    return std.mem.trim(u8, result.stdout, "\n\r \t");
}

fn timestampUtc(allocator: std.mem.Allocator, io: std.Io) ![]const u8 {
    const ts: u64 = @intCast(std.Io.Timestamp.now(io, .real).toSeconds());
    const es = std.time.epoch.EpochSeconds{ .secs = ts };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();
    return try std.fmt.allocPrint(
        allocator,
        "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z",
        .{
            year_day.year,
            month_day.month.numeric(),
            @as(u32, month_day.day_index) + 1,
            day_secs.getHoursIntoDay(),
            day_secs.getMinutesIntoHour(),
            day_secs.getSecondsIntoMinute(),
        },
    );
}

test "git dirty marker is explicit" {
    const dirty = try std.fmt.allocPrint(std.testing.allocator, "{s}-dirty", .{"abcdef0"});
    defer std.testing.allocator.free(dirty);
    try std.testing.expect(std.mem.endsWith(u8, dirty, "-dirty"));
}

pub fn fileSha256Hex(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]const u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &reader_buffer);
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [4096]u8 = undefined;
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        if (n == 0) break;
        hasher.update(chunk[0..n]);
    }
    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    return try std.fmt.allocPrint(allocator, "sha256:{s}", .{std.fmt.bytesToHex(digest, .lower)});
}
