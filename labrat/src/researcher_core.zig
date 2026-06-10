const std = @import("std");

const Allocator = std.mem.Allocator;
const MAX_FILE_BYTES: usize = 2 * 1024 * 1024;

pub const CommandSpec = struct {
    name: []const u8,
    argv: []const []const u8,
    cwd: []const u8 = "../engine",
    artifact: []const u8,
};

const CommandResult = struct {
    name: []const u8,
    argv: []const []const u8,
    cwd: []const u8,
    artifact: []const u8,
    artifact_exists: bool,
    artifact_status_pass: bool,
};

pub fn writeResearcherSummary(
    allocator: Allocator,
    lane: []const u8,
    output_artifact: []const u8,
    commands: []const CommandSpec,
) !void {
    std.debug.assert(lane.len > 0);
    std.debug.assert(output_artifact.len > 0);
    std.debug.assert(commands.len > 0);

    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var results = try arena.alloc(CommandResult, commands.len);
    var status_pass = true;
    for (commands, 0..) |cmd, i| {
        const artifact_exists = fileExists(cmd.artifact);
        const artifact_pass = if (artifact_exists)
            artifactHasPassStatus(arena, cmd.artifact)
        else
            false;
        results[i] = .{
            .name = cmd.name,
            .argv = cmd.argv,
            .cwd = cmd.cwd,
            .artifact = cmd.artifact,
            .artifact_exists = artifact_exists,
            .artifact_status_pass = artifact_pass,
        };
        if (!artifact_exists or !artifact_pass) status_pass = false;
    }

    try writeSummary(arena, lane, output_artifact, results, status_pass);
    if (!status_pass) std.process.exit(1);
}

fn artifactHasPassStatus(
    arena: Allocator,
    path: []const u8,
) bool {
    const content = readSmallFile(arena, path) catch return false;
    const pass_markers = [_][]const u8{
        "\"status\": \"pass\"",
        "\"status\":\"pass\"",
        "\"passed\": true",
        "\"ok\": true",
        "\"passes_reference_relative_threshold\": true",
        "\"selected_label_parity\": true",
        "\"tokens_match\": true",
        "\"matches\": true",
    };
    for (pass_markers) |marker| {
        if (std.mem.indexOf(u8, content, marker) != null) return true;
    }
    return false;
}

fn readSmallFile(arena: Allocator, path: []const u8) ![]const u8 {
    const zpath = try arena.dupeZ(u8, path);
    const file = std.c.fopen(zpath.ptr, "rb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(file);
    var buf: std.ArrayList(u8) = .empty;
    var tmp: [4096]u8 = undefined;
    while (true) {
        const n = std.c.fread(&tmp, 1, tmp.len, file);
        if (n == 0) break;
        if (buf.items.len + n > MAX_FILE_BYTES) return error.FileTooLarge;
        try buf.appendSlice(arena, tmp[0..n]);
    }
    return buf.items;
}

fn writeFile(path: []const u8, content: []const u8) !void {
    var path_buf: [1024:0]u8 = undefined;
    if (path.len >= path_buf.len) return error.PathTooLong;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const file = std.c.fopen(&path_buf, "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(file);
    const n = std.c.fwrite(content.ptr, 1, content.len, file);
    if (n != content.len) return error.WriteFailed;
}

fn fileExists(path: []const u8) bool {
    var path_buf: [1024:0]u8 = undefined;
    if (path.len >= path_buf.len) return false;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const file = std.c.fopen(&path_buf, "rb") orelse return false;
    _ = std.c.fclose(file);
    return true;
}

fn writeSummary(
    arena: Allocator,
    lane: []const u8,
    output_artifact: []const u8,
    results: []const CommandResult,
    status_pass: bool,
) !void {
    const status = if (status_pass) "pass" else "blocked";
    const epoch = epochSeconds();
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\n");
    try appendField(&buf, arena, "schema_version", "1", true);
    try appendStringField(&buf, arena, "milestone", "Labrat Phase 2 M3 researcher parity", true);
    try appendStringField(&buf, arena, "lane", lane, true);
    try appendStringField(&buf, arena, "status", status, true);
    const epoch_str = try std.fmt.allocPrint(arena, "{d}", .{epoch});
    try appendField(&buf, arena, "timestamp_unix", epoch_str, true);
    try buf.appendSlice(arena, "  \"commands\": [\n");
    for (results, 0..) |result, i| {
        if (i > 0) try buf.appendSlice(arena, ",\n");
        const escaped_name = try jsonEscape(arena, result.name);
        const escaped_cwd = try jsonEscape(arena, result.cwd);
        const escaped_artifact = try jsonEscape(arena, result.artifact);
        const argv_json = try argvJson(arena, result.argv);
        try appendFmt(
            &buf,
            arena,
            "    {{\"name\":\"{s}\",\"argv\":{s}," ++
                "\"cwd\":\"{s}\",\"artifact\":\"{s}\"," ++
                "\"artifact_exists\":{},\"artifact_status_pass\":{}}}",
            .{
                escaped_name,
                argv_json,
                escaped_cwd,
                escaped_artifact,
                result.artifact_exists,
                result.artifact_status_pass,
            },
        );
    }
    try buf.appendSlice(arena, "\n  ]\n}\n");
    try writeFile(output_artifact, buf.items);
}

fn argvJson(arena: Allocator, argv: []const []const u8) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    try buf.append(arena, '[');
    for (argv, 0..) |arg, i| {
        if (i > 0) try buf.append(arena, ',');
        const escaped = try jsonEscape(arena, arg);
        try appendFmt(&buf, arena, "\"{s}\"", .{escaped});
    }
    try buf.append(arena, ']');
    return buf.items;
}

fn jsonEscape(arena: Allocator, input: []const u8) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    for (input) |c| switch (c) {
        '"' => try buf.appendSlice(arena, "\\\""),
        '\\' => try buf.appendSlice(arena, "\\\\"),
        '\n' => try buf.appendSlice(arena, "\\n"),
        '\r' => try buf.appendSlice(arena, "\\r"),
        '\t' => try buf.appendSlice(arena, "\\t"),
        else => try buf.append(arena, c),
    };
    return buf.items;
}

fn appendStringField(
    buf: *std.ArrayList(u8),
    arena: Allocator,
    key: []const u8,
    value: []const u8,
    comma: bool,
) !void {
    const escaped = try jsonEscape(arena, value);
    try appendFmt(
        buf,
        arena,
        "  \"{s}\": \"{s}\"{s}\n",
        .{ key, escaped, if (comma) "," else "" },
    );
}

fn appendField(
    buf: *std.ArrayList(u8),
    arena: Allocator,
    key: []const u8,
    value: []const u8,
    comma: bool,
) !void {
    try appendFmt(
        buf,
        arena,
        "  \"{s}\": {s}{s}\n",
        .{ key, value, if (comma) "," else "" },
    );
}

fn appendFmt(
    buf: *std.ArrayList(u8),
    arena: Allocator,
    comptime fmt: []const u8,
    args: anytype,
) !void {
    const text = try std.fmt.allocPrint(arena, fmt, args);
    try buf.appendSlice(arena, text);
}

fn epochSeconds() i64 {
    var tv: std.c.timeval = undefined;
    if (std.c.gettimeofday(&tv, null) == 0) return @intCast(tv.sec);
    return 0;
}
