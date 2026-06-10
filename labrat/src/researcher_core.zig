const std = @import("std");

const Allocator = std.mem.Allocator;
const MAX_FILE_BYTES: usize = 2 * 1024 * 1024;

pub const ResearcherMode = enum {
    summary,
    bench_compare,
    summaries,
};

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
    try writeResearcherSummaryMode(
        allocator,
        lane,
        output_artifact,
        commands,
        .summary,
    );
}

pub fn writeResearcherSummaryMode(
    allocator: Allocator,
    lane: []const u8,
    output_artifact: []const u8,
    commands: []const CommandSpec,
    mode: ResearcherMode,
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

    try writeSummary(arena, lane, output_artifact, results, status_pass, mode);
    if (!status_pass) std.process.exit(1);
}

pub fn researcherModeFromArgs(args: []const []const u8) !ResearcherMode {
    if (args.len < 2) return .summary;
    return researcherModeFromArg(args[1]);
}

pub fn researcherModeFromArg(arg: ?[]const u8) !ResearcherMode {
    const value = arg orelse return .summary;
    if (std.mem.eql(u8, value, "summary")) return .summary;
    if (std.mem.eql(u8, value, "bench-compare")) return .bench_compare;
    if (std.mem.eql(u8, value, "summaries")) return .summaries;
    if (std.mem.eql(u8, value, "history")) return .summaries;
    return error.UnsupportedResearcherCommand;
}

fn artifactHasPassStatus(
    arena: Allocator,
    path: []const u8,
) bool {
    const content = readSmallFile(arena, path) catch return false;
    var parsed = std.json.parseFromSlice(
        std.json.Value,
        arena,
        content,
        .{ .allocate = .alloc_always },
    ) catch return false;
    defer parsed.deinit();
    return jsonValueHasPassStatus(parsed.value);
}

fn jsonValueHasPassStatus(value: std.json.Value) bool {
    switch (value) {
        .object => |object| {
            if (object.get("status")) |status| {
                if (status == .string and std.mem.eql(u8, status.string, "pass")) return true;
            }
            const pass_booleans = [_][]const u8{
                "passed",
                "ok",
                "passes_reference_relative_threshold",
                "selected_label_parity",
                "tokens_match",
                "matches",
                "matches_reference",
            };
            for (pass_booleans) |key| {
                if (object.get(key)) |field| {
                    if (field == .bool and field.bool) return true;
                }
            }
            var it = object.iterator();
            while (it.next()) |entry| {
                if (jsonValueHasPassStatus(entry.value_ptr.*)) return true;
            }
            return false;
        },
        .array => |array| {
            for (array.items) |item| {
                if (jsonValueHasPassStatus(item)) return true;
            }
            return false;
        },
        else => return false,
    }
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
    try ensureParentDir(path);
    var path_buf: [1024:0]u8 = undefined;
    if (path.len >= path_buf.len) return error.PathTooLong;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const file = std.c.fopen(&path_buf, "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(file);
    const n = std.c.fwrite(content.ptr, 1, content.len, file);
    if (n != content.len) return error.WriteFailed;
}

fn ensureParentDir(path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    try std.Io.Dir.cwd().createDirPath(io_state.io(), parent);
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
    mode: ResearcherMode,
) !void {
    const status = if (status_pass) "pass" else "blocked";
    const summaries_path = try researcherSummariesPath(arena, lane);
    try writeTextSummary(arena, summaries_path, lane, results, status_pass, mode);
    const epoch = epochSeconds();
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\n");
    try appendField(&buf, arena, "schema_version", "1", true);
    try appendStringField(&buf, arena, "gate", "Labrat researcher parity", true);
    try appendStringField(&buf, arena, "lane", lane, true);
    try appendStringField(&buf, arena, "mode", researcherModeName(mode), true);
    try appendStringField(&buf, arena, "status", status, true);
    try appendStringField(&buf, arena, "summaries_path", summaries_path, true);
    const epoch_str = try std.fmt.allocPrint(arena, "{d}", .{epoch});
    try appendField(&buf, arena, "timestamp_unix", epoch_str, true);
    try appendProductWorkflowAliases(&buf, arena, summaries_path);
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

fn appendProductWorkflowAliases(
    buf: *std.ArrayList(u8),
    arena: Allocator,
    summaries_path: []const u8,
) !void {
    const escaped = try jsonEscape(arena, summaries_path);
    try appendFmt(
        buf,
        arena,
        "  \"product_workflows\": {{\"bench_compare\":\"<researcher> bench-compare\",\"summaries\":\"cat {s}\"}},\n",
        .{escaped},
    );
}

fn writeTextSummary(
    arena: Allocator,
    path: []const u8,
    lane: []const u8,
    results: []const CommandResult,
    status_pass: bool,
    mode: ResearcherMode,
) !void {
    var buf: std.ArrayList(u8) = .empty;
    try appendFmt(&buf, arena, "Labrat {s} summaries\n", .{lane});
    try appendFmt(&buf, arena, "mode: {s}\n", .{researcherModeName(mode)});
    try appendFmt(&buf, arena, "status: {s}\n", .{if (status_pass) "pass" else "blocked"});
    try buf.appendSlice(arena, "commands:\n");
    for (results) |result| {
        try appendFmt(
            &buf,
            arena,
            "- {s}: artifact={s} exists={} pass={}\n",
            .{ result.name, result.artifact, result.artifact_exists, result.artifact_status_pass },
        );
    }
    try writeFile(path, buf.items);
}

fn researcherSummariesPath(arena: Allocator, lane: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(arena, "../artifacts/labrat-history/{s}/summaries.txt", .{lane});
}

fn researcherModeName(mode: ResearcherMode) []const u8 {
    return switch (mode) {
        .summary => "summary",
        .bench_compare => "bench-compare",
        .summaries => "summaries",
    };
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

test "researcher command aliases parse product workflow modes" {
    try std.testing.expectEqual(ResearcherMode.summary, try researcherModeFromArg(null));
    try std.testing.expectEqual(ResearcherMode.summary, try researcherModeFromArg("summary"));
    try std.testing.expectEqual(ResearcherMode.bench_compare, try researcherModeFromArg("bench-compare"));
    try std.testing.expectEqual(ResearcherMode.summaries, try researcherModeFromArg("summaries"));
    try std.testing.expectEqual(ResearcherMode.summaries, try researcherModeFromArg("history"));
    try std.testing.expectError(error.UnsupportedResearcherCommand, researcherModeFromArg("unknown"));
}

test "researcher summaries path is lane-scoped and artifact-local" {
    const path = try researcherSummariesPath(std.testing.allocator, "bonsai");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("../artifacts/labrat-history/bonsai/summaries.txt", path);
}

test "researcher artifact pass detection parses JSON fields" {
    var pass_status = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"status\":\"pass\"}", .{});
    defer pass_status.deinit();
    try std.testing.expect(jsonValueHasPassStatus(pass_status.value));

    var nested_match = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"real_generation_probe\":{\"matches_reference\":true}}", .{});
    defer nested_match.deinit();
    try std.testing.expect(jsonValueHasPassStatus(nested_match.value));

    var blocked = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, "{\"status\":\"blocked\",\"matches_reference\":false}", .{});
    defer blocked.deinit();
    try std.testing.expect(!jsonValueHasPassStatus(blocked.value));
}
