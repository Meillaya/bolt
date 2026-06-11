const std = @import("std");
const builtin = @import("builtin");

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

const ArtifactMetadata = struct {
    command: []const u8,
    cwd: []const u8,
    git_commit: []const u8,
    timestamp_utc: []const u8,
    zig_version: []const u8,
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
            artifactMatchesCommandSchema(arena, cmd)
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

fn artifactMatchesCommandSchema(
    arena: Allocator,
    cmd: CommandSpec,
) bool {
    const content = readSmallFile(arena, cmd.artifact) catch return false;
    var parsed = std.json.parseFromSlice(
        std.json.Value,
        arena,
        content,
        .{ .allocate = .alloc_always },
    ) catch return false;
    defer parsed.deinit();
    return jsonArtifactMatchesCommandSchema(arena, parsed.value, cmd);
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
    return jsonArtifactHasKnownPassSchema(parsed.value, path);
}

fn jsonArtifactMatchesCommandSchema(
    arena: Allocator,
    value: std.json.Value,
    cmd: CommandSpec,
) bool {
    if (!jsonArtifactHasKnownPassSchema(value, cmd.artifact)) return false;
    return hasCommandProvenance(arena, value, cmd);
}

fn hasCommandProvenance(
    arena: Allocator,
    value: std.json.Value,
    cmd: CommandSpec,
) bool {
    const object = topObject(value) orelse return false;
    const command = stringField(object, "command") orelse return false;
    if (!nonEmptyStringField(object, "cwd")) return false;
    if (!nonEmptyStringField(object, "git_commit")) return false;
    if (!nonEmptyStringField(object, "timestamp_utc")) return false;
    const expected = argvCommand(arena, cmd.argv) catch return false;
    defer arena.free(expected);
    return std.mem.eql(u8, command, expected);
}

const PassValue = union(enum) {
    bool: bool,
    int: i64,
};

const ArtifactRule = struct {
    suffix: []const u8,
    artifact_type: []const u8,
    pass_path: []const []const u8,
    expected: PassValue,
    real_asset: bool = false,
};

const mnist_run_path = [_][]const u8{"passes_reference_relative_threshold"};
const mnist_one_bit_path = [_][]const u8{"selected_label_parity"};
const mnist_infer_path = [_][]const u8{ "inference", "passed" };
const metal_runtime_path = [_][]const u8{ "runtime", "passed" };
const bonsai_smoke_path = [_][]const u8{ "smoke", "passed" };
const bonsai_readiness_path = [_][]const u8{ "readiness", "passed" };
const bonsai_bench_path = [_][]const u8{ "benchmark", "passed" };
const q4_golden_path = [_][]const u8{ "golden", "tokens_match" };
const q4_bench_path = [_][]const u8{ "benchmark", "passed" };

const artifact_rules = [_]ArtifactRule{
    .{ .suffix = "bolt-mnist-run.json", .artifact_type = "engine.mnist.run", .pass_path = &mnist_run_path, .expected = .{ .bool = true } },
    .{ .suffix = "bolt-mnist-run-1bit.json", .artifact_type = "engine.mnist.run_1bit", .pass_path = &mnist_one_bit_path, .expected = .{ .bool = true } },
    .{ .suffix = "bolt-mnist-run-infer.json", .artifact_type = "engine.mnist.run_infer", .pass_path = &mnist_infer_path, .expected = .{ .bool = true } },
    .{ .suffix = "bolt-metal-mlp-runtime.json", .artifact_type = "engine.metal.mlp_runtime", .pass_path = &metal_runtime_path, .expected = .{ .bool = true } },
    .{ .suffix = "bolt-bonsai-smoke.json", .artifact_type = "engine.bonsai.smoke", .pass_path = &bonsai_smoke_path, .expected = .{ .bool = true }, .real_asset = true },
    .{ .suffix = "bolt-bonsai-readiness.json", .artifact_type = "engine.bonsai.readiness", .pass_path = &bonsai_readiness_path, .expected = .{ .bool = true }, .real_asset = true },
    .{ .suffix = "bolt-bonsai-bench.json", .artifact_type = "engine.bonsai.bench", .pass_path = &bonsai_bench_path, .expected = .{ .bool = true }, .real_asset = true },
    .{ .suffix = "bolt-q4-golden.json", .artifact_type = "engine.q4.golden", .pass_path = &q4_golden_path, .expected = .{ .bool = true }, .real_asset = true },
    .{ .suffix = "bolt-q4-bench.json", .artifact_type = "engine.q4.bench", .pass_path = &q4_bench_path, .expected = .{ .bool = true }, .real_asset = true },
};

fn jsonArtifactHasKnownPassSchema(value: std.json.Value, path: []const u8) bool {
    const rule = artifactRuleForPath(path) orelse return false;
    return commonEnvelopePasses(value, rule) and passConditionMatches(value, rule);
}

fn artifactRuleForPath(path: []const u8) ?ArtifactRule {
    for (artifact_rules) |rule| {
        if (std.mem.endsWith(u8, path, rule.suffix)) return rule;
    }
    return null;
}

fn commonEnvelopePasses(value: std.json.Value, rule: ArtifactRule) bool {
    return schemaString(value, "1") and
        topStringEquals(value, "artifact_type", rule.artifact_type) and
        topStringEquals(value, "status", "pass") and
        hasToolchain(value) and
        (!rule.real_asset or hasValidManifestDigest(value));
}

fn passConditionMatches(value: std.json.Value, rule: ArtifactRule) bool {
    const actual = nestedValue(value, rule.pass_path) orelse return false;
    return switch (rule.expected) {
        .bool => |expected| actual == .bool and actual.bool == expected,
        .int => |expected| actual == .integer and actual.integer == expected,
    };
}

fn nestedValue(value: std.json.Value, path: []const []const u8) ?std.json.Value {
    var current = value;
    for (path) |part| {
        const object = topObject(current) orelse return null;
        current = object.get(part) orelse return null;
    }
    return current;
}

fn hasToolchain(value: std.json.Value) bool {
    const object = topObject(value) orelse return false;
    const field = object.get("toolchain") orelse return false;
    if (field != .object) return false;
    var iterator = field.object.iterator();
    while (iterator.next()) |entry| {
        if (entry.value_ptr.* == .string and entry.value_ptr.string.len > 0) return true;
    }
    return false;
}

fn hasValidManifestDigest(value: std.json.Value) bool {
    const object = topObject(value) orelse return false;
    const digest = stringField(object, "manifest_digest") orelse return false;
    if (!std.mem.startsWith(u8, digest, "sha256:")) return false;
    if (digest.len != "sha256:".len + 64) return false;
    for (digest["sha256:".len..]) |c| {
        const lower = std.ascii.toLower(c);
        if (!((lower >= '0' and lower <= '9') or (lower >= 'a' and lower <= 'f'))) return false;
    }
    return true;
}

fn schemaString(value: std.json.Value, expected: []const u8) bool {
    return topStringEquals(value, "schema_version", expected);
}

fn topStringEquals(value: std.json.Value, key: []const u8, expected: []const u8) bool {
    const object = topObject(value) orelse return false;
    const actual = stringField(object, key) orelse return false;
    return std.mem.eql(u8, actual, expected);
}

fn topObject(value: std.json.Value) ?std.json.ObjectMap {
    return switch (value) {
        .object => |object| object,
        else => null,
    };
}

fn stringField(object: std.json.ObjectMap, key: []const u8) ?[]const u8 {
    const field = object.get(key) orelse return null;
    if (field != .string) return null;
    return field.string;
}

fn nonEmptyStringField(object: std.json.ObjectMap, key: []const u8) bool {
    const value = stringField(object, key) orelse return false;
    return value.len > 0;
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
    const command = try researcherCommand(arena, lane);
    const meta = try captureArtifactMetadata(arena, command);
    var buf: std.ArrayList(u8) = .empty;
    try buf.appendSlice(arena, "{\n");
    try appendStringField(&buf, arena, "schema_version", "1", true);
    try appendStringField(&buf, arena, "artifact_type", "labrat.researcher", true);
    try appendStringField(&buf, arena, "status", status, true);
    try appendStringField(&buf, arena, "command", meta.command, true);
    try appendStringField(&buf, arena, "cwd", meta.cwd, true);
    try appendStringField(&buf, arena, "git_commit", meta.git_commit, true);
    try appendStringField(&buf, arena, "timestamp_utc", meta.timestamp_utc, true);
    try appendFmt(&buf, arena, "  \"toolchain\": {{\"zig\":\"{s}\"}},\n", .{meta.zig_version});
    try appendStringField(&buf, arena, "gate", "Labrat researcher parity", true);
    try appendStringField(&buf, arena, "lane", lane, true);
    try appendStringField(&buf, arena, "mode", researcherModeName(mode), true);
    try appendStringField(&buf, arena, "summaries_path", summaries_path, true);
    try appendFmt(&buf, arena, "  \"researcher\": {{\"passed\":{}}},\n", .{status_pass});
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

fn researcherCommand(arena: Allocator, lane: []const u8) ![]const u8 {
    const step = try laneBuildStep(arena, lane);
    return try std.fmt.allocPrint(arena, "zig build {s}-researcher --summary all", .{step});
}

fn laneBuildStep(arena: Allocator, lane: []const u8) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    for (lane) |c| {
        try buf.append(arena, if (c == '_') '-' else c);
    }
    return buf.items;
}

fn argvCommand(arena: Allocator, argv: []const []const u8) ![]const u8 {
    var buf: std.ArrayList(u8) = .empty;
    for (argv, 0..) |arg, i| {
        if (i > 0) try buf.append(arena, ' ');
        try buf.appendSlice(arena, arg);
    }
    try buf.appendSlice(arena, " --summary all");
    return buf.items;
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

fn captureArtifactMetadata(arena: Allocator, command: []const u8) !ArtifactMetadata {
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();
    return .{
        .command = command,
        .cwd = try commandOutput(arena, io, &.{"pwd"}),
        .git_commit = try gitIdentity(arena, io),
        .timestamp_utc = try formatTimestampUtcAlloc(arena, io),
        .zig_version = builtin.zig_version_string,
    };
}

fn gitIdentity(arena: Allocator, io: std.Io) ![]const u8 {
    const sha = commandOutput(arena, io, &.{ "git", "rev-parse", "--short", "HEAD" }) catch return "dirty";
    if (std.mem.eql(u8, sha, "unknown")) return "dirty";
    const status = commandOutput(arena, io, &.{ "git", "status", "--short" }) catch "dirty";
    if (status.len == 0) return sha;
    return try std.fmt.allocPrint(arena, "{s}-dirty", .{sha});
}

fn commandOutput(arena: Allocator, io: std.Io, argv: []const []const u8) ![]const u8 {
    const result = std.process.run(arena, io, .{
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

fn formatTimestampUtcAlloc(arena: Allocator, io: std.Io) ![]const u8 {
    const ts: u64 = @intCast(std.Io.Timestamp.now(io, .real).toSeconds());
    const es = std.time.epoch.EpochSeconds{ .secs = ts };
    const epoch_day = es.getEpochDay();
    const year_day = epoch_day.calculateYearDay();
    const month_day = year_day.calculateMonthDay();
    const day_secs = es.getDaySeconds();
    return try std.fmt.allocPrint(
        arena,
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

test "researcher artifact pass detection validates exact schemas" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var mnist_pass = try std.json.parseFromSlice(
        std.json.Value,
        arena,
        "{\"schema_version\":\"1\",\"artifact_type\":\"engine.mnist.run\",\"status\":\"pass\",\"toolchain\":{\"zig\":\"0.16.0\"},\"selected_label_parity\":false,\"passes_reference_relative_threshold\":true}",
        .{},
    );
    defer mnist_pass.deinit();
    try std.testing.expect(jsonArtifactHasKnownPassSchema(mnist_pass.value, "../artifacts/bolt-mnist-run.json"));

    var nested_match = try std.json.parseFromSlice(
        std.json.Value,
        arena,
        "{\"schema_version\":1,\"gate\":\"bolt-bonsai-q4-golden\",\"status\":\"blocked\",\"matches_reference\":true,\"tokenizer_pass\":true,\"tensor_pass\":true,\"integrated_metal_decode\":{\"used\":true},\"metal_q4mv_probe\":{\"pass\":true}}",
        .{},
    );
    defer nested_match.deinit();
    try std.testing.expect(!jsonArtifactHasKnownPassSchema(nested_match.value, "../artifacts/bolt-q4-golden.json"));

    var wrong_schema = try std.json.parseFromSlice(
        std.json.Value,
        arena,
        "{\"schema_version\":2,\"gate\":\"bolt-bonsai-bench\",\"status\":\"pass\",\"correctness_gate\":{\"passed\":true}}",
        .{},
    );
    defer wrong_schema.deinit();
    try std.testing.expect(!jsonArtifactHasKnownPassSchema(wrong_schema.value, "../artifacts/bolt-bonsai-bench.json"));
}

test "researcher rejects missing or stale command provenance" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cmd = CommandSpec{
        .name = "run",
        .argv = &.{ "zig", "build", "run" },
        .artifact = "../artifacts/bolt-mnist-run.json",
    };

    var missing_provenance = try std.json.parseFromSlice(
        std.json.Value,
        arena,
        "{\"schema_version\":\"1\",\"artifact_type\":\"engine.mnist.run\",\"status\":\"pass\",\"toolchain\":{\"zig\":\"0.16.0\"},\"selected_label_parity\":false,\"passes_reference_relative_threshold\":true}",
        .{},
    );
    defer missing_provenance.deinit();
    try std.testing.expect(!jsonArtifactMatchesCommandSchema(arena, missing_provenance.value, cmd));

    var stale_command = try std.json.parseFromSlice(
        std.json.Value,
        arena,
        "{\"schema_version\":\"1\",\"artifact_type\":\"engine.mnist.run\",\"status\":\"pass\",\"command\":\"zig build run-1bit\",\"cwd\":\"/tmp/bolt/engine\",\"git_commit\":\"abc1234\",\"timestamp_utc\":\"2026-06-10T00:00:00Z\",\"toolchain\":{\"zig\":\"0.16.0\"},\"selected_label_parity\":false,\"passes_reference_relative_threshold\":true}",
        .{},
    );
    defer stale_command.deinit();
    try std.testing.expect(!jsonArtifactMatchesCommandSchema(arena, stale_command.value, cmd));

    var fresh_command = try std.json.parseFromSlice(
        std.json.Value,
        arena,
        "{\"schema_version\":\"1\",\"artifact_type\":\"engine.mnist.run\",\"status\":\"pass\",\"command\":\"zig build run --summary all\",\"cwd\":\"/tmp/bolt/engine\",\"git_commit\":\"abc1234\",\"timestamp_utc\":\"2026-06-10T00:00:00Z\",\"toolchain\":{\"zig\":\"0.16.0\"},\"selected_label_parity\":false,\"passes_reference_relative_threshold\":true}",
        .{},
    );
    defer fresh_command.deinit();
    try std.testing.expect(jsonArtifactMatchesCommandSchema(arena, fresh_command.value, cmd));
}

test "researcher rejects nested pass top fail artifact" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    try std.testing.expect(!artifactHasPassStatus(
        arena_state.allocator(),
        "testdata/researcher/nested-pass-top-fail.json",
    ));
}
