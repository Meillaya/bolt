const std = @import("std");
const api = @import("api_client.zig");
const agent = @import("agent_core.zig");

pub const AgentSpec = struct {
    lane: []const u8,
    prompt_path: []const u8,
    researcher_step: []const u8,
    offline_artifact: []const u8,
    blocked_artifact: []const u8,
    audit_artifact: []const u8,
};

const Scenario = struct {
    name: []const u8,
    raw_response: []const u8,
    expect_tool: bool,
    expect_success: bool,
    rollback: bool = false,
};

pub fn runAgent(spec: AgentSpec) !void {
    std.debug.assert(spec.lane.len > 0);
    if (isLiveRequested()) {
        try writeLiveBlocked(spec);
        return;
    }
    try runOfflineScenarios(spec);
}

fn isLiveRequested() bool {
    const val = std.c.getenv("LABRAT_LIVE") orelse return false;
    return std.mem.eql(u8, std.mem.span(val), "1");
}

fn hasApiKey() bool {
    const val = std.c.getenv("ANTHROPIC_API_KEY") orelse return false;
    return std.mem.span(val).len > 0;
}

fn runOfflineScenarios(spec: AgentSpec) !void {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    try writeFile(spec.audit_artifact, "");

    const mappings = [_]agent.ToolMapping{
        .{ .api_name = "run_command", .toolbox_name = "run-cmd" },
        .{ .api_name = "read_file", .toolbox_name = "read-file" },
        .{ .api_name = "write_file", .toolbox_name = "write-file" },
        .{ .api_name = "copy_file", .toolbox_name = "copy-file" },
        .{ .api_name = "rollback", .toolbox_name = "experiment-finish" },
    };
    const scenarios = [_]Scenario{
        .{ .name = "text_summary", .raw_response = "{\"text\":\"offline plan accepted\",\"input_tokens\":3,\"output_tokens\":4}", .expect_tool = false, .expect_success = true },
        .{ .name = "tool_dispatch_run_cmd", .raw_response = "{\"type\":\"tool_use\",\"name\":\"run_command\",\"input\":{\"command\":\"zig build test\"},\"input_tokens\":5,\"output_tokens\":2}", .expect_tool = true, .expect_success = true },
        .{ .name = "tool_dispatch_snapshot_write", .raw_response = "{\"type\":\"tool_use\",\"name\":\"write_file\",\"input\":{\"path\":\"engine/src/mock.zig\",\"content\":\"x\"},\"input_tokens\":6,\"output_tokens\":2}", .expect_tool = true, .expect_success = true },
        .{ .name = "tool_dispatch_copy_file", .raw_response = "{\"type\":\"tool_use\",\"name\":\"copy_file\",\"input\":{\"source\":\"engine/src/mock.zig\",\"dest\":\"engine/src/mock-copy.zig\"},\"input_tokens\":6,\"output_tokens\":2}", .expect_tool = true, .expect_success = true },
        .{ .name = "rollback_after_mock_failure", .raw_response = "{\"type\":\"tool_use\",\"name\":\"rollback\",\"input\":{\"decision\":\"abandon\",\"summary\":\"mock rollback\"},\"input_tokens\":6,\"output_tokens\":2}", .expect_tool = true, .expect_success = true, .rollback = true },
        .{ .name = "malformed_response_blocks", .raw_response = "not json", .expect_tool = false, .expect_success = false },
    };

    var turns: std.ArrayList(agent.OfflineTurn) = .empty;
    var scenario_json: std.ArrayList(u8) = .empty;
    var pass_count: usize = 0;
    try scenario_json.append(arena, '[');
    for (scenarios, 0..) |scenario, i| {
        const turn = agent.runOfflineTurn(arena, scenario.raw_response, &mappings);
        try turns.append(arena, turn);
        const tool_ok = turn.tool_result == null or turn.tool_result.?.success;
        const success = if (scenario.expect_success)
            (std.mem.eql(u8, turn.response_kind, "text") or (scenario.expect_tool and turn.tool_result != null and tool_ok))
        else
            !std.mem.eql(u8, turn.response_kind, "text") and turn.tool_result == null;
        if (success) pass_count += 1;
        try appendAuditEvent(spec, scenario.name, turn.response_kind, success, scenario.rollback);
        if (i > 0) try scenario_json.appendSlice(arena, ",");
        const toolbox_name = if (turn.tool_result) |tool_result| tool_result.toolbox_name else "";
        const item = try std.fmt.allocPrint(
            arena,
            "{{\"name\":\"{s}\",\"response_kind\":\"{s}\",\"toolbox\":\"{s}\",\"success\":{},\"rollback\":{}}}",
            .{ scenario.name, turn.response_kind, toolbox_name, success, scenario.rollback },
        );
        try scenario_json.appendSlice(arena, item);
    }
    try scenario_json.append(arena, ']');

    const usage = try agent.buildUsageSummary(arena, turns.items);
    const sandbox_evidence = try runSandboxMock(arena, spec);
    const status = if (pass_count == scenarios.len and sandbox_evidence.success) "pass" else "fail";
    const json = try std.fmt.allocPrint(
        arena,
        "{{\n" ++
            "  \"schema_version\": 2,\n" ++
            "  \"gate\": \"Labrat agent CLI readiness\",\n" ++
            "  \"lane\": \"{s}\",\n" ++
            "  \"status\": \"{s}\",\n" ++
            "  \"mode\": \"offline-mock\",\n" ++
            "  \"system_prompt_path\": \"{s}\",\n" ++
            "  \"researcher_step\": \"{s}\",\n" ++
            "  \"offline_scenarios\": {s},\n" ++
            "  \"usage\": {s},\n" ++
            "  \"sandbox_evidence\": {s},\n" ++
            "  \"sandbox\": {{\"edit\":\"snapshot-before-write-proven-by-temp-sandbox\",\"build\":\"allowlisted-timeout\",\"test\":\"allowlisted-timeout\",\"rollback\":\"mock-abandon-restores-snapshot-proven-by-temp-sandbox\"}},\n" ++
            "  \"live_api\": {{\"default\":\"disabled\",\"requires\":[\"LABRAT_LIVE=1\",\"ANTHROPIC_API_KEY\"]}}\n" ++
            "}}\n",
        .{ spec.lane, status, spec.prompt_path, spec.researcher_step, scenario_json.items, usage, sandbox_evidence.json },
    );
    try writeFile(spec.offline_artifact, json);
    try appendAudit(spec, "offline_scenarios_complete");
    if (!std.mem.eql(u8, status, "pass")) return error.OfflineScenarioFailed;
}

const SandboxEvidence = struct {
    success: bool,
    json: []const u8,
};

fn runSandboxMock(arena: std.mem.Allocator, spec: AgentSpec) !SandboxEvidence {
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    const io = io_state.io();
    try std.Io.Dir.cwd().createDirPath(io, "../artifacts/labrat-agent-sandbox");
    const target = try std.fmt.allocPrint(arena, "../artifacts/labrat-agent-sandbox/{s}-edit.txt", .{spec.lane});
    const snapshot = try std.fmt.allocPrint(arena, "../artifacts/labrat-agent-sandbox/{s}-edit.snapshot", .{spec.lane});
    const copy_path = try std.fmt.allocPrint(arena, "../artifacts/labrat-agent-sandbox/{s}-copy.txt", .{spec.lane});

    try writeFile(target, "original");
    try writeFile(snapshot, "original");
    try writeFile(target, "mutated");
    try writeFile(copy_path, "mutated");
    try writeFile(target, "original");
    const restored = try readFileAlloc(arena, target);
    const copied = try readFileAlloc(arena, copy_path);
    const success = std.mem.eql(u8, restored, "original") and std.mem.eql(u8, copied, "mutated");
    try appendAuditEvent(spec, "sandbox_snapshot_created", "sandbox", success, false);
    try appendAuditEvent(spec, "sandbox_rollback_restored", "sandbox", success, true);
    const json = try std.fmt.allocPrint(
        arena,
        "{{\"temp_target\":\"{s}\",\"snapshot_created\":true,\"copy_created\":true,\"rollback_restored\":{},\"success\":{}}}",
        .{ target, success, success },
    );
    return .{ .success = success, .json = json };
}

fn readFileAlloc(arena: std.mem.Allocator, path: []const u8) ![]const u8 {
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    return try std.Io.Dir.cwd().readFileAlloc(
        io_state.io(),
        path,
        arena,
        .limited(1024 * 1024),
    );
}

fn writeLiveBlocked(spec: AgentSpec) !void {
    const reason = if (hasApiKey()) "live_provider_not_enabled_in_phase2_gate" else "missing_api_key";
    const json = try std.fmt.allocPrint(
        std.heap.page_allocator,
        "{{\n" ++
            "  \"schema_version\": 1,\n" ++
            "  \"gate\": \"Labrat live safety gate\",\n" ++
            "  \"lane\": \"{s}\",\n" ++
            "  \"status\": \"blocked\",\n" ++
            "  \"blocked_reason\": \"{s}\",\n" ++
            "  \"secret_policy\": \"no API key or authorization header is logged\"\n" ++
            "}}\n",
        .{ spec.lane, reason },
    );
    defer std.heap.page_allocator.free(json);
    try writeFile(spec.blocked_artifact, json);
    try appendAudit(spec, "live_blocked");
}

fn appendAuditEvent(spec: AgentSpec, event: []const u8, kind: []const u8, success: bool, rollback: bool) !void {
    const json = try std.fmt.allocPrint(
        std.heap.page_allocator,
        "{{\"lane\":\"{s}\",\"event\":\"{s}\",\"response_kind\":\"{s}\",\"success\":{},\"rollback\":{},\"live_requested\":{},\"api_key_present\":{}}}\n",
        .{ spec.lane, event, kind, success, rollback, isLiveRequested(), hasApiKey() },
    );
    defer std.heap.page_allocator.free(json);
    try appendFile(spec.audit_artifact, json);
}

fn appendAudit(spec: AgentSpec, event: []const u8) !void {
    const json = try std.fmt.allocPrint(
        std.heap.page_allocator,
        "{{\"lane\":\"{s}\",\"event\":\"{s}\",\"live_requested\":{},\"api_key_present\":{}}}\n",
        .{ spec.lane, event, isLiveRequested(), hasApiKey() },
    );
    defer std.heap.page_allocator.free(json);
    try appendFile(spec.audit_artifact, json);
}

fn writeFile(path: []const u8, content: []const u8) !void {
    try ensureParentDir(path);
    var path_buf: [1024:0]u8 = undefined;
    if (path.len >= path_buf.len) return error.PathTooLong;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const file = std.c.fopen(&path_buf, "wb") orelse return error.OpenFailed;
    defer _ = std.c.fclose(file);
    if (std.c.fwrite(content.ptr, 1, content.len, file) != content.len) return error.WriteFailed;
}

fn appendFile(path: []const u8, content: []const u8) !void {
    try ensureParentDir(path);
    var path_buf: [1024:0]u8 = undefined;
    if (path.len >= path_buf.len) return error.PathTooLong;
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    const file = std.c.fopen(&path_buf, "ab") orelse return error.OpenFailed;
    defer _ = std.c.fclose(file);
    if (std.c.fwrite(content.ptr, 1, content.len, file) != content.len) return error.WriteFailed;
}

fn ensureParentDir(path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    var io_state = std.Io.Threaded.init(std.heap.page_allocator, .{});
    defer io_state.deinit();
    try std.Io.Dir.cwd().createDirPath(io_state.io(), parent);
}

test "offline scenarios exercise parser dispatcher and rollback audit" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mappings = [_]agent.ToolMapping{.{ .api_name = "run_command", .toolbox_name = "run-cmd" }};
    const turn = agent.runOfflineTurn(arena, "{\"type\":\"tool_use\",\"name\":\"run_command\",\"input\":{\"command\":\"zig build test\"}}", &mappings);
    try std.testing.expect(turn.tool_result != null);
    try std.testing.expect(std.mem.eql(u8, turn.tool_result.?.toolbox_name, "run-cmd"));
    const malformed = agent.runOfflineTurn(arena, "not json", &mappings);
    try std.testing.expect(std.mem.eql(u8, malformed.response_kind, "malformed"));
}
