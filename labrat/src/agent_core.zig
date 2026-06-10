const std = @import("std");
const api = @import("api_client.zig");

const Allocator = std.mem.Allocator;

pub const ToolMapping = struct {
    api_name: []const u8,
    toolbox_name: []const u8,
};

pub const ToolCall = struct {
    name: []const u8,
    input_json: []const u8 = "{}",
};

pub const ToolResult = struct {
    success: bool,
    toolbox_name: []const u8,
    content: []const u8,
};

pub const OfflineTurn = struct {
    response_kind: []const u8,
    tool_result: ?ToolResult = null,
    text: []const u8 = "",
    input_tokens: u64 = 0,
    output_tokens: u64 = 0,
};

pub fn dispatchTool(
    mappings: []const ToolMapping,
    call: ToolCall,
) ToolResult {
    for (mappings) |mapping| {
        if (std.mem.eql(u8, mapping.api_name, call.name)) {
            return .{
                .success = true,
                .toolbox_name = mapping.toolbox_name,
                .content = call.input_json,
            };
        }
    }
    return .{
        .success = false,
        .toolbox_name = "",
        .content = "unknown tool",
    };
}

pub fn runOfflineTurn(
    arena: Allocator,
    raw_response: []const u8,
    mappings: []const ToolMapping,
) OfflineTurn {
    const resp = api.parseOfflineResponse(arena, raw_response, api.MAX_API_RESPONSE);
    if (!resp.success) {
        return .{
            .response_kind = resp.kind,
            .text = resp.error_message,
        };
    }
    if (std.mem.eql(u8, resp.kind, "tool_use")) {
        const result = dispatchTool(
            mappings,
            .{ .name = resp.tool_name, .input_json = resp.tool_input_json },
        );
        return .{
            .response_kind = resp.kind,
            .tool_result = result,
            .input_tokens = resp.input_tokens,
            .output_tokens = resp.output_tokens,
        };
    }
    return .{
        .response_kind = resp.kind,
        .text = resp.text,
        .input_tokens = resp.input_tokens,
        .output_tokens = resp.output_tokens,
    };
}

pub fn buildUsageSummary(
    arena: Allocator,
    turns: []const OfflineTurn,
) ![]const u8 {
    var input: u64 = 0;
    var output: u64 = 0;
    var tool_calls: u64 = 0;
    for (turns) |turn| {
        input += turn.input_tokens;
        output += turn.output_tokens;
        if (turn.tool_result != null) tool_calls += 1;
    }
    return std.fmt.allocPrint(
        arena,
        "{{\"turns\":{d},\"tool_calls\":{d},\"input_tokens\":{d},\"output_tokens\":{d}}}",
        .{ turns.len, tool_calls, input, output },
    );
}

pub fn historySummary(
    arena: Allocator,
    history_jsonl: []const u8,
    max_entries: usize,
) ![]const u8 {
    std.debug.assert(max_entries > 0);
    var lines: std.ArrayList([]const u8) = .empty;
    var it = std.mem.splitScalar(u8, history_jsonl, '\n');
    while (it.next()) |line| {
        if (line.len > 0) try lines.append(arena, line);
    }
    const start = if (lines.items.len > max_entries) lines.items.len - max_entries else 0;
    var out: std.ArrayList(u8) = .empty;
    for (lines.items[start..], 0..) |line, i| {
        if (i > 0) try out.append(arena, '\n');
        try out.appendSlice(arena, line);
    }
    return out.items;
}

test "deterministic dispatcher maps API tool names to toolbox names" {
    const mappings = [_]ToolMapping{.{ .api_name = "run_command", .toolbox_name = "run-cmd" }};
    const result = dispatchTool(&mappings, .{ .name = "run_command", .input_json = "{\"command\":\"zig build test\"}" });
    try std.testing.expect(result.success);
    try std.testing.expect(std.mem.eql(u8, result.toolbox_name, "run-cmd"));
}

test "offline turn handles tool call text and malformed response" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mappings = [_]ToolMapping{.{ .api_name = "check", .toolbox_name = "check" }};
    const tool_turn = runOfflineTurn(arena, "{\"type\":\"tool_use\",\"name\":\"check\",\"input\":{}}", &mappings);
    try std.testing.expect(tool_turn.tool_result.?.success);
    const text_turn = runOfflineTurn(arena, "{\"text\":\"done\",\"input_tokens\":2,\"output_tokens\":5}", &mappings);
    try std.testing.expect(std.mem.eql(u8, text_turn.text, "done"));
    const malformed = runOfflineTurn(arena, "nope", &mappings);
    try std.testing.expect(std.mem.eql(u8, malformed.response_kind, "malformed"));
}

test "usage and history summaries are bounded" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const turns = [_]OfflineTurn{
        .{ .response_kind = "text", .input_tokens = 1, .output_tokens = 2 },
        .{ .response_kind = "tool_use", .tool_result = .{ .success = true, .toolbox_name = "check", .content = "{}" }, .input_tokens = 3, .output_tokens = 4 },
    };
    const usage = try buildUsageSummary(arena, &turns);
    try std.testing.expect(std.mem.indexOf(u8, usage, "\"tool_calls\":1") != null);
    const history = try historySummary(arena, "a\nb\nc\n", 2);
    try std.testing.expect(std.mem.eql(u8, history, "b\nc"));
}
