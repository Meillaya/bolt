const std = @import("std");

const Allocator = std.mem.Allocator;

pub const DEFAULT_MODEL = "offline-mock";
pub const MAX_API_RESPONSE: usize = 8 * 1024 * 1024;

pub const ApiResponse = struct {
    success: bool,
    retryable: bool = false,
    kind: []const u8,
    text: []const u8 = "",
    stop_reason: []const u8 = "",
    tool_name: []const u8 = "",
    tool_input_json: []const u8 = "",
    error_message: []const u8 = "",
    input_tokens: u64 = 0,
    output_tokens: u64 = 0,
};

pub const LiveConfig = struct {
    allow_live: bool = false,
    api_key: []const u8 = "",
    provider: []const u8 = "anthropic",
};

pub fn liveEnabled(config: LiveConfig) bool {
    return config.allow_live and config.api_key.len > 0 and config.provider.len > 0;
}

pub fn liveBlockedReason(config: LiveConfig) []const u8 {
    if (config.allow_live and config.api_key.len == 0) return "missing_api_key";
    if (!config.allow_live) return "live_disabled";
    if (config.provider.len == 0) return "missing_provider";
    return "none";
}

pub fn buildRequestJson(
    arena: Allocator,
    model: []const u8,
    system_prompt: []const u8,
    user_message: []const u8,
    tool_schemas: []const u8,
) ![]const u8 {
    const model_e = try jsonEscape(arena, model);
    const system_e = try jsonEscape(arena, system_prompt);
    const user_e = try jsonEscape(arena, user_message);
    return std.fmt.allocPrint(
        arena,
        "{{\"model\":\"{s}\",\"system\":\"{s}\"," ++
            "\"messages\":[{{\"role\":\"user\",\"content\":\"{s}\"}}]," ++
            "\"tools\":{s},\"offline\":true}}",
        .{ model_e, system_e, user_e, tool_schemas },
    );
}

pub fn parseOfflineResponse(
    arena: Allocator,
    raw: []const u8,
    max_bytes: usize,
) ApiResponse {
    std.debug.assert(max_bytes > 0);
    if (raw.len > max_bytes) {
        return .{
            .success = false,
            .kind = "truncated",
            .error_message = "response exceeded max_bytes",
        };
    }
    const trimmed = std.mem.trim(u8, raw, " \n\r\t");
    if (trimmed.len < 2 or trimmed[0] != '{') {
        return .{
            .success = false,
            .kind = "malformed",
            .error_message = "response is not a JSON object",
        };
    }
    if (contains(trimmed, "\"refusal\"") or contains(trimmed, "\"type\":\"refusal\"")) {
        return .{
            .success = true,
            .kind = "refusal",
            .text = extractString(arena, trimmed, "text") orelse "",
            .stop_reason = "refusal",
            .input_tokens = extractInt(trimmed, "input_tokens"),
            .output_tokens = extractInt(trimmed, "output_tokens"),
        };
    }
    if (contains(trimmed, "\"tool_use\"") or contains(trimmed, "\"tool_calls\"")) {
        return .{
            .success = true,
            .kind = "tool_use",
            .tool_name = extractString(arena, trimmed, "name") orelse "unknown",
            .tool_input_json = extractObjectAfter(trimmed, "input") orelse "{}",
            .stop_reason = "tool_use",
            .input_tokens = extractInt(trimmed, "input_tokens"),
            .output_tokens = extractInt(trimmed, "output_tokens"),
        };
    }
    if (extractString(arena, trimmed, "text")) |text| {
        return .{
            .success = true,
            .kind = "text",
            .text = text,
            .stop_reason = extractString(arena, trimmed, "stop_reason") orelse "end_turn",
            .input_tokens = extractInt(trimmed, "input_tokens"),
            .output_tokens = extractInt(trimmed, "output_tokens"),
        };
    }
    return .{
        .success = false,
        .kind = "malformed",
        .error_message = "no text, refusal, or tool_use content",
    };
}

pub fn redactSecrets(arena: Allocator, input: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < input.len) {
        if (startsAtAnyCase(input, i, "Authorization:")) {
            try out.appendSlice(arena, "Authorization: [REDACTED]");
            i = skipUntilLineEnd(input, i);
            continue;
        }
        if (startsAtAnyCase(input, i, "Cookie:") or startsAtAnyCase(input, i, "Set-Cookie:")) {
            try out.appendSlice(arena, "Cookie: [REDACTED]");
            i = skipUntilLineEnd(input, i);
            continue;
        }
        if (startsAtAnyCase(input, i, "Bearer ")) {
            try out.appendSlice(arena, "Bearer [REDACTED]");
            i += "Bearer ".len;
            while (i < input.len and !isSecretTerminator(input[i])) : (i += 1) {}
            continue;
        }
        if (startsAtJsonSecretKey(input, i)) {
            const key_end = std.mem.indexOfScalarPos(u8, input, i + 1, '"') orelse i;
            try out.appendSlice(arena, input[i .. key_end + 1]);
            var pos = key_end + 1;
            while (pos < input.len and (input[pos] == ' ' or input[pos] == '\t')) : (pos += 1) {}
            if (pos < input.len and input[pos] == ':') {
                try out.append(arena, ':');
                pos += 1;
                while (pos < input.len and (input[pos] == ' ' or input[pos] == '\t')) : (pos += 1) try out.append(arena, input[pos]);
                if (pos < input.len and input[pos] == '"') {
                    try out.appendSlice(arena, "\"[REDACTED]\"");
                    i = skipJsonString(input, pos);
                    continue;
                }
            }
        }
        const anthropic_prefix = "sk-" ++ "ant-";
        const generic_prefix = "sk" ++ "-";
        if (startsAt(input, i, anthropic_prefix) or startsAt(input, i, generic_prefix)) {
            try out.appendSlice(arena, "[REDACTED]");
            while (i < input.len and !isSecretTerminator(input[i])) : (i += 1) {}
            continue;
        }
        try out.append(arena, input[i]);
        i += 1;
    }
    return out.items;
}

fn isSecretTerminator(c: u8) bool {
    return std.ascii.isWhitespace(c) or c == '"' or c == '\'' or c == ',' or c == ';';
}

fn skipUntilLineEnd(input: []const u8, start_index: usize) usize {
    var i = start_index;
    while (i < input.len and input[i] != '\n' and input[i] != '\r') : (i += 1) {}
    return i;
}

fn skipJsonString(input: []const u8, quote: usize) usize {
    var i = quote + 1;
    var escaped = false;
    while (i < input.len) : (i += 1) {
        if (escaped) {
            escaped = false;
        } else if (input[i] == '\\') {
            escaped = true;
        } else if (input[i] == '"') {
            return i + 1;
        }
    }
    return input.len;
}

fn startsAtAnyCase(haystack: []const u8, index: usize, needle: []const u8) bool {
    if (index + needle.len > haystack.len) return false;
    return std.ascii.eqlIgnoreCase(haystack[index .. index + needle.len], needle);
}

fn startsAtJsonSecretKey(haystack: []const u8, index: usize) bool {
    if (index >= haystack.len or haystack[index] != '"') return false;
    const key_end = std.mem.indexOfScalarPos(u8, haystack, index + 1, '"') orelse return false;
    return isSensitiveJsonKey(haystack[index + 1 .. key_end]);
}

fn isSensitiveJsonKey(key: []const u8) bool {
    if (key.len == 0) return false;
    if (containsIgnoreCase(key, "authorization")) return true;
    if (containsIgnoreCase(key, "cookie")) return true;
    if (containsIgnoreCase(key, "token")) return true;
    if (containsIgnoreCase(key, "secret")) return true;
    if (containsIgnoreCase(key, "password")) return true;
    if (containsIgnoreCase(key, "credential")) return true;
    return containsIgnoreCase(key, "api") and containsIgnoreCase(key, "key");
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len == 0) return true;
    if (haystack.len < needle.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(haystack[i .. i + needle.len], needle)) return true;
    }
    return false;
}

pub fn jsonEscape(arena: Allocator, input: []const u8) ![]const u8 {
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

fn contains(haystack: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, haystack, needle) != null;
}

fn startsAt(haystack: []const u8, index: usize, needle: []const u8) bool {
    return index + needle.len <= haystack.len and
        std.mem.eql(u8, haystack[index .. index + needle.len], needle);
}

fn extractString(arena: Allocator, raw: []const u8, key: []const u8) ?[]const u8 {
    const pat = std.fmt.allocPrint(arena, "\"{s}\":\"", .{key}) catch return null;
    const start = std.mem.indexOf(u8, raw, pat) orelse return null;
    var pos = start + pat.len;
    var buf: std.ArrayList(u8) = .empty;
    while (pos < raw.len) : (pos += 1) {
        const c = raw[pos];
        if (c == '"') return buf.items;
        if (c == '\\' and pos + 1 < raw.len) {
            pos += 1;
            const esc = raw[pos];
            tryAppendEscaped(arena, &buf, esc) catch return null;
            continue;
        }
        buf.append(arena, c) catch return null;
    }
    return null;
}

fn tryAppendEscaped(arena: Allocator, buf: *std.ArrayList(u8), esc: u8) !void {
    switch (esc) {
        'n' => try buf.append(arena, '\n'),
        'r' => try buf.append(arena, '\r'),
        't' => try buf.append(arena, '\t'),
        else => try buf.append(arena, esc),
    }
}

fn extractObjectAfter(raw: []const u8, key: []const u8) ?[]const u8 {
    const key_pos = std.mem.indexOf(u8, raw, key) orelse return null;
    const brace_rel = std.mem.indexOfScalar(u8, raw[key_pos..], '{') orelse return null;
    const start = key_pos + brace_rel;
    var depth: i32 = 0;
    var pos = start;
    while (pos < raw.len) : (pos += 1) {
        if (raw[pos] == '{') depth += 1;
        if (raw[pos] == '}') {
            depth -= 1;
            if (depth == 0) return raw[start .. pos + 1];
        }
    }
    return null;
}

fn extractInt(raw: []const u8, key: []const u8) u64 {
    var arena_state = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const pat = std.fmt.allocPrint(arena, "\"{s}\":", .{key}) catch return 0;
    const start = std.mem.indexOf(u8, raw, pat) orelse return 0;
    var pos = start + pat.len;
    while (pos < raw.len and raw[pos] == ' ') : (pos += 1) {}
    const num_start = pos;
    while (pos < raw.len and std.ascii.isDigit(raw[pos])) : (pos += 1) {}
    if (pos == num_start) return 0;
    return std.fmt.parseInt(u64, raw[num_start..pos], 10) catch 0;
}

test "offline parser handles text response and usage" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const resp = parseOfflineResponse(arena, "{\"text\":\"hello\",\"stop_reason\":\"end_turn\",\"input_tokens\":3,\"output_tokens\":4}", MAX_API_RESPONSE);
    try std.testing.expect(resp.success);
    try std.testing.expect(std.mem.eql(u8, resp.kind, "text"));
    try std.testing.expectEqual(@as(u64, 3), resp.input_tokens);
}

test "offline parser handles tool use" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const resp = parseOfflineResponse(arena, "{\"type\":\"tool_use\",\"name\":\"check\",\"input\":{\"x\":1}}", MAX_API_RESPONSE);
    try std.testing.expect(resp.success);
    try std.testing.expect(std.mem.eql(u8, resp.kind, "tool_use"));
    try std.testing.expect(std.mem.eql(u8, resp.tool_name, "check"));
}

test "offline parser handles refusal malformed and truncation" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try std.testing.expect(parseOfflineResponse(arena, "{\"type\":\"refusal\",\"text\":\"no\"}", MAX_API_RESPONSE).success);
    try std.testing.expect(!parseOfflineResponse(arena, "not json", MAX_API_RESPONSE).success);
    try std.testing.expect(std.mem.eql(u8, parseOfflineResponse(arena, "{\"text\":\"too long\"}", 4).kind, "truncated"));
}

test "live API is disabled by default and secrets redact" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try std.testing.expect(!liveEnabled(.{}));
    try std.testing.expect(std.mem.eql(u8, liveBlockedReason(.{}), "live_disabled"));
    const fake_key = "sk-" ++ "ant-secret";
    const bearer = "plain-" ++ "token-123";
    const cookie = "session" ++ "=abc";
    const api_key = "api-" ++ "secret";
    const access_token = "tok-" ++ "secret";
    const input = try std.fmt.allocPrint(arena, "Authorization: Bearer {s}\nCookie: {s}\n{{\"api_key\":\"{s}\",\"apiKey\":\"{s}\",\"x-api-key\":\"{s}\",\"access_token\":\"{s}\"}}", .{ bearer, cookie, fake_key, api_key, api_key, access_token });
    const redacted = try redactSecrets(arena, input);
    try std.testing.expect(std.mem.indexOf(u8, redacted, fake_key) == null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, bearer) == null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, cookie) == null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, api_key) == null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, access_token) == null);
    try std.testing.expect(std.mem.indexOf(u8, redacted, "[REDACTED]") != null);
}
