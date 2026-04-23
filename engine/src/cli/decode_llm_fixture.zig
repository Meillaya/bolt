const std = @import("std");
const common = @import("common.zig");
const decoder = @import("bolt").decoder;
const llm_assets = @import("bolt").runtime.llm_assets;

fn writeJsonString(
    writer: anytype,
    value: []const u8,
) !void {
    try writer.writeByte('"');
    for (value) |byte| {
        switch (byte) {
            '\\' => try writer.writeAll("\\\\"),
            '"' => try writer.writeAll("\\\""),
            '\n' => try writer.writeAll("\\n"),
            '\r' => try writer.writeAll("\\r"),
            '\t' => try writer.writeAll("\\t"),
            else => try writer.writeByte(byte),
        }
    }
    try writer.writeByte('"');
}

fn parseStepCount(input: []const u8) !usize {
    const step_count = try std.fmt.parseUnsigned(usize, input, 10);
    if (step_count == 0) return error.InvalidStepCount;
    return step_count;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    if (args.len != 2 and args.len != 3) {
        std.debug.print("usage: decode_llm_fixture <manifest-path> [steps]\n", .{});
        return error.InvalidArguments;
    }

    const manifest_path = args[1];
    const step_count = if (args.len == 3) try parseStepCount(args[2]) else 3;

    var context = try common.loadManifestContext(
        init.io,
        allocator,
        manifest_path,
        .llm,
    );
    defer common.deinitManifestContext(allocator, &context);

    var payload = try decoder.loadFixturePayloadFromFile(
        init.io,
        allocator,
        context.payload_path,
    );
    defer payload.deinit();

    var runtime_assets = try llm_assets.loadFromManifestContext(
        init.io,
        allocator,
        context,
    );
    defer runtime_assets.deinit(allocator);

    var decode = try decoder.decodeFixtureWithRuntime(
        allocator,
        payload.value,
        runtime_assets.assets.tokenizer,
        runtime_assets.assets.weights,
        step_count,
    );
    defer decoder.freeDecode(allocator, &decode);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.writeAll("{\n");
    try stdout.print("  \"family\": \"{s}\",\n", .{decode.family});
    try stdout.print("  \"fixture_name\": \"{s}\",\n", .{decode.fixture_name});
    try stdout.writeAll("  \"prompt_text\": ");
    try writeJsonString(stdout, decode.prompt_text);
    try stdout.writeAll(",\n");
    try stdout.writeAll("  \"prompt_token_ids\": [");
    for (decode.prompt_token_ids, 0..) |token_id, index| {
        if (index != 0) try stdout.writeAll(", ");
        try stdout.print("{d}", .{token_id});
    }
    try stdout.writeAll("],\n");
    try stdout.writeAll("  \"generated\": [\n");
    for (decode.generated, 0..) |generated_token, index| {
        try stdout.writeAll("    {\n");
        try stdout.print("      \"step_index\": {d},\n", .{generated_token.step_index});
        try stdout.print("      \"source_context_token_count\": {d},\n", .{generated_token.source_context_token_count});
        try stdout.print("      \"source_token_id\": {d},\n", .{generated_token.source_token_id});
        try stdout.writeAll("      \"source_token_text\": ");
        try writeJsonString(stdout, generated_token.source_token_text);
        try stdout.writeAll(",\n");
        try stdout.print("      \"token_id\": {d},\n", .{generated_token.token_id});
        try stdout.writeAll("      \"token_text\": ");
        try writeJsonString(stdout, generated_token.token_text);
        try stdout.writeAll(",\n");
        try stdout.print("      \"projection_milli\": {d},\n", .{generated_token.projection_milli});
        try stdout.print("      \"context_bias_milli\": {d},\n", .{generated_token.context_bias_milli});
        try stdout.print("      \"conditioned_score_milli\": {d}\n", .{generated_token.conditioned_score_milli});
        try stdout.writeAll("    }");
        if (index + 1 != decode.generated.len) {
            try stdout.writeAll(",\n");
        } else {
            try stdout.writeAll("\n");
        }
    }
    try stdout.writeAll("  ]\n}\n");
    try stdout.flush();
}

test "parse step count" {
    try std.testing.expectEqual(@as(usize, 3), try parseStepCount("3"));
    try std.testing.expectError(error.InvalidStepCount, parseStepCount("0"));
}
