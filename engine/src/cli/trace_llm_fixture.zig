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

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = try common.expectSinglePathArg(
        allocator,
        args,
        "usage: trace_llm_fixture <manifest-path>",
    );

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

    var trace = try decoder.traceFixtureWithRuntime(
        allocator,
        payload.value,
        runtime_assets.assets.tokenizer,
        runtime_assets.assets.weights,
    );
    defer decoder.freeTrace(allocator, &trace);

    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.writeAll("{\n");
    try stdout.print("  \"family\": \"{s}\",\n", .{trace.family});
    try stdout.print("  \"fixture_name\": \"{s}\",\n", .{trace.fixture_name});
    try stdout.writeAll("  \"prompt_text\": ");
    try writeJsonString(stdout, trace.prompt_text);
    try stdout.writeAll(",\n");
    try stdout.print("  \"prompt_tail_token_id\": {d},\n", .{trace.prompt_tail_token_id});
    try stdout.writeAll("  \"prompt_tail_token_text\": ");
    try writeJsonString(stdout, trace.prompt_tail_token_text);
    try stdout.writeAll(",\n");
    try stdout.print("  \"raw_top_token_id\": {d},\n", .{trace.raw_top_token_id});
    try stdout.writeAll("  \"raw_top_token_text\": ");
    try writeJsonString(stdout, trace.raw_top_token_text);
    try stdout.writeAll(",\n");
    try stdout.print("  \"conditioned_top_token_id\": {d},\n", .{trace.conditioned_top_token_id});
    try stdout.writeAll("  \"conditioned_top_token_text\": ");
    try writeJsonString(stdout, trace.conditioned_top_token_text);
    try stdout.writeAll(",\n");
    try stdout.writeAll("  \"candidates\": [\n");
    for (trace.candidates, 0..) |candidate, index| {
        try stdout.writeAll("    {\n");
        try stdout.print("      \"token_id\": {d},\n", .{candidate.token_id});
        try stdout.writeAll("      \"token_text\": ");
        try writeJsonString(stdout, candidate.token_text);
        try stdout.writeAll(",\n");
        try stdout.print("      \"raw_logit_milli\": {d},\n", .{candidate.raw_logit_milli});
        try stdout.print("      \"output_projection_milli\": {d},\n", .{candidate.output_projection_milli});
        try stdout.print("      \"transition_bias_milli\": {d},\n", .{candidate.transition_bias_milli});
        try stdout.print("      \"conditioned_score_milli\": {d}\n", .{candidate.conditioned_score_milli});
        try stdout.writeAll("    }");
        if (index + 1 != trace.candidates.len) {
            try stdout.writeAll(",\n");
        } else {
            try stdout.writeAll("\n");
        }
    }
    try stdout.writeAll("  ]\n}\n");
    try stdout.flush();
}
