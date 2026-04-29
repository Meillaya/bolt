const std = @import("std");
const common = @import("common.zig");
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
        "usage: dump_llm_runtime <manifest-path>",
    );

    var context = try common.loadManifestContext(
        init.io,
        allocator,
        manifest_path,
        .llm,
    );
    defer common.deinitManifestContext(allocator, &context);

    var runtime_assets = try llm_assets.loadFromManifestContext(
        init.io,
        allocator,
        context,
    );
    defer runtime_assets.deinit(allocator);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.writeAll("{\n");
    try stdout.print("  \"family\": \"{s}\",\n", .{context.family.label()});
    try stdout.print("  \"fixture_name\": \"{s}\",\n", .{context.manifest.value.fixture_name});
    try stdout.print("  \"loader\": \"{s}\",\n", .{runtime_assets.assets.model.loader});
    try stdout.print("  \"model_name\": \"{s}\",\n", .{runtime_assets.assets.model.model_name});
    try stdout.print("  \"model_architecture\": \"{s}\",\n", .{runtime_assets.assets.model.architecture});
    try stdout.print("  \"model_vocab_size\": {d},\n", .{runtime_assets.assets.model.vocab_size});
    try stdout.print("  \"model_hidden_size\": {d},\n", .{runtime_assets.assets.model.hidden_size});
    try stdout.print("  \"model_context_length\": {d},\n", .{runtime_assets.assets.model.context_length});
    try stdout.print("  \"weights_format\": \"{s}\",\n", .{runtime_assets.assets.weights_format});
    try stdout.writeAll("  \"tokenizer_vocab\": [");
    for (runtime_assets.assets.tokenizer.vocab, 0..) |token, index| {
        if (index != 0) try stdout.writeAll(", ");
        try writeJsonString(stdout, token);
    }
    try stdout.writeAll("],\n");

    try stdout.writeAll("  \"output_projection_milli\": [");
    for (runtime_assets.assets.weights.output_projection, 0..) |weight, index| {
        if (index != 0) try stdout.writeAll(", ");
        try stdout.print("{d}", .{@as(usize, @intFromFloat((weight * 1000.0) + 0.5))});
    }
    try stdout.writeAll("],\n");

    const vocab_size = runtime_assets.assets.weights.vocabSize();
    try stdout.writeAll("  \"transition_bias_rows_milli\": [\n");
    for (0..vocab_size) |row_index| {
        try stdout.writeAll("    [");
        for (0..vocab_size) |column_index| {
            if (column_index != 0) try stdout.writeAll(", ");
            const bias_milli = try runtime_assets.assets.weights.transitionBiasMilli(
                row_index,
                column_index,
            );
            try stdout.print("{d}", .{bias_milli});
        }
        try stdout.writeAll("]");
        if (row_index + 1 != vocab_size) {
            try stdout.writeAll(",\n");
        } else {
            try stdout.writeAll("\n");
        }
    }
    try stdout.writeAll("  ]\n}\n");
    try stdout.flush();
}
