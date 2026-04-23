const std = @import("std");
const common = @import("common.zig");
const fixtures = @import("bolt").fixtures;
const llm_assets = @import("bolt").runtime.llm_assets;

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = try common.expectSinglePathArg(
        allocator,
        args,
        "usage: fixture_inspect <manifest-path>",
    );

    var context = try common.loadManifestContextAuto(init.io, allocator, manifest_path);
    defer common.deinitManifestContext(allocator, &context);

    std.debug.print("family={s}\n", .{context.family.label()});
    std.debug.print("fixture_name={s}\n", .{context.manifest.value.fixture_name});
    std.debug.print("payload_file={s}\n", .{context.manifest.value.payload_file});
    std.debug.print("expected_file={s}\n", .{context.manifest.value.expected_file});
    if (context.manifest.value.runtime_bundle_file) |runtime_bundle_file| {
        std.debug.print("runtime_bundle_file={s}\n", .{runtime_bundle_file});
    }
    std.debug.print("fixture_version={d}\n", .{context.manifest.value.fixture_version});

    if (context.family == .llm) {
        var runtime_assets = try llm_assets.loadFromManifestContext(
            init.io,
            allocator,
            context,
        );
        defer runtime_assets.deinit(allocator);

        std.debug.print("tokenizer_file={s}\n", .{runtime_assets.bundle.value.tokenizer_file});
        std.debug.print("weights_file={s}\n", .{runtime_assets.bundle.value.weights_file});
        std.debug.print("tokenizer_vocab_size={d}\n", .{runtime_assets.assets.tokenizer.vocabSize()});
        std.debug.print("weights_vocab_size={d}\n", .{runtime_assets.assets.weights.vocabSize()});
        std.debug.print("transition_bias_value_count={d}\n", .{runtime_assets.assets.weights.transition_bias.len});
    }
}
