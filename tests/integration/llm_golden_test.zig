const std = @import("std");
const bolt = @import("bolt");

fn fromEngineCwd(
    allocator: std.mem.Allocator,
    relative_path: []const u8,
) ![]u8 {
    return std.fs.path.join(allocator, &.{ "..", relative_path });
}

test "llm committed fixture payload matches committed runtime-bundle golden summary" {
    const allocator = std.testing.allocator;
    const fixture_dir = "../python/fixtures/llm";
    const manifest_path = try fromEngineCwd(
        allocator,
        "python/fixtures/llm/manifest.json",
    );
    defer allocator.free(manifest_path);

    var manifest = try bolt.fixtures.loadFromFile(
        std.testing.io,
        allocator,
        manifest_path,
    );
    defer manifest.deinit();
    try manifest.value.validate(.llm);

    const payload_path = try std.fs.path.join(allocator, &.{ fixture_dir, manifest.value.payload_file });
    defer allocator.free(payload_path);
    const expected_path = try std.fs.path.join(allocator, &.{ fixture_dir, manifest.value.expected_file });
    defer allocator.free(expected_path);
    const runtime_bundle_path = try std.fs.path.join(allocator, &.{ fixture_dir, manifest.value.runtime_bundle_file.? });
    defer allocator.free(runtime_bundle_path);

    var payload = try bolt.decoder.loadFixturePayloadFromFile(std.testing.io, allocator, payload_path);
    defer payload.deinit();

    var expected = try bolt.decoder.loadExpectedSummaryFromFile(std.testing.io, allocator, expected_path);
    defer expected.deinit();

    const Context = struct {
        runtime_bundle_path: ?[]u8,
    };
    const context: Context = .{
        .runtime_bundle_path = runtime_bundle_path,
    };
    var loaded_assets = try bolt.runtime.llm_assets.loadFromManifestContext(
        std.testing.io,
        allocator,
        context,
    );
    defer loaded_assets.deinit(allocator);

    try std.testing.expectEqualStrings("llm-smoke.tokenizer.json", loaded_assets.bundle.value.tokenizer_file);
    try std.testing.expectEqualStrings("llm-smoke.weights.bin", loaded_assets.bundle.value.weights_file);
    try std.testing.expectEqualStrings("llm-smoke.model.json", loaded_assets.bundle.value.model_file);
    try std.testing.expectEqualStrings("bolt-runtime-bundle-v1", loaded_assets.assets.model.loader);
    try std.testing.expectEqualStrings("llm-smoke-decoder", loaded_assets.assets.model.model_name);
    try bolt.runtime.llm_assets.validate(loaded_assets.assets);

    const summary = try bolt.decoder.summarizeFixtureWithRuntimeAssets(
        payload.value,
        loaded_assets.assets,
    );
    try bolt.decoder.validateSummary(summary, expected.value);
    try std.testing.expectEqualStrings("metal", summary.backend);
    try std.testing.expect(summary.dispatched_kernels.bias_add_f32);
    try std.testing.expect(summary.dispatched_kernels.softmax_f32);
    try std.testing.expectEqual(@as(usize, 343), summary.conditioned_next_probability_milli);
}
