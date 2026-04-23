const std = @import("std");
const bolt = @import("bolt");

fn fromEngineCwd(
    allocator: std.mem.Allocator,
    relative_path: []const u8,
) ![]u8 {
    return std.fs.path.join(allocator, &.{ "..", relative_path });
}

test "decoder weights decode from binary bytes and preserve deterministic routing" {
    const allocator = std.testing.allocator;
    const weights_path = try fromEngineCwd(
        allocator,
        "python/fixtures/llm/llm-smoke.weights.bin",
    );
    defer allocator.free(weights_path);

    const decoded = try bolt.weights.loadValuesFromBinaryFile(
        std.testing.io,
        allocator,
        weights_path,
    );
    defer allocator.free(decoded);

    try std.testing.expectEqual(@as(usize, 20), decoded.len);

    const vocab_size = bolt.tokenizer.defaultTokenizer().vocabSize();
    const runtime_weights = bolt.weights.weightsFromSlices(
        decoded[0..vocab_size],
        decoded[vocab_size..],
    );

    const conditioned = try runtime_weights.conditionedTopToken(&.{ 0.1, 0.2, 0.9, 0.4 }, 3);
    const prompt_context = try runtime_weights.promptContextTopToken(&.{ 1, 2, 3 });

    try std.testing.expect(runtime_weights.hasValidTransitionBias());
    try std.testing.expectEqual(@as(usize, 1), conditioned.token_id);
    try std.testing.expectEqual(@as(usize, 2400), prompt_context.score_milli);
}
