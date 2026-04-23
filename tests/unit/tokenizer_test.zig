const std = @import("std");
const bolt = @import("bolt");

test "tokenizer config parses and round-trips the smoke prompt" {
    const allocator = std.testing.allocator;

    var parsed = try bolt.tokenizer.parseConfigFromSlice(
        allocator,
        bolt.tokenizer.sample_config_input,
    );
    defer parsed.deinit();

    const fixture_tokenizer = bolt.tokenizer.tokenizerFromConfig(parsed.value);
    const encoded = try fixture_tokenizer.encodeText(allocator, "zig metal bolt");
    defer allocator.free(encoded);

    try std.testing.expectEqual(@as(usize, 4), fixture_tokenizer.vocabSize());
    try std.testing.expectEqualSlices(usize, &.{ 1, 2, 3 }, encoded);
    try std.testing.expectEqualStrings("bolt", try fixture_tokenizer.decodeToken(3));
}
