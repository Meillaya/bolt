const std = @import("std");
const mnist = @import("../models/mnist.zig");
const decoder = @import("../models/decoder.zig");
const llm_assets = @import("llm_assets.zig");
const reporting = @import("reporting.zig");
const payload_identity = @import("../fixtures/payload_identity.zig");
const llm_assertions = @import("../testing/llm_assertions.zig");
const llm_samples = @import("../testing/llm_samples.zig");
const mnist_samples = @import("../testing/mnist_samples.zig");

pub fn mnistOutput(
    io: std.Io,
    allocator: std.mem.Allocator,
    context: anytype,
) ![]u8 {
    var payload = try mnist.loadFixturePayloadFromFile(
        io,
        allocator,
        context.payload_path,
    );
    defer payload.deinit();

    try payload_identity.validate(
        context.manifest.value,
        payload.value.family,
        payload.value.fixture_name,
    );

    const summary = try mnist.summarizeFixture(payload.value);
    return reporting.formatMnistSummary(allocator, summary);
}

pub fn llmOutput(
    io: std.Io,
    allocator: std.mem.Allocator,
    context: anytype,
) ![]u8 {
    var payload = try decoder.loadFixturePayloadFromFile(
        io,
        allocator,
        context.payload_path,
    );
    defer payload.deinit();

    var runtime_assets = try llm_assets.loadFromManifestContext(
        io,
        allocator,
        context,
    );
    defer runtime_assets.deinit(allocator);

    try payload_identity.validate(
        context.manifest.value,
        payload.value.family,
        payload.value.fixture_name,
    );

    const summary = try decoder.summarizeFixtureWithRuntime(
        payload.value,
        runtime_assets.assets.tokenizer,
        runtime_assets.assets.weights,
    );
    return reporting.formatLlmSummary(allocator, summary);
}

test "mnist output helper formats predicted label" {
    const allocator = std.testing.allocator;
    const expected = mnist_samples.summary();
    const summary = try reporting.formatMnistSummary(allocator, expected);
    defer allocator.free(summary);

    try llm_assertions.expectContainsAll(summary, &.{
        std.fmt.comptimePrint("\"predicted_label\": {d}", .{expected.predicted_label}),
    });
}

test "llm output helper format includes weighted sum" {
    const allocator = std.testing.allocator;
    const expected = llm_samples.summary();
    const summary = try reporting.formatLlmSummary(allocator, expected);
    defer allocator.free(summary);

    try llm_assertions.expectContainsAll(summary, &.{
        std.fmt.comptimePrint("\"weighted_logit_sum_milli\": {d}", .{expected.weighted_logit_sum_milli}),
    });
}
