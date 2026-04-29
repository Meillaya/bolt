const std = @import("std");
const mnist = @import("../models/mnist.zig");
const decoder = @import("../models/decoder.zig");
const llm_assets = @import("llm_assets.zig");
const mnist_assets = @import("mnist_assets.zig");
const payload_identity = @import("../fixtures/payload_identity.zig");

pub fn mnistCheck(
    io: std.Io,
    allocator: std.mem.Allocator,
    context: anytype,
) !void {
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

    var expected = try mnist.loadExpectedSummaryFromFile(
        io,
        allocator,
        context.expected_path,
    );
    defer expected.deinit();

    var runtime_assets = try mnist_assets.loadFromManifestContext(
        io,
        allocator,
        context,
    );
    defer runtime_assets.deinit(allocator);

    const summary = try mnist.summarizeFixtureWithRuntime(
        payload.value,
        runtime_assets.assets,
    );
    try mnist.validateSummary(summary, expected.value);
}

pub fn llmCheck(
    io: std.Io,
    allocator: std.mem.Allocator,
    context: anytype,
) !void {
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

    var expected = try decoder.loadExpectedSummaryFromFile(
        io,
        allocator,
        context.expected_path,
    );
    defer expected.deinit();

    const summary = try decoder.summarizeFixtureWithRuntimeAssets(
        payload.value,
        runtime_assets.assets,
    );
    try decoder.validateSummary(summary, expected.value);
}
