const std = @import("std");
const buffer = @import("../tensor/buffer.zig");
const layout = @import("../tensor/layout.zig");
const metal = @import("../metal/context.zig");
const mnist_samples = @import("../testing/mnist_samples.zig");

pub const fixture_family = "mnist";

pub const MnistModel = struct {
    input_size: usize = 784,
    output_size: usize = 10,
};

pub const MnistFixturePayload = struct {
    family: []const u8,
    fixture_name: []const u8,
    image_shape: layout.Shape,
    pixels: []const f32,
    expected_sum: usize,
    expected_non_zero_count: usize,
    expected_label: usize,
};

pub const MnistFixtureSummary = struct {
    family: []const u8,
    fixture_name: []const u8,
    rows: usize,
    cols: usize,
    element_count: usize,
    pixel_sum: usize,
    non_zero_count: usize,
    predicted_label: usize,
};

pub const MnistExpectedSummary = MnistFixtureSummary;

pub const MnistNonZeroPixel = struct {
    index: usize,
    row: usize,
    col: usize,
    value_milli: usize,
};

pub const MnistFixtureTrace = struct {
    family: []const u8,
    fixture_name: []const u8,
    rows: usize,
    cols: usize,
    pixel_sum: usize,
    non_zero_count: usize,
    predicted_label: usize,
    top_pixel_index: usize,
    top_pixel_row: usize,
    top_pixel_col: usize,
    top_pixel_value_milli: usize,
    non_zero_pixels: []MnistNonZeroPixel,
};

pub fn parseFixturePayloadFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(MnistFixturePayload) {
    return std.json.parseFromSlice(
        MnistFixturePayload,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadFixturePayloadFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(MnistFixturePayload) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(input);

    return parseFixturePayloadFromSlice(allocator, input);
}

pub fn summarizeFixture(
    payload: MnistFixturePayload,
) !MnistFixtureSummary {
    return summarizeFixtureWithMetal(payload);
}

pub fn summarizeFixtureWithMetal(
    payload: MnistFixturePayload,
) !MnistFixtureSummary {
    const model = MnistModel{};
    const element_count = payload.image_shape.elementCount();
    if (!std.mem.eql(u8, payload.family, fixture_family)) {
        return error.UnexpectedFamily;
    }
    if (payload.fixture_name.len == 0) {
        return error.EmptyFixtureName;
    }
    if (element_count != payload.pixels.len) {
        return error.LengthMismatch;
    }
    if (element_count != model.input_size) {
        return error.UnexpectedInputSize;
    }

    var materialized_pixels = try materializePixels(
        std.heap.page_allocator,
        payload.image_shape,
        payload.pixels,
    );
    defer materialized_pixels.deinit();

    var sum: f32 = 0;
    var non_zero_count: usize = 0;
    for (materialized_pixels.values) |pixel| {
        sum += pixel;
        if (pixel != 0) {
            non_zero_count += 1;
        }
    }

    const pixel_sum = @as(usize, @intFromFloat(sum));
    if (pixel_sum != payload.expected_sum) {
        return error.UnexpectedPixelSum;
    }
    if (non_zero_count != payload.expected_non_zero_count) {
        return error.UnexpectedNonZeroCount;
    }

    const predicted_label = @mod(pixel_sum, model.output_size);
    if (predicted_label != payload.expected_label) {
        return error.UnexpectedPredictedLabel;
    }

    return .{
        .family = payload.family,
        .fixture_name = payload.fixture_name,
        .rows = payload.image_shape.rows,
        .cols = payload.image_shape.cols,
        .element_count = element_count,
        .pixel_sum = pixel_sum,
        .non_zero_count = non_zero_count,
        .predicted_label = predicted_label,
    };
}

pub fn parseExpectedSummaryFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(MnistExpectedSummary) {
    return std.json.parseFromSlice(
        MnistExpectedSummary,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadExpectedSummaryFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(MnistExpectedSummary) {
    const input = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(1024 * 1024),
    );
    defer allocator.free(input);

    return parseExpectedSummaryFromSlice(allocator, input);
}

pub fn validateSummary(
    summary: MnistFixtureSummary,
    expected: MnistExpectedSummary,
) !void {
    if (!std.mem.eql(u8, summary.family, expected.family)) {
        return error.FamilyMismatch;
    }
    if (!std.mem.eql(u8, summary.fixture_name, expected.fixture_name)) {
        return error.FixtureNameMismatch;
    }
    if (summary.rows != expected.rows) return error.RowsMismatch;
    if (summary.cols != expected.cols) return error.ColsMismatch;
    if (summary.element_count != expected.element_count) return error.ElementCountMismatch;
    if (summary.pixel_sum != expected.pixel_sum) return error.PixelSumMismatch;
    if (summary.non_zero_count != expected.non_zero_count) return error.NonZeroCountMismatch;
    if (summary.predicted_label != expected.predicted_label) return error.PredictedLabelMismatch;
}

pub fn traceFixture(
    allocator: std.mem.Allocator,
    payload: MnistFixturePayload,
) !MnistFixtureTrace {
    const summary = try summarizeFixtureWithMetal(payload);

    var materialized_pixels = try materializePixels(
        allocator,
        payload.image_shape,
        payload.pixels,
    );
    defer materialized_pixels.deinit();

    const non_zero_pixels = try allocator.alloc(MnistNonZeroPixel, summary.non_zero_count);
    errdefer allocator.free(non_zero_pixels);

    var top_pixel_index: usize = 0;
    var top_pixel_value: f32 = materialized_pixels.values[0];
    var non_zero_index: usize = 0;

    for (materialized_pixels.values, 0..) |pixel, index| {
        if (pixel > top_pixel_value) {
            top_pixel_value = pixel;
            top_pixel_index = index;
        }

        if (pixel != 0) {
            non_zero_pixels[non_zero_index] = .{
                .index = index,
                .row = index / payload.image_shape.cols,
                .col = @mod(index, payload.image_shape.cols),
                .value_milli = @as(usize, @intFromFloat((pixel * 1000.0) + 0.5)),
            };
            non_zero_index += 1;
        }
    }

    return .{
        .family = summary.family,
        .fixture_name = summary.fixture_name,
        .rows = summary.rows,
        .cols = summary.cols,
        .pixel_sum = summary.pixel_sum,
        .non_zero_count = summary.non_zero_count,
        .predicted_label = summary.predicted_label,
        .top_pixel_index = top_pixel_index,
        .top_pixel_row = top_pixel_index / payload.image_shape.cols,
        .top_pixel_col = @mod(top_pixel_index, payload.image_shape.cols),
        .top_pixel_value_milli = @as(usize, @intFromFloat((top_pixel_value * 1000.0) + 0.5)),
        .non_zero_pixels = non_zero_pixels,
    };
}

pub fn freeTrace(
    allocator: std.mem.Allocator,
    trace: *MnistFixtureTrace,
) void {
    allocator.free(trace.non_zero_pixels);
}

fn materializePixels(
    allocator: std.mem.Allocator,
    shape: layout.Shape,
    pixels: []const f32,
) !buffer.OwnedTensorF32 {
    var host_tensor = try buffer.OwnedTensorF32.initCopy(
        allocator,
        shape,
        pixels,
        .host,
    );
    errdefer host_tensor.deinit();

    if (!metal.Context.isAvailable()) {
        return host_tensor;
    }

    const metal_tensor = try host_tensor.roundTripThroughMetal();
    host_tensor.deinit();
    return metal_tensor;
}

fn parseSamplePayload(allocator: std.mem.Allocator) !std.json.Parsed(MnistFixturePayload) {
    return mnist_samples.parsePayload(allocator);
}

fn parseInvalidFamilyPayload(allocator: std.mem.Allocator) !std.json.Parsed(MnistFixturePayload) {
    return mnist_samples.parseInvalidFamilyPayload(allocator);
}

fn expectSampleSummary(summary: MnistFixtureSummary) !void {
    const expected = mnist_samples.summary();
    try std.testing.expectEqual(expected.element_count, summary.element_count);
    try std.testing.expectEqual(expected.pixel_sum, summary.pixel_sum);
    try std.testing.expectEqual(@as(usize, mnist_samples.sample_payload_non_zero_count), summary.non_zero_count);
    try std.testing.expectEqual(expected.predicted_label, summary.predicted_label);
}

fn expectSampleTrace(trace: MnistFixtureTrace) !void {
    const first_non_zero_pixel = trace.non_zero_pixels[0];
    try std.testing.expectEqual(@as(usize, mnist_samples.sample_payload_non_zero_count), trace.non_zero_pixels.len);
    try std.testing.expectEqual(@as(usize, mnist_samples.top_pixel_index), trace.top_pixel_index);
    try std.testing.expectEqual(@as(usize, mnist_samples.top_pixel_row), trace.top_pixel_row);
    try std.testing.expectEqual(@as(usize, mnist_samples.top_pixel_col), trace.top_pixel_col);
    try std.testing.expectEqual(@as(usize, mnist_samples.trace_top_pixel_value_milli), trace.top_pixel_value_milli);
    try std.testing.expectEqual(@as(usize, mnist_samples.top_pixel_row), first_non_zero_pixel.row);
    try std.testing.expectEqual(@as(usize, mnist_samples.top_pixel_col), first_non_zero_pixel.col);
}

test "summarize mnist fixture payload" {
    const allocator = std.testing.allocator;

    var parsed = try parseSamplePayload(allocator);
    defer parsed.deinit();

    const summary = try summarizeFixture(parsed.value);
    try expectSampleSummary(summary);
}

test "reject non-mnist payload family" {
    const allocator = std.testing.allocator;

    var parsed = try parseInvalidFamilyPayload(allocator);
    defer parsed.deinit();

    try std.testing.expectError(error.UnexpectedFamily, summarizeFixture(parsed.value));
}

test "validate mnist summary" {
    const expected = mnist_samples.summary();
    const summary = expected;
    try validateSummary(summary, expected);
}

test "trace mnist fixture payload" {
    const allocator = std.testing.allocator;

    var parsed = try parseSamplePayload(allocator);
    defer parsed.deinit();

    var trace = try traceFixture(allocator, parsed.value);
    defer freeTrace(allocator, &trace);

    try expectSampleTrace(trace);
}

test "mnist summary stays deterministic through the Metal-backed tensor round-trip" {
    const allocator = std.testing.allocator;

    var parsed = try parseSamplePayload(allocator);
    defer parsed.deinit();

    const summary = try summarizeFixtureWithMetal(parsed.value);
    const expected = mnist_samples.summary();

    try std.testing.expectEqual(expected.pixel_sum, summary.pixel_sum);
    try std.testing.expectEqual(expected.predicted_label, summary.predicted_label);
}
