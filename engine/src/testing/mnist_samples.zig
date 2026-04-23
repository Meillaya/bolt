const std = @import("std");
const fixtures = @import("../fixtures/manifest.zig");
const mnist = @import("../models/mnist.zig");

pub const family_name = "mnist";
pub const fixture_name = "mnist-smoke";
pub const payload_file_name = "mnist-smoke.json";
pub const expected_file_name = "mnist-smoke.expected.json";
pub const description = "Deterministic smoke fixture for the compact non-LLM path.";
pub const predicted_label = 7;
pub const sample_payload_non_zero_count = 1;
pub const sample_payload_pixel_sum = 7;
pub const sample_summary_non_zero_count = 3;
pub const sample_rows = 28;
pub const sample_cols = 28;
pub const sample_element_count = sample_rows * sample_cols;
pub const top_pixel_index = sample_element_count - 1;
pub const top_pixel_row = sample_rows - 1;
pub const top_pixel_col = sample_cols - 1;
pub const top_pixel_value_milli = 4000;
pub const trace_top_pixel_value_milli = 7000;
pub const semantic_middle_index = 111;
pub const semantic_middle_row = 3;
pub const semantic_middle_value_milli = 2000;
pub const origin_index = 0;
pub const origin_row = 0;
pub const origin_col = 0;
pub const origin_value_milli = 1000;

pub fn summary() mnist.MnistFixtureSummary {
    return .{
        .family = family_name,
        .fixture_name = fixture_name,
        .rows = sample_rows,
        .cols = sample_cols,
        .element_count = sample_element_count,
        .pixel_sum = sample_payload_pixel_sum,
        .non_zero_count = sample_summary_non_zero_count,
        .predicted_label = predicted_label,
    };
}

pub fn manifest() fixtures.FixtureManifest {
    return .{
        .fixture_version = fixtures.current_fixture_version,
        .family = family_name,
        .fixture_name = fixture_name,
        .payload_file = payload_file_name,
        .expected_file = expected_file_name,
        .description = description,
    };
}

fn payloadPrefix(comptime payload_family: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\{{
        \\  "family": "{s}",
        \\  "fixture_name": "{s}",
        \\  "image_shape": {{"rows": {d}, "cols": {d}}},
        \\  "pixels": [
    , .{ payload_family, fixture_name, sample_rows, sample_cols });
}

const payload_suffix = std.fmt.comptimePrint(
    \\    {d}
    \\  ],
    \\  "expected_sum": {d},
    \\  "expected_non_zero_count": {d},
    \\  "expected_label": {d}
    \\}}
, .{ sample_payload_pixel_sum, sample_payload_pixel_sum, sample_payload_non_zero_count, predicted_label });

pub const payload_input =
    payloadPrefix(family_name) ++ ("0," ** top_pixel_index) ++
    payload_suffix;

pub const invalid_family_payload_input =
    payloadPrefix("llm") ++ ("0," ** top_pixel_index) ++
    payload_suffix;

pub fn parsePayload(allocator: std.mem.Allocator) !std.json.Parsed(mnist.MnistFixturePayload) {
    return mnist.parseFixturePayloadFromSlice(allocator, payload_input);
}

pub fn parseInvalidFamilyPayload(allocator: std.mem.Allocator) !std.json.Parsed(mnist.MnistFixturePayload) {
    return mnist.parseFixturePayloadFromSlice(allocator, invalid_family_payload_input);
}

fn originDebugPixel() mnist.MnistNonZeroPixel {
    return .{ .index = origin_index, .row = origin_row, .col = origin_col, .value_milli = origin_value_milli };
}

fn semanticMiddleDebugPixel() mnist.MnistNonZeroPixel {
    return .{ .index = semantic_middle_index, .row = semantic_middle_row, .col = top_pixel_col, .value_milli = semantic_middle_value_milli };
}

fn topDebugPixel() mnist.MnistNonZeroPixel {
    return .{ .index = top_pixel_index, .row = top_pixel_row, .col = top_pixel_col, .value_milli = top_pixel_value_milli };
}

pub fn debugPixelsIncludes() [2]mnist.MnistNonZeroPixel {
    return .{
        originDebugPixel(),
        topDebugPixel(),
    };
}

pub fn debugPixelsSemantic() [sample_summary_non_zero_count]mnist.MnistNonZeroPixel {
    return .{
        originDebugPixel(),
        semanticMiddleDebugPixel(),
        topDebugPixel(),
    };
}

pub fn trace(non_zero_pixels: []const mnist.MnistNonZeroPixel) mnist.MnistFixtureTrace {
    const base = summary();
    const top_pixel = topDebugPixel();
    return .{
        .family = base.family,
        .fixture_name = base.fixture_name,
        .rows = base.rows,
        .cols = base.cols,
        .pixel_sum = base.pixel_sum,
        .non_zero_count = non_zero_pixels.len,
        .predicted_label = base.predicted_label,
        .top_pixel_index = top_pixel.index,
        .top_pixel_row = top_pixel.row,
        .top_pixel_col = top_pixel.col,
        .top_pixel_value_milli = top_pixel.value_milli,
        .non_zero_pixels = non_zero_pixels,
    };
}
