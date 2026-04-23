const std = @import("std");
const mnist = @import("../models/mnist.zig");
const decoder = @import("../models/decoder.zig");
const llm_assertions = @import("../testing/llm_assertions.zig");
const llm_samples = @import("../testing/llm_samples.zig");
const mnist_samples = @import("../testing/mnist_samples.zig");

pub fn formatMnistSummary(
    allocator: std.mem.Allocator,
    summary: mnist.MnistFixtureSummary,
) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{{\n" ++
            "  \"family\": \"{s}\",\n" ++
            "  \"fixture_name\": \"{s}\",\n" ++
            "  \"rows\": {d},\n" ++
            "  \"cols\": {d},\n" ++
            "  \"element_count\": {d},\n" ++
            "  \"pixel_sum\": {d},\n" ++
            "  \"non_zero_count\": {d},\n" ++
            "  \"predicted_label\": {d}\n" ++
            "}}\n",
        .{
            summary.family,
            summary.fixture_name,
            summary.rows,
            summary.cols,
            summary.element_count,
            summary.pixel_sum,
            summary.non_zero_count,
            summary.predicted_label,
        },
    );
}

pub fn formatLlmSummary(
    allocator: std.mem.Allocator,
    summary: decoder.DecoderFixtureSummary,
) ![]u8 {
    try decoder.validateSummaryConsistency(summary);
    const raw = summary.route_comparison.raw;
    const conditioned = summary.route_comparison.conditioned;
    const model = summary.route_comparison.model;
    const prompt_context = summary.route_comparison.prompt_context;
    const preferred = summary.route_comparison.preferred;
    const gains = summary.route_comparison.gains;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer aw.deinit();

    try aw.writer.print(
        "{{\n" ++
            "  \"family\": \"{s}\",\n" ++
            "  \"fixture_name\": \"{s}\",\n" ++
            "  \"prompt_token_count\": {d},\n" ++
            "  \"prompt_sum\": {d},\n" ++
            "  \"logits_count\": {d},\n" ++
            "  \"next_token_id\": {d},\n" ++
            "  \"generated_token_count\": {d},\n" ++
            "  \"generated_last_token\": {d},\n" ++
            "  \"prompt_tail_token_id\": {d},\n" ++
            "  \"prompt_tail_token_text\": \"{s}\",\n" ++
            "  \"next_token_text\": \"{s}\",\n" ++
            "  \"raw_next_score_milli\": {d},\n" ++
            "  \"conditioned_next_token_id\": {d},\n" ++
            "  \"conditioned_next_token_text\": \"{s}\",\n",
        .{
            summary.family,
            summary.fixture_name,
            summary.prompt_token_count,
            summary.prompt_sum,
            summary.logits_count,
            raw.token_id,
            summary.generated_token_count,
            preferred.token_id,
            summary.prompt_tail_token_id,
            summary.prompt_tail_token_text,
            raw.token_text,
            raw.score_milli,
            conditioned.token_id,
            conditioned.token_text,
        },
    );
    try aw.writer.print(
        "  \"model_next_token_id\": {d},\n" ++
            "  \"model_next_token_text\": \"{s}\",\n" ++
            "  \"prompt_context_next_token_id\": {d},\n" ++
            "  \"prompt_context_next_token_text\": \"{s}\",\n" ++
            "  \"preferred_route\": \"{s}\",\n" ++
            "  \"preferred_next_token_id\": {d},\n" ++
            "  \"preferred_next_token_text\": \"{s}\",\n" ++
            "  \"preferred_next_score_milli\": {d},\n" ++
            "  \"generated_projection_milli\": {d},\n" ++
            "  \"generated_context_bias_milli\": {d},\n" ++
            "  \"prompt_context_over_raw_gain_milli\": {d},\n" ++
            "  \"prompt_context_over_conditioned_gain_milli\": {d},\n" ++
            "  \"prompt_context_over_model_gain_milli\": {d},\n",
        .{
            model.token_id,
            model.token_text,
            prompt_context.token_id,
            prompt_context.token_text,
            preferred.route,
            preferred.token_id,
            preferred.token_text,
            preferred.score_milli,
            prompt_context.projection_milli.?,
            prompt_context.context_bias_milli.?,
            gains.prompt_context_over_raw_milli,
            gains.prompt_context_over_conditioned_milli,
            gains.prompt_context_over_model_milli,
        },
    );
    try aw.writer.print(
        "  \"raw_conditioning_flipped\": {any},\n" ++
            "  \"conditioned_matches_model\": {any},\n" ++
            "  \"route_comparison\": {{\n" ++
            "    \"raw\": {{ \"route\": \"{s}\", \"token_id\": {d}, \"token_text\": \"{s}\", \"score_milli\": {d} }},\n" ++
            "    \"conditioned\": {{ \"route\": \"{s}\", \"token_id\": {d}, \"token_text\": \"{s}\", \"score_milli\": {d} }},\n" ++
            "    \"model\": {{ \"route\": \"{s}\", \"token_id\": {d}, \"token_text\": \"{s}\", \"score_milli\": {d} }},\n",
        .{
            summary.raw_conditioning_flipped,
            summary.conditioned_matches_model,
            raw.route,
            raw.token_id,
            raw.token_text,
            raw.score_milli,
            conditioned.route,
            conditioned.token_id,
            conditioned.token_text,
            conditioned.score_milli,
            model.route,
            model.token_id,
            model.token_text,
            model.score_milli,
        },
    );
    try aw.writer.print(
        "    \"prompt_context\": {{ \"route\": \"{s}\", \"token_id\": {d}, \"token_text\": \"{s}\", \"projection_milli\": {d}, \"context_bias_milli\": {d}, \"score_milli\": {d} }},\n" ++
            "    \"preferred\": {{ \"route\": \"{s}\", \"token_id\": {d}, \"token_text\": \"{s}\", \"projection_milli\": {d}, \"context_bias_milli\": {d}, \"score_milli\": {d} }},\n" ++
            "    \"gains\": {{ \"raw_to_conditioned_milli\": {d}, \"conditioned_to_model_milli\": {d}, \"model_to_prompt_context_milli\": {d}, \"prompt_context_over_raw_milli\": {d}, \"prompt_context_over_conditioned_milli\": {d}, \"prompt_context_over_model_milli\": {d} }}\n" ++
            "  }},\n",
        .{
            prompt_context.route,
            prompt_context.token_id,
            prompt_context.token_text,
            prompt_context.projection_milli.?,
            prompt_context.context_bias_milli.?,
            prompt_context.score_milli,
            preferred.route,
            preferred.token_id,
            preferred.token_text,
            preferred.projection_milli.?,
            preferred.context_bias_milli.?,
            preferred.score_milli,
            gains.raw_to_conditioned_milli,
            gains.conditioned_to_model_milli,
            gains.model_to_prompt_context_milli,
            gains.prompt_context_over_raw_milli,
            gains.prompt_context_over_conditioned_milli,
            gains.prompt_context_over_model_milli,
        },
    );
    try aw.writer.print(
        "  \"top_token_weight_milli\": {d},\n" ++
            "  \"weighted_logit_sum_milli\": {d},\n" ++
            "  \"prompt_condition_bias_milli\": {d},\n" ++
            "  \"prompt_context_bias_milli\": {d},\n" ++
            "  \"conditioned_next_score_milli\": {d},\n" ++
            "  \"model_next_score_milli\": {d},\n" ++
            "  \"prompt_context_next_score_milli\": {d},\n" ++
            "  \"model_condition_gap_milli\": {d},\n" ++
            "  \"logit_margin_milli\": {d}\n" ++
            "}}\n",
        .{
            summary.top_token_weight_milli,
            summary.weighted_logit_sum_milli,
            summary.prompt_condition_bias_milli,
            prompt_context.context_bias_milli.?,
            conditioned.score_milli,
            model.score_milli,
            prompt_context.score_milli,
            gains.conditioned_to_model_milli,
            summary.logit_margin_milli,
        },
    );

    return aw.toOwnedSlice();
}

test "format mnist summary" {
    const allocator = std.testing.allocator;
    const expected = mnist_samples.summary();
    const output = try formatMnistSummary(allocator, expected);
    defer allocator.free(output);

    try llm_assertions.expectContainsAll(output, &.{
        std.fmt.comptimePrint("\"predicted_label\": {d}", .{expected.predicted_label}),
    });
}

test "format llm summary" {
    const allocator = std.testing.allocator;
    var summary = llm_samples.summary();
    summary.prompt_token_count = 4;
    summary.prompt_sum = 11;
    summary.generated_token_count = 5;

    const output = try formatLlmSummary(allocator, summary);
    defer allocator.free(output);

    try llm_assertions.expectSummaryContainsCore(output);
}

test "format llm summary parses route comparison JSON" {
    const allocator = std.testing.allocator;
    const expected = llm_samples.summary();

    const output = try formatLlmSummary(allocator, expected);
    defer allocator.free(output);

    var parsed = try llm_assertions.parseJsonObject(allocator, output);
    defer parsed.deinit();

    const root = parsed.value.object;
    try llm_assertions.expectSummaryReportSemanticRoot(root);
}

test "format llm summary rejects inconsistent compatibility fields" {
    const allocator = std.testing.allocator;
    const summary = llm_samples.invalidPreferredScoreSummary();
    try std.testing.expectError(error.PreferredNextScoreRouteComparisonMismatch, formatLlmSummary(allocator, summary));
}
