const std = @import("std");
const llm_samples = @import("llm_samples.zig");

const sample_summary = llm_samples.summary();
const sample_route = llm_samples.routeComparison();
const sample_generated = llm_samples.generatedTokens();

pub fn parseJsonObject(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, input, .{});
}

pub fn expectContainsAll(haystack: []const u8, needles: []const []const u8) !void {
    for (needles) |needle| {
        try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
    }
}

fn expectPreferredPromptContextRouteComparison(
    route_comparison: std.json.ObjectMap,
) !void {
    const gains = sample_route.gains;
    const route_comparison_preferred = route_comparison.get("preferred").?.object;
    const route_comparison_gains = route_comparison.get("gains").?.object;
    try std.testing.expectEqualStrings(llm_samples.preferred_route_name, route_comparison_preferred.get("route").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(gains.raw_to_conditioned_milli)), route_comparison_gains.get("raw_to_conditioned_milli").?.integer);
}

pub fn expectSummaryContainsCore(output: []const u8) !void {
    try expectContainsAll(output, &.{
        std.fmt.comptimePrint("\"prompt_tail_token_text\": \"{s}\"", .{llm_samples.bolt_token_text}),
        std.fmt.comptimePrint("\"raw_next_score_milli\": {d}", .{sample_summary.raw_next_score_milli}),
        std.fmt.comptimePrint("\"conditioned_next_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"model_next_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"prompt_context_next_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"preferred_route\": \"{s}\"", .{llm_samples.preferred_route_name}),
        std.fmt.comptimePrint("\"preferred_next_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"preferred_next_score_milli\": {d}", .{sample_summary.preferred_next_score_milli}),
        std.fmt.comptimePrint("\"generated_projection_milli\": {d}", .{sample_summary.generated_projection_milli}),
        std.fmt.comptimePrint("\"generated_context_bias_milli\": {d}", .{sample_summary.generated_context_bias_milli}),
        std.fmt.comptimePrint("\"prompt_context_over_raw_gain_milli\": {d}", .{sample_summary.prompt_context_over_raw_gain_milli}),
        std.fmt.comptimePrint("\"prompt_context_over_conditioned_gain_milli\": {d}", .{sample_summary.prompt_context_over_conditioned_gain_milli}),
        std.fmt.comptimePrint("\"prompt_context_over_model_gain_milli\": {d}", .{sample_summary.prompt_context_over_model_gain_milli}),
        "\"route_comparison\"",
        std.fmt.comptimePrint("\"prompt_context_over_model_milli\": {d}", .{sample_route.gains.prompt_context_over_model_milli}),
        std.fmt.comptimePrint("\"weighted_logit_sum_milli\": {d}", .{sample_summary.weighted_logit_sum_milli}),
        std.fmt.comptimePrint("\"prompt_condition_bias_milli\": {d}", .{sample_summary.prompt_condition_bias_milli}),
        std.fmt.comptimePrint("\"prompt_context_bias_milli\": {d}", .{sample_summary.prompt_context_bias_milli}),
        std.fmt.comptimePrint("\"conditioned_next_score_milli\": {d}", .{sample_summary.conditioned_next_score_milli}),
        std.fmt.comptimePrint("\"model_next_score_milli\": {d}", .{sample_summary.model_next_score_milli}),
        std.fmt.comptimePrint("\"prompt_context_next_score_milli\": {d}", .{sample_summary.prompt_context_next_score_milli}),
        "\"raw_conditioning_flipped\": true",
        "\"conditioned_matches_model\": true",
        std.fmt.comptimePrint("\"model_condition_gap_milli\": {d}", .{sample_summary.model_condition_gap_milli}),
    });
}

pub fn expectSummaryRouteComparison(root: std.json.ObjectMap) !void {
    const summary = sample_summary;
    const prompt_context = summary.route_comparison.prompt_context;
    const gains = summary.route_comparison.gains;
    try std.testing.expectEqualStrings(llm_samples.preferred_route_name, root.get("preferred_route").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(summary.preferred_next_score_milli)), root.get("preferred_next_score_milli").?.integer);
    const route_comparison = root.get("route_comparison").?.object;
    const route_comparison_prompt_context = route_comparison.get("prompt_context").?.object;
    const route_comparison_gains = route_comparison.get("gains").?.object;
    try expectPreferredPromptContextRouteComparison(route_comparison);
    try std.testing.expectEqual(@as(i64, @intCast(prompt_context.projection_milli.?)), route_comparison_prompt_context.get("projection_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(gains.prompt_context_over_model_milli)), route_comparison_gains.get("prompt_context_over_model_milli").?.integer);
}

pub fn expectSummaryReportSemanticRoot(root: std.json.ObjectMap) !void {
    const summary = sample_summary;
    try std.testing.expectEqualStrings(llm_samples.family_name, root.get("family").?.string);
    try std.testing.expectEqualStrings(llm_samples.bolt_token_text, root.get("prompt_tail_token_text").?.string);
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, root.get("conditioned_next_token_text").?.string);
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, root.get("model_next_token_text").?.string);
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, root.get("prompt_context_next_token_text").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(summary.raw_next_score_milli)), root.get("raw_next_score_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.generated_projection_milli)), root.get("generated_projection_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.generated_context_bias_milli)), root.get("generated_context_bias_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.prompt_context_over_raw_gain_milli)), root.get("prompt_context_over_raw_gain_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.prompt_context_over_conditioned_gain_milli)), root.get("prompt_context_over_conditioned_gain_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.prompt_context_over_model_gain_milli)), root.get("prompt_context_over_model_gain_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.weighted_logit_sum_milli)), root.get("weighted_logit_sum_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.prompt_condition_bias_milli)), root.get("prompt_condition_bias_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.prompt_context_bias_milli)), root.get("prompt_context_bias_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.conditioned_next_score_milli)), root.get("conditioned_next_score_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.model_next_score_milli)), root.get("model_next_score_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(summary.prompt_context_next_score_milli)), root.get("prompt_context_next_score_milli").?.integer);
    try std.testing.expect(root.get("raw_conditioning_flipped").?.bool);
    try std.testing.expect(root.get("conditioned_matches_model").?.bool);
    try std.testing.expectEqual(@as(i64, @intCast(summary.model_condition_gap_milli)), root.get("model_condition_gap_milli").?.integer);
    try expectSummaryRouteComparison(root);
}

pub fn expectDebugRouteComparison(trace_root: std.json.ObjectMap) !void {
    const route = sample_route;
    try std.testing.expectEqual(@as(i64, @intCast(route.prompt_context.score_milli)), trace_root.get("prompt_context_top_score_milli").?.integer);
    try std.testing.expectEqualStrings(llm_samples.preferred_route_name, trace_root.get("preferred_route").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(route.gains.conditioned_to_model_milli)), trace_root.get("conditioned_to_model_gain_milli").?.integer);
    const route_comparison = trace_root.get("route_comparison").?.object;
    const route_comparison_prompt_context = route_comparison.get("prompt_context").?.object;
    try expectPreferredPromptContextRouteComparison(route_comparison);
    try std.testing.expectEqual(@as(i64, @intCast(route.prompt_context.score_milli)), route_comparison_prompt_context.get("score_milli").?.integer);
}

pub fn expectDebugReportSemanticRoot(root: std.json.ObjectMap) !void {
    const summary = sample_summary;
    var candidates = llm_samples.debugCandidatesSemantic();
    const trace = llm_samples.debugTrace(candidates[0..]);
    const generated = sample_generated;
    const manifest_root = root.get("manifest").?.object;
    const runtime_root = root.get("runtime").?.object;
    try std.testing.expectEqualStrings(llm_samples.family_name, root.get("family").?.string);
    try std.testing.expectEqualStrings(llm_samples.runtime_bundle_file_name, manifest_root.get("runtime_bundle_file").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(llm_samples.transition_bias_value_count)), runtime_root.get("transition_bias_value_count").?.integer);

    const trace_root = root.get("trace").?.object;
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, trace_root.get("conditioned_top_token_text").?.string);
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, trace_root.get("model_top_token_text").?.string);
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, trace_root.get("prompt_context_top_token_text").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(trace.raw_top_logit_milli)), trace_root.get("raw_top_logit_milli").?.integer);
    try expectDebugRouteComparison(trace_root);
    try std.testing.expectEqual(@as(i64, @intCast(summary.preferred_next_score_milli)), trace_root.get("preferred_top_score_milli").?.integer);
    const trace_candidates = trace_root.get("candidates").?.array.items;
    const trace_candidate1 = trace_candidates[1].object;
    try std.testing.expectEqual(@as(usize, candidates.len), trace_candidates.len);
    try std.testing.expectEqual(@as(i64, @intCast(candidates[1].conditioned_score_milli)), trace_candidate1.get("conditioned_score_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(candidates[1].model_score_milli)), trace_candidate1.get("model_score_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(candidates[1].prompt_context_bias_milli)), trace_candidate1.get("prompt_context_bias_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(candidates[1].prompt_context_score_milli)), trace_candidate1.get("prompt_context_score_milli").?.integer);

    const decode_root = root.get("decode").?.object;
    const decode_generated = decode_root.get("generated").?.array.items;
    const decode_generated0 = decode_generated[0].object;
    const decode_generated1 = decode_generated[1].object;
    try std.testing.expectEqual(@as(i64, @intCast(generated.len)), decode_root.get("step_count").?.integer);
    try std.testing.expectEqualStrings(llm_samples.bolt_token_text, decode_generated0.get("source_token_text").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(generated[0].projection_milli)), decode_generated0.get("projection_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(generated[0].context_bias_milli)), decode_generated0.get("context_bias_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(generated[0].conditioned_score_milli)), decode_generated0.get("conditioned_score_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(generated[1].source_context_token_count)), decode_generated1.get("source_context_token_count").?.integer);
    try std.testing.expectEqualStrings(llm_samples.zig_token_text, decode_generated1.get("token_text").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(generated[1].context_bias_milli)), decode_generated1.get("context_bias_milli").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(generated[1].conditioned_score_milli)), decode_generated1.get("conditioned_score_milli").?.integer);
}

pub fn expectDebugReportContainsCore(output: []const u8) !void {
    const sample_generated0 = sample_generated[0];
    const sample_generated1 = sample_generated[1];
    try expectContainsAll(output, &.{
        std.fmt.comptimePrint("\"runtime_bundle_file\": \"{s}\"", .{llm_samples.runtime_bundle_file_name}),
        std.fmt.comptimePrint("\"transition_bias_value_count\": {d}", .{llm_samples.transition_bias_value_count}),
        std.fmt.comptimePrint("\"conditioned_top_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"model_top_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"prompt_context_top_token_text\": \"{s}\"", .{llm_samples.zig_token_text}),
        std.fmt.comptimePrint("\"raw_top_logit_milli\": {d}", .{sample_summary.raw_next_score_milli}),
        std.fmt.comptimePrint("\"prompt_context_top_score_milli\": {d}", .{sample_route.prompt_context.score_milli}),
        std.fmt.comptimePrint("\"preferred_route\": \"{s}\"", .{llm_samples.preferred_route_name}),
        "\"route_comparison\"",
        std.fmt.comptimePrint("\"raw_to_conditioned_milli\": {d}", .{sample_route.gains.raw_to_conditioned_milli}),
        std.fmt.comptimePrint("\"conditioned_to_model_gain_milli\": {d}", .{sample_route.gains.conditioned_to_model_milli}),
        std.fmt.comptimePrint("\"step_count\": {d}", .{sample_generated.len}),
        std.fmt.comptimePrint("\"source_context_token_count\": {d}", .{sample_generated1.source_context_token_count}),
        std.fmt.comptimePrint("\"source_token_text\": \"{s}\"", .{llm_samples.bolt_token_text}),
        std.fmt.comptimePrint("\"prompt_context_bias_milli\": {d}", .{sample_generated0.context_bias_milli}),
        std.fmt.comptimePrint("\"projection_milli\": {d}", .{sample_generated0.projection_milli}),
        std.fmt.comptimePrint("\"context_bias_milli\": {d}", .{sample_generated1.context_bias_milli}),
        std.fmt.comptimePrint("\"prompt_context_score_milli\": {d}", .{sample_route.prompt_context.score_milli}),
    });
}
