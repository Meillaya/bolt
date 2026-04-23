const std = @import("std");
const tokenizer = @import("../tokenizer.zig");
const weights = @import("../weights.zig");
const llm_samples = @import("../testing/llm_samples.zig");

pub const fixture_family = "llm";
pub const preferred_route_label = "prompt_context";

pub const DecoderModel = struct {
    vocab_size: usize,
    hidden_size: usize = 4,
};

pub const DecoderFixturePayload = struct {
    family: []const u8,
    fixture_name: []const u8,
    prompt_text: []const u8,
    token_ids: []const usize,
    logits: []const f32,
    expected_prompt_sum: usize,
    expected_top_token: usize,
};

pub const DecoderRouteWinner = struct {
    route: []const u8,
    token_id: usize,
    token_text: []const u8,
    score_milli: usize,
    projection_milli: ?usize = null,
    context_bias_milli: ?usize = null,
};

pub const DecoderRouteGains = struct {
    raw_to_conditioned_milli: usize,
    conditioned_to_model_milli: usize,
    model_to_prompt_context_milli: usize,
    prompt_context_over_raw_milli: usize,
    prompt_context_over_conditioned_milli: usize,
    prompt_context_over_model_milli: usize,
};

pub const DecoderRouteComparison = struct {
    raw: DecoderRouteWinner,
    conditioned: DecoderRouteWinner,
    model: DecoderRouteWinner,
    prompt_context: DecoderRouteWinner,
    preferred: DecoderRouteWinner,
    gains: DecoderRouteGains,
};

pub const DecoderFixtureSummary = struct {
    family: []const u8,
    fixture_name: []const u8,
    prompt_token_count: usize,
    prompt_sum: usize,
    logits_count: usize,
    next_token_id: usize,
    generated_token_count: usize,
    generated_last_token: usize,
    prompt_tail_token_id: usize,
    prompt_tail_token_text: []const u8,
    next_token_text: []const u8,
    raw_next_score_milli: usize,
    conditioned_next_token_id: usize,
    conditioned_next_token_text: []const u8,
    model_next_token_id: usize,
    model_next_token_text: []const u8,
    prompt_context_next_token_id: usize,
    prompt_context_next_token_text: []const u8,
    preferred_route: []const u8,
    preferred_next_token_id: usize,
    preferred_next_token_text: []const u8,
    preferred_next_score_milli: usize,
    route_comparison: DecoderRouteComparison,
    generated_projection_milli: usize,
    generated_context_bias_milli: usize,
    prompt_context_over_raw_gain_milli: usize,
    prompt_context_over_conditioned_gain_milli: usize,
    prompt_context_over_model_gain_milli: usize,
    raw_conditioning_flipped: bool,
    conditioned_matches_model: bool,
    top_token_weight_milli: usize,
    weighted_logit_sum_milli: usize,
    prompt_condition_bias_milli: usize,
    prompt_context_bias_milli: usize,
    conditioned_next_score_milli: usize,
    model_next_score_milli: usize,
    prompt_context_next_score_milli: usize,
    model_condition_gap_milli: usize,
    logit_margin_milli: usize,
};

pub const DecoderExpectedSummary = DecoderFixtureSummary;

pub const DecoderCandidateScore = struct {
    token_id: usize,
    token_text: []const u8,
    raw_logit_milli: usize,
    output_projection_milli: usize,
    transition_bias_milli: usize,
    conditioned_score_milli: usize,
    model_score_milli: usize,
    prompt_context_bias_milli: usize,
    prompt_context_score_milli: usize,
};

pub const DecoderFixtureTrace = struct {
    family: []const u8,
    fixture_name: []const u8,
    prompt_text: []const u8,
    prompt_tail_token_id: usize,
    prompt_tail_token_text: []const u8,
    raw_top_token_id: usize,
    raw_top_token_text: []const u8,
    raw_top_logit_milli: usize,
    conditioned_top_token_id: usize,
    conditioned_top_token_text: []const u8,
    conditioned_top_score_milli: usize,
    model_top_token_id: usize,
    model_top_token_text: []const u8,
    model_top_score_milli: usize,
    prompt_context_top_token_id: usize,
    prompt_context_top_token_text: []const u8,
    prompt_context_top_projection_milli: usize,
    prompt_context_top_bias_milli: usize,
    prompt_context_top_score_milli: usize,
    preferred_route: []const u8,
    preferred_top_token_id: usize,
    preferred_top_token_text: []const u8,
    preferred_top_score_milli: usize,
    raw_to_conditioned_gain_milli: usize,
    conditioned_to_model_gain_milli: usize,
    model_to_prompt_context_gain_milli: usize,
    route_comparison: DecoderRouteComparison,
    candidates: []DecoderCandidateScore,
};

pub const DecoderGeneratedToken = struct {
    step_index: usize,
    source_context_token_count: usize,
    source_token_id: usize,
    source_token_text: []const u8,
    token_id: usize,
    token_text: []const u8,
    projection_milli: usize,
    context_bias_milli: usize,
    conditioned_score_milli: usize,
};

pub const DecoderFixtureDecode = struct {
    family: []const u8,
    fixture_name: []const u8,
    prompt_text: []const u8,
    prompt_token_ids: []const usize,
    generated: []DecoderGeneratedToken,
};

pub fn parseFixturePayloadFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(DecoderFixturePayload) {
    return std.json.parseFromSlice(
        DecoderFixturePayload,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadFixturePayloadFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(DecoderFixturePayload) {
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
    payload: DecoderFixturePayload,
) !DecoderFixtureSummary {
    return summarizeFixtureWithRuntime(
        payload,
        tokenizer.defaultTokenizer(),
        weights.defaultWeights(),
    );
}

pub fn summarizeFixtureWithRuntime(
    payload: DecoderFixturePayload,
    token_decoder: tokenizer.FixtureTokenizer,
    projection: weights.FixtureDecoderWeights,
) !DecoderFixtureSummary {
    const model = DecoderModel{
        .vocab_size = projection.vocabSize(),
    };
    if (!std.mem.eql(u8, payload.family, fixture_family)) {
        return error.UnexpectedFamily;
    }
    if (payload.fixture_name.len == 0) {
        return error.EmptyFixtureName;
    }
    if (payload.logits.len != model.vocab_size) {
        return error.UnexpectedLogitsSize;
    }
    if (payload.prompt_text.len == 0) {
        return error.EmptyPromptText;
    }
    if (payload.token_ids.len == 0) {
        return error.EmptyPrompt;
    }

    const encoded_prompt = try token_decoder.encodeText(std.heap.page_allocator, payload.prompt_text);
    defer std.heap.page_allocator.free(encoded_prompt);
    if (!std.mem.eql(usize, encoded_prompt, payload.token_ids)) {
        return error.PromptTokenizationMismatch;
    }

    var prompt_sum: usize = 0;
    for (payload.token_ids) |token_id| {
        _ = try token_decoder.decodeToken(token_id);
        prompt_sum += token_id;
    }
    if (prompt_sum != payload.expected_prompt_sum) {
        return error.UnexpectedPromptSum;
    }

    var next_token_id: usize = 0;
    var top_logit = payload.logits[0];
    var second_logit: f32 = -std.math.inf(f32);
    for (payload.logits[1..], 1..) |logit, index| {
        if (logit > top_logit) {
            second_logit = top_logit;
            top_logit = logit;
            next_token_id = index;
        } else if (logit > second_logit) {
            second_logit = logit;
        }
    }
    if (next_token_id != payload.expected_top_token) {
        return error.UnexpectedTopToken;
    }

    const generated_token_count = payload.token_ids.len + 1;
    const prompt_tail_token_id = payload.token_ids[payload.token_ids.len - 1];
    const prompt_tail_token_text = try token_decoder.decodeToken(prompt_tail_token_id);
    const next_token_text = try token_decoder.decodeToken(next_token_id);
    const conditioned_top = try projection.conditionedTopToken(payload.logits, prompt_tail_token_id);
    const conditioned_next_token_id = conditioned_top.token_id;
    const conditioned_next_token_text = try token_decoder.decodeToken(conditioned_next_token_id);
    const model_top = try projection.modelTopToken(prompt_tail_token_id);
    const model_next_token_id = model_top.token_id;
    const model_next_token_text = try token_decoder.decodeToken(model_next_token_id);
    const prompt_context_top = try projection.promptContextTopToken(payload.token_ids);
    const prompt_context_next_token_id = prompt_context_top.token_id;
    const prompt_context_next_token_text = try token_decoder.decodeToken(prompt_context_next_token_id);
    const generated_last_token = prompt_context_next_token_id;
    const generated_projection_milli = try projection.topTokenWeightMilli(prompt_context_next_token_id);
    const raw_conditioning_flipped = next_token_id != conditioned_next_token_id;
    const conditioned_matches_model = conditioned_next_token_id == model_next_token_id;
    const top_token_weight_milli = try projection.topTokenWeightMilli(next_token_id);
    const weighted_logit_sum_milli = try projection.weightedLogitSumMilli(payload.logits);
    const prompt_condition_bias_milli = try projection.transitionBiasMilli(
        prompt_tail_token_id,
        conditioned_next_token_id,
    );
    const prompt_context_bias_milli = try projection.promptContextBiasMilli(
        payload.token_ids,
        prompt_context_next_token_id,
    );
    const conditioned_next_score_milli = conditioned_top.score_milli;
    const model_next_score_milli = model_top.score_milli;
    const prompt_context_next_score_milli = prompt_context_top.score_milli;
    const raw_top_logit_milli = @as(usize, @intFromFloat((payload.logits[next_token_id] * 1000.0) + 0.5));
    const prompt_context_over_raw_gain_milli = prompt_context_next_score_milli - raw_top_logit_milli;
    const prompt_context_over_conditioned_gain_milli = if (prompt_context_next_score_milli >= conditioned_next_score_milli)
        prompt_context_next_score_milli - conditioned_next_score_milli
    else
        conditioned_next_score_milli - prompt_context_next_score_milli;
    const prompt_context_over_model_gain_milli = if (prompt_context_next_score_milli >= model_next_score_milli)
        prompt_context_next_score_milli - model_next_score_milli
    else
        model_next_score_milli - prompt_context_next_score_milli;
    const model_condition_gap_milli = if (model_next_score_milli >= conditioned_next_score_milli)
        model_next_score_milli - conditioned_next_score_milli
    else
        conditioned_next_score_milli - model_next_score_milli;
    const margin = if (second_logit == -std.math.inf(f32)) top_logit else top_logit - second_logit;
    const logit_margin_milli = @as(usize, @intFromFloat((margin * 1000.0) + 0.5));
    const route_comparison: DecoderRouteComparison = .{
        .raw = .{
            .route = "raw",
            .token_id = next_token_id,
            .token_text = next_token_text,
            .score_milli = raw_top_logit_milli,
        },
        .conditioned = .{
            .route = "conditioned",
            .token_id = conditioned_next_token_id,
            .token_text = conditioned_next_token_text,
            .score_milli = conditioned_next_score_milli,
        },
        .model = .{
            .route = "model",
            .token_id = model_next_token_id,
            .token_text = model_next_token_text,
            .score_milli = model_next_score_milli,
        },
        .prompt_context = .{
            .route = preferred_route_label,
            .token_id = prompt_context_next_token_id,
            .token_text = prompt_context_next_token_text,
            .score_milli = prompt_context_next_score_milli,
            .projection_milli = generated_projection_milli,
            .context_bias_milli = prompt_context_bias_milli,
        },
        .preferred = .{
            .route = preferred_route_label,
            .token_id = prompt_context_next_token_id,
            .token_text = prompt_context_next_token_text,
            .score_milli = prompt_context_next_score_milli,
            .projection_milli = generated_projection_milli,
            .context_bias_milli = prompt_context_bias_milli,
        },
        .gains = .{
            .raw_to_conditioned_milli = if (conditioned_next_score_milli >= raw_top_logit_milli)
                conditioned_next_score_milli - raw_top_logit_milli
            else
                raw_top_logit_milli - conditioned_next_score_milli,
            .conditioned_to_model_milli = model_condition_gap_milli,
            .model_to_prompt_context_milli = prompt_context_over_model_gain_milli,
            .prompt_context_over_raw_milli = prompt_context_over_raw_gain_milli,
            .prompt_context_over_conditioned_milli = prompt_context_over_conditioned_gain_milli,
            .prompt_context_over_model_milli = prompt_context_over_model_gain_milli,
        },
    };
    const raw_route = route_comparison.raw;
    const conditioned_route = route_comparison.conditioned;
    const model_route = route_comparison.model;
    const prompt_context_route = route_comparison.prompt_context;
    const preferred_route = route_comparison.preferred;
    const route_gains = route_comparison.gains;

    const summary: DecoderFixtureSummary = .{
        .family = payload.family,
        .fixture_name = payload.fixture_name,
        .prompt_token_count = payload.token_ids.len,
        .prompt_sum = prompt_sum,
        .logits_count = payload.logits.len,
        .next_token_id = raw_route.token_id,
        .generated_token_count = generated_token_count,
        .generated_last_token = generated_last_token,
        .prompt_tail_token_id = prompt_tail_token_id,
        .prompt_tail_token_text = prompt_tail_token_text,
        .next_token_text = raw_route.token_text,
        .raw_next_score_milli = raw_route.score_milli,
        .conditioned_next_token_id = conditioned_route.token_id,
        .conditioned_next_token_text = conditioned_route.token_text,
        .model_next_token_id = model_route.token_id,
        .model_next_token_text = model_route.token_text,
        .prompt_context_next_token_id = prompt_context_route.token_id,
        .prompt_context_next_token_text = prompt_context_route.token_text,
        .preferred_route = preferred_route.route,
        .preferred_next_token_id = preferred_route.token_id,
        .preferred_next_token_text = preferred_route.token_text,
        .preferred_next_score_milli = preferred_route.score_milli,
        .route_comparison = route_comparison,
        .generated_projection_milli = prompt_context_route.projection_milli.?,
        .generated_context_bias_milli = prompt_context_route.context_bias_milli.?,
        .prompt_context_over_raw_gain_milli = route_gains.prompt_context_over_raw_milli,
        .prompt_context_over_conditioned_gain_milli = route_gains.prompt_context_over_conditioned_milli,
        .prompt_context_over_model_gain_milli = route_gains.prompt_context_over_model_milli,
        .raw_conditioning_flipped = raw_conditioning_flipped,
        .conditioned_matches_model = conditioned_matches_model,
        .top_token_weight_milli = top_token_weight_milli,
        .weighted_logit_sum_milli = weighted_logit_sum_milli,
        .prompt_condition_bias_milli = prompt_condition_bias_milli,
        .prompt_context_bias_milli = prompt_context_route.context_bias_milli.?,
        .conditioned_next_score_milli = conditioned_route.score_milli,
        .model_next_score_milli = model_route.score_milli,
        .prompt_context_next_score_milli = prompt_context_route.score_milli,
        .model_condition_gap_milli = model_condition_gap_milli,
        .logit_margin_milli = logit_margin_milli,
    };

    try validateSummaryConsistency(summary);
    return summary;
}

pub fn validateSummaryConsistency(summary: DecoderFixtureSummary) !void {
    const route_comparison = summary.route_comparison;
    const raw = route_comparison.raw;
    const conditioned = route_comparison.conditioned;
    const model = route_comparison.model;
    const prompt_context = route_comparison.prompt_context;
    const preferred = route_comparison.preferred;
    const gains = route_comparison.gains;
    if (!std.mem.eql(u8, summary.preferred_route, preferred_route_label)) return error.UnexpectedPreferredRoute;
    if (!std.mem.eql(u8, prompt_context.route, preferred_route_label)) return error.PromptContextRouteMismatch;
    if (!std.mem.eql(u8, preferred.route, preferred_route_label)) return error.RouteComparisonPreferredRouteMismatch;
    if (summary.generated_last_token != summary.prompt_context_next_token_id) return error.GeneratedLastTokenMismatch;
    if (summary.generated_last_token != summary.preferred_next_token_id) return error.GeneratedLastTokenPreferredMismatch;
    if (summary.prompt_context_next_token_id != prompt_context.token_id) return error.PromptContextTokenRouteComparisonMismatch;
    if (summary.preferred_next_token_id != preferred.token_id) return error.PreferredNextTokenRouteComparisonMismatch;
    if (!std.mem.eql(u8, summary.prompt_context_next_token_text, prompt_context.token_text)) return error.PromptContextTokenTextRouteComparisonMismatch;
    if (!std.mem.eql(u8, summary.preferred_next_token_text, preferred.token_text)) return error.PreferredNextTokenTextRouteComparisonMismatch;
    if (summary.prompt_context_next_score_milli != prompt_context.score_milli) return error.PromptContextScoreRouteComparisonMismatch;
    if (summary.preferred_next_score_milli != preferred.score_milli) return error.PreferredNextScoreRouteComparisonMismatch;
    if (summary.raw_next_score_milli != raw.score_milli) return error.RawScoreRouteComparisonMismatch;
    if (summary.conditioned_next_score_milli != conditioned.score_milli) return error.ConditionedScoreRouteComparisonMismatch;
    if (summary.model_next_score_milli != model.score_milli) return error.ModelScoreRouteComparisonMismatch;

    const prompt_context_projection = prompt_context.projection_milli orelse return error.MissingPromptContextProjection;
    const preferred_projection = preferred.projection_milli orelse return error.MissingPreferredProjection;
    if (summary.generated_projection_milli != prompt_context_projection) return error.GeneratedProjectionRouteComparisonMismatch;
    if (summary.generated_projection_milli != preferred_projection) return error.GeneratedProjectionPreferredMismatch;

    const prompt_context_bias = prompt_context.context_bias_milli orelse return error.MissingPromptContextBias;
    const preferred_bias = preferred.context_bias_milli orelse return error.MissingPreferredBias;
    if (summary.generated_context_bias_milli != prompt_context_bias) return error.GeneratedContextBiasRouteComparisonMismatch;
    if (summary.generated_context_bias_milli != preferred_bias) return error.GeneratedContextBiasPreferredMismatch;

    if (summary.prompt_context_over_raw_gain_milli != gains.prompt_context_over_raw_milli) return error.PromptContextRawGainRouteComparisonMismatch;
    if (summary.prompt_context_over_conditioned_gain_milli != gains.prompt_context_over_conditioned_milli) return error.PromptContextConditionedGainRouteComparisonMismatch;
    if (summary.prompt_context_over_model_gain_milli != gains.prompt_context_over_model_milli) return error.PromptContextModelGainRouteComparisonMismatch;
    if (summary.model_condition_gap_milli != gains.conditioned_to_model_milli) return error.ModelConditionGapRouteComparisonMismatch;
    if (gains.model_to_prompt_context_milli != gains.prompt_context_over_model_milli) return error.ModelPromptContextGainMismatch;
}

pub fn parseExpectedSummaryFromSlice(
    allocator: std.mem.Allocator,
    input: []const u8,
) !std.json.Parsed(DecoderExpectedSummary) {
    return std.json.parseFromSlice(
        DecoderExpectedSummary,
        allocator,
        input,
        .{ .allocate = .alloc_always },
    );
}

pub fn loadExpectedSummaryFromFile(
    io: std.Io,
    allocator: std.mem.Allocator,
    path: []const u8,
) !std.json.Parsed(DecoderExpectedSummary) {
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
    summary: DecoderFixtureSummary,
    expected: DecoderExpectedSummary,
) !void {
    try validateSummaryConsistency(summary);
    try validateSummaryConsistency(expected);
    const summary_routes = summary.route_comparison;
    const expected_routes = expected.route_comparison;
    const summary_raw = summary_routes.raw;
    const expected_raw = expected_routes.raw;
    const summary_conditioned = summary_routes.conditioned;
    const expected_conditioned = expected_routes.conditioned;
    const summary_model = summary_routes.model;
    const expected_model = expected_routes.model;
    const summary_prompt_context = summary_routes.prompt_context;
    const expected_prompt_context = expected_routes.prompt_context;
    const summary_preferred = summary_routes.preferred;
    const expected_preferred = expected_routes.preferred;
    const summary_gains = summary_routes.gains;
    const expected_gains = expected_routes.gains;
    if (!std.mem.eql(u8, summary.family, expected.family)) {
        return error.FamilyMismatch;
    }
    if (!std.mem.eql(u8, summary.fixture_name, expected.fixture_name)) {
        return error.FixtureNameMismatch;
    }
    if (summary.prompt_token_count != expected.prompt_token_count) return error.TokenCountMismatch;
    if (summary.prompt_sum != expected.prompt_sum) return error.PromptSumMismatch;
    if (summary.logits_count != expected.logits_count) return error.LogitsCountMismatch;
    if (summary.next_token_id != expected.next_token_id) return error.NextTokenMismatch;
    if (summary.generated_token_count != expected.generated_token_count) return error.GeneratedTokenCountMismatch;
    if (summary.generated_last_token != expected.generated_last_token) return error.GeneratedLastTokenMismatch;
    if (summary.prompt_tail_token_id != expected.prompt_tail_token_id) return error.PromptTailTokenMismatch;
    if (!std.mem.eql(u8, summary.prompt_tail_token_text, expected.prompt_tail_token_text)) return error.PromptTailTokenTextMismatch;
    if (!std.mem.eql(u8, summary.next_token_text, expected.next_token_text)) return error.NextTokenTextMismatch;
    if (summary.raw_next_score_milli != expected.raw_next_score_milli) return error.RawNextScoreMismatch;
    if (summary.conditioned_next_token_id != expected.conditioned_next_token_id) return error.ConditionedNextTokenMismatch;
    if (!std.mem.eql(u8, summary.conditioned_next_token_text, expected.conditioned_next_token_text)) return error.ConditionedNextTokenTextMismatch;
    if (summary.model_next_token_id != expected.model_next_token_id) return error.ModelNextTokenMismatch;
    if (!std.mem.eql(u8, summary.model_next_token_text, expected.model_next_token_text)) return error.ModelNextTokenTextMismatch;
    if (summary.prompt_context_next_token_id != expected.prompt_context_next_token_id) return error.PromptContextNextTokenMismatch;
    if (!std.mem.eql(u8, summary.prompt_context_next_token_text, expected.prompt_context_next_token_text)) return error.PromptContextNextTokenTextMismatch;
    if (!std.mem.eql(u8, summary.preferred_route, expected.preferred_route)) return error.PreferredRouteMismatch;
    if (summary.preferred_next_token_id != expected.preferred_next_token_id) return error.PreferredNextTokenMismatch;
    if (!std.mem.eql(u8, summary.preferred_next_token_text, expected.preferred_next_token_text)) return error.PreferredNextTokenTextMismatch;
    if (summary.preferred_next_score_milli != expected.preferred_next_score_milli) return error.PreferredNextScoreMismatch;
    if (!std.mem.eql(u8, summary_raw.route, expected_raw.route)) return error.RouteComparisonRawRouteMismatch;
    if (summary_raw.token_id != expected_raw.token_id) return error.RouteComparisonRawTokenMismatch;
    if (!std.mem.eql(u8, summary_raw.token_text, expected_raw.token_text)) return error.RouteComparisonRawTokenTextMismatch;
    if (summary_raw.score_milli != expected_raw.score_milli) return error.RouteComparisonRawScoreMismatch;
    if (!std.mem.eql(u8, summary_conditioned.route, expected_conditioned.route)) return error.RouteComparisonConditionedRouteMismatch;
    if (summary_conditioned.token_id != expected_conditioned.token_id) return error.RouteComparisonConditionedTokenMismatch;
    if (!std.mem.eql(u8, summary_conditioned.token_text, expected_conditioned.token_text)) return error.RouteComparisonConditionedTokenTextMismatch;
    if (summary_conditioned.score_milli != expected_conditioned.score_milli) return error.RouteComparisonConditionedScoreMismatch;
    if (!std.mem.eql(u8, summary_model.route, expected_model.route)) return error.RouteComparisonModelRouteMismatch;
    if (summary_model.token_id != expected_model.token_id) return error.RouteComparisonModelTokenMismatch;
    if (!std.mem.eql(u8, summary_model.token_text, expected_model.token_text)) return error.RouteComparisonModelTokenTextMismatch;
    if (summary_model.score_milli != expected_model.score_milli) return error.RouteComparisonModelScoreMismatch;
    if (!std.mem.eql(u8, summary_prompt_context.route, expected_prompt_context.route)) return error.RouteComparisonPromptContextRouteMismatch;
    if (summary_prompt_context.token_id != expected_prompt_context.token_id) return error.RouteComparisonPromptContextTokenMismatch;
    if (!std.mem.eql(u8, summary_prompt_context.token_text, expected_prompt_context.token_text)) return error.RouteComparisonPromptContextTokenTextMismatch;
    if (summary_prompt_context.score_milli != expected_prompt_context.score_milli) return error.RouteComparisonPromptContextScoreMismatch;
    if (summary_prompt_context.projection_milli != expected_prompt_context.projection_milli) return error.RouteComparisonPromptContextProjectionMismatch;
    if (summary_prompt_context.context_bias_milli != expected_prompt_context.context_bias_milli) return error.RouteComparisonPromptContextBiasMismatch;
    if (!std.mem.eql(u8, summary_preferred.route, expected_preferred.route)) return error.RouteComparisonPreferredRouteMismatch;
    if (summary_preferred.token_id != expected_preferred.token_id) return error.RouteComparisonPreferredTokenMismatch;
    if (!std.mem.eql(u8, summary_preferred.token_text, expected_preferred.token_text)) return error.RouteComparisonPreferredTokenTextMismatch;
    if (summary_preferred.score_milli != expected_preferred.score_milli) return error.RouteComparisonPreferredScoreMismatch;
    if (summary_preferred.projection_milli != expected_preferred.projection_milli) return error.RouteComparisonPreferredProjectionMismatch;
    if (summary_preferred.context_bias_milli != expected_preferred.context_bias_milli) return error.RouteComparisonPreferredBiasMismatch;
    if (summary_gains.raw_to_conditioned_milli != expected_gains.raw_to_conditioned_milli) return error.RouteComparisonRawToConditionedGainMismatch;
    if (summary_gains.conditioned_to_model_milli != expected_gains.conditioned_to_model_milli) return error.RouteComparisonConditionedToModelGainMismatch;
    if (summary_gains.model_to_prompt_context_milli != expected_gains.model_to_prompt_context_milli) return error.RouteComparisonModelToPromptContextGainMismatch;
    if (summary_gains.prompt_context_over_raw_milli != expected_gains.prompt_context_over_raw_milli) return error.RouteComparisonRawGainMismatch;
    if (summary_gains.prompt_context_over_conditioned_milli != expected_gains.prompt_context_over_conditioned_milli) return error.RouteComparisonConditionedGainMismatch;
    if (summary_gains.prompt_context_over_model_milli != expected_gains.prompt_context_over_model_milli) return error.RouteComparisonModelGainMismatch;
    if (summary.generated_projection_milli != expected.generated_projection_milli) return error.GeneratedProjectionMismatch;
    if (summary.generated_context_bias_milli != expected.generated_context_bias_milli) return error.GeneratedContextBiasMismatch;
    if (summary.prompt_context_over_raw_gain_milli != expected.prompt_context_over_raw_gain_milli) return error.PromptContextRawGainMismatch;
    if (summary.prompt_context_over_conditioned_gain_milli != expected.prompt_context_over_conditioned_gain_milli) return error.PromptContextConditionedGainMismatch;
    if (summary.prompt_context_over_model_gain_milli != expected.prompt_context_over_model_gain_milli) return error.PromptContextModelGainMismatch;
    if (summary.raw_conditioning_flipped != expected.raw_conditioning_flipped) return error.RawConditioningFlipMismatch;
    if (summary.conditioned_matches_model != expected.conditioned_matches_model) return error.ConditionedModelAgreementMismatch;
    if (summary.top_token_weight_milli != expected.top_token_weight_milli) return error.TopTokenWeightMismatch;
    if (summary.weighted_logit_sum_milli != expected.weighted_logit_sum_milli) return error.WeightedLogitSumMismatch;
    if (summary.prompt_condition_bias_milli != expected.prompt_condition_bias_milli) return error.PromptConditionBiasMismatch;
    if (summary.prompt_context_bias_milli != expected.prompt_context_bias_milli) return error.PromptContextBiasMismatch;
    if (summary.conditioned_next_score_milli != expected.conditioned_next_score_milli) return error.ConditionedNextScoreMismatch;
    if (summary.model_next_score_milli != expected.model_next_score_milli) return error.ModelNextScoreMismatch;
    if (summary.prompt_context_next_score_milli != expected.prompt_context_next_score_milli) return error.PromptContextNextScoreMismatch;
    if (summary.model_condition_gap_milli != expected.model_condition_gap_milli) return error.ModelConditionGapMismatch;
    if (summary.logit_margin_milli != expected.logit_margin_milli) return error.LogitMarginMismatch;
}

pub fn traceFixtureWithRuntime(
    allocator: std.mem.Allocator,
    payload: DecoderFixturePayload,
    token_decoder: tokenizer.FixtureTokenizer,
    projection: weights.FixtureDecoderWeights,
) !DecoderFixtureTrace {
    const summary = try summarizeFixtureWithRuntime(
        payload,
        token_decoder,
        projection,
    );

    const candidates = try allocator.alloc(DecoderCandidateScore, payload.logits.len);
    errdefer allocator.free(candidates);

    for (payload.logits, 0..) |logit, token_id| {
        const token_text = try token_decoder.decodeToken(token_id);
        const raw_logit_milli = @as(usize, @intFromFloat((logit * 1000.0) + 0.5));
        const output_projection_milli = try projection.topTokenWeightMilli(token_id);
        const transition_bias_milli = try projection.transitionBiasMilli(
            summary.prompt_tail_token_id,
            token_id,
        );
        const conditioned_score_milli = raw_logit_milli + transition_bias_milli;
        const model_score_milli = try projection.modelScoreMilli(
            summary.prompt_tail_token_id,
            token_id,
        );
        const prompt_context_score_milli = try projection.promptContextScoreMilli(
            payload.token_ids,
            token_id,
        );
        const prompt_context_bias_milli = try projection.promptContextBiasMilli(
            payload.token_ids,
            token_id,
        );

        candidates[token_id] = .{
            .token_id = token_id,
            .token_text = token_text,
            .raw_logit_milli = raw_logit_milli,
            .output_projection_milli = output_projection_milli,
            .transition_bias_milli = transition_bias_milli,
            .conditioned_score_milli = conditioned_score_milli,
            .model_score_milli = model_score_milli,
            .prompt_context_bias_milli = prompt_context_bias_milli,
            .prompt_context_score_milli = prompt_context_score_milli,
        };
    }

    const raw = summary.route_comparison.raw;
    const conditioned = summary.route_comparison.conditioned;
    const model = summary.route_comparison.model;
    const prompt_context = summary.route_comparison.prompt_context;
    const preferred = summary.route_comparison.preferred;
    const gains = summary.route_comparison.gains;

    const trace: DecoderFixtureTrace = .{
        .family = payload.family,
        .fixture_name = payload.fixture_name,
        .prompt_text = payload.prompt_text,
        .prompt_tail_token_id = summary.prompt_tail_token_id,
        .prompt_tail_token_text = summary.prompt_tail_token_text,
        .raw_top_token_id = raw.token_id,
        .raw_top_token_text = raw.token_text,
        .raw_top_logit_milli = raw.score_milli,
        .conditioned_top_token_id = conditioned.token_id,
        .conditioned_top_token_text = conditioned.token_text,
        .conditioned_top_score_milli = conditioned.score_milli,
        .model_top_token_id = model.token_id,
        .model_top_token_text = model.token_text,
        .model_top_score_milli = model.score_milli,
        .prompt_context_top_token_id = prompt_context.token_id,
        .prompt_context_top_token_text = prompt_context.token_text,
        .prompt_context_top_projection_milli = prompt_context.projection_milli.?,
        .prompt_context_top_bias_milli = prompt_context.context_bias_milli.?,
        .prompt_context_top_score_milli = prompt_context.score_milli,
        .preferred_route = preferred.route,
        .preferred_top_token_id = preferred.token_id,
        .preferred_top_token_text = preferred.token_text,
        .preferred_top_score_milli = preferred.score_milli,
        .raw_to_conditioned_gain_milli = gains.raw_to_conditioned_milli,
        .conditioned_to_model_gain_milli = gains.conditioned_to_model_milli,
        .model_to_prompt_context_gain_milli = gains.model_to_prompt_context_milli,
        .route_comparison = summary.route_comparison,
        .candidates = candidates,
    };

    try validateTraceConsistency(trace);
    return trace;
}

pub fn traceFixture(
    allocator: std.mem.Allocator,
    payload: DecoderFixturePayload,
) !DecoderFixtureTrace {
    return traceFixtureWithRuntime(
        allocator,
        payload,
        tokenizer.defaultTokenizer(),
        weights.defaultWeights(),
    );
}

pub fn validateTraceConsistency(trace: DecoderFixtureTrace) !void {
    const route_comparison = trace.route_comparison;
    const raw = route_comparison.raw;
    const conditioned = route_comparison.conditioned;
    const model = route_comparison.model;
    const prompt_context = route_comparison.prompt_context;
    const preferred = route_comparison.preferred;
    const gains = route_comparison.gains;
    if (!std.mem.eql(u8, trace.preferred_route, preferred_route_label)) return error.UnexpectedPreferredTraceRoute;
    if (!std.mem.eql(u8, preferred.route, preferred_route_label)) return error.TracePreferredRouteComparisonMismatch;
    if (trace.raw_top_token_id != raw.token_id) return error.TraceRawTokenRouteComparisonMismatch;
    if (!std.mem.eql(u8, trace.raw_top_token_text, raw.token_text)) return error.TraceRawTokenTextRouteComparisonMismatch;
    if (trace.raw_top_logit_milli != raw.score_milli) return error.TraceRawScoreRouteComparisonMismatch;
    if (trace.conditioned_top_token_id != conditioned.token_id) return error.TraceConditionedTokenRouteComparisonMismatch;
    if (!std.mem.eql(u8, trace.conditioned_top_token_text, conditioned.token_text)) return error.TraceConditionedTokenTextRouteComparisonMismatch;
    if (trace.conditioned_top_score_milli != conditioned.score_milli) return error.TraceConditionedScoreRouteComparisonMismatch;
    if (trace.model_top_token_id != model.token_id) return error.TraceModelTokenRouteComparisonMismatch;
    if (!std.mem.eql(u8, trace.model_top_token_text, model.token_text)) return error.TraceModelTokenTextRouteComparisonMismatch;
    if (trace.model_top_score_milli != model.score_milli) return error.TraceModelScoreRouteComparisonMismatch;
    if (trace.prompt_context_top_token_id != prompt_context.token_id) return error.TracePromptContextTokenRouteComparisonMismatch;
    if (!std.mem.eql(u8, trace.prompt_context_top_token_text, prompt_context.token_text)) return error.TracePromptContextTokenTextRouteComparisonMismatch;
    if (trace.prompt_context_top_score_milli != prompt_context.score_milli) return error.TracePromptContextScoreRouteComparisonMismatch;
    if (trace.preferred_top_token_id != preferred.token_id) return error.TracePreferredTokenRouteComparisonMismatch;
    if (!std.mem.eql(u8, trace.preferred_top_token_text, preferred.token_text)) return error.TracePreferredTokenTextRouteComparisonMismatch;
    if (trace.preferred_top_score_milli != preferred.score_milli) return error.TracePreferredScoreRouteComparisonMismatch;

    const prompt_context_projection = prompt_context.projection_milli orelse return error.MissingTracePromptContextProjection;
    const prompt_context_bias = prompt_context.context_bias_milli orelse return error.MissingTracePromptContextBias;
    if (trace.prompt_context_top_projection_milli != prompt_context_projection) return error.TracePromptContextProjectionRouteComparisonMismatch;
    if (trace.prompt_context_top_bias_milli != prompt_context_bias) return error.TracePromptContextBiasRouteComparisonMismatch;
    if (trace.raw_to_conditioned_gain_milli != gains.raw_to_conditioned_milli) return error.TraceRawToConditionedGainRouteComparisonMismatch;
    if (trace.conditioned_to_model_gain_milli != gains.conditioned_to_model_milli) return error.TraceConditionedToModelGainRouteComparisonMismatch;
    if (trace.model_to_prompt_context_gain_milli != gains.model_to_prompt_context_milli) return error.TraceModelToPromptContextGainRouteComparisonMismatch;
}

pub fn freeTrace(
    allocator: std.mem.Allocator,
    trace: *DecoderFixtureTrace,
) void {
    allocator.free(trace.candidates);
}

pub fn decodeFixtureWithRuntime(
    allocator: std.mem.Allocator,
    payload: DecoderFixturePayload,
    token_decoder: tokenizer.FixtureTokenizer,
    projection: weights.FixtureDecoderWeights,
    step_count: usize,
) !DecoderFixtureDecode {
    _ = try summarizeFixtureWithRuntime(
        payload,
        token_decoder,
        projection,
    );

    const generated = try allocator.alloc(DecoderGeneratedToken, step_count);
    errdefer allocator.free(generated);

    const context_token_ids = try allocator.alloc(usize, payload.token_ids.len + step_count);
    defer allocator.free(context_token_ids);
    @memcpy(context_token_ids[0..payload.token_ids.len], payload.token_ids);

    var current_context_len = payload.token_ids.len;
    var source_token_id = context_token_ids[current_context_len - 1];
    for (0..step_count) |step_index| {
        const source_token_text = try token_decoder.decodeToken(source_token_id);
        const model_top = try projection.promptContextTopToken(
            context_token_ids[0..current_context_len],
        );
        const token_id = model_top.token_id;
        const token_text = try token_decoder.decodeToken(token_id);
        const projection_milli = try projection.topTokenWeightMilli(token_id);
        const context_bias_milli = try projection.promptContextBiasMilli(
            context_token_ids[0..current_context_len],
            token_id,
        );

        generated[step_index] = .{
            .step_index = step_index,
            .source_context_token_count = current_context_len,
            .source_token_id = source_token_id,
            .source_token_text = source_token_text,
            .token_id = token_id,
            .token_text = token_text,
            .projection_milli = projection_milli,
            .context_bias_milli = context_bias_milli,
            .conditioned_score_milli = model_top.score_milli,
        };

        context_token_ids[current_context_len] = token_id;
        current_context_len += 1;
        source_token_id = token_id;
    }

    return .{
        .family = payload.family,
        .fixture_name = payload.fixture_name,
        .prompt_text = payload.prompt_text,
        .prompt_token_ids = payload.token_ids,
        .generated = generated,
    };
}

pub fn decodeFixture(
    allocator: std.mem.Allocator,
    payload: DecoderFixturePayload,
    step_count: usize,
) !DecoderFixtureDecode {
    return decodeFixtureWithRuntime(
        allocator,
        payload,
        tokenizer.defaultTokenizer(),
        weights.defaultWeights(),
        step_count,
    );
}

pub fn freeDecode(
    allocator: std.mem.Allocator,
    decode: *DecoderFixtureDecode,
) void {
    allocator.free(decode.generated);
}

fn parseSamplePayload(allocator: std.mem.Allocator) !std.json.Parsed(DecoderFixturePayload) {
    return llm_samples.parsePayload(allocator);
}

fn traceSampleFixture(allocator: std.mem.Allocator) !DecoderFixtureTrace {
    var parsed = try parseSamplePayload(allocator);
    defer parsed.deinit();

    return traceFixture(allocator, parsed.value);
}

fn decodeSampleFixture(
    allocator: std.mem.Allocator,
    step_count: usize,
) !DecoderFixtureDecode {
    var parsed = try parseSamplePayload(allocator);
    defer parsed.deinit();

    return decodeFixture(allocator, parsed.value, step_count);
}

fn expectSampleTrace(trace: DecoderFixtureTrace) !void {
    const expected_routes = llm_samples.routeComparison();
    const expected_candidates = llm_samples.debugCandidatesSemantic();
    try std.testing.expectEqual(@as(usize, expected_routes.raw.token_id), trace.raw_top_token_id);
    try std.testing.expectEqualStrings(expected_routes.raw.token_text, trace.raw_top_token_text);
    try std.testing.expectEqual(@as(usize, expected_routes.raw.score_milli), trace.raw_top_logit_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.conditioned.token_id), trace.conditioned_top_token_id);
    try std.testing.expectEqualStrings(expected_routes.conditioned.token_text, trace.conditioned_top_token_text);
    try std.testing.expectEqual(@as(usize, expected_routes.conditioned.score_milli), trace.conditioned_top_score_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.model.token_id), trace.model_top_token_id);
    try std.testing.expectEqualStrings(expected_routes.model.token_text, trace.model_top_token_text);
    try std.testing.expectEqual(@as(usize, expected_routes.model.score_milli), trace.model_top_score_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.prompt_context.token_id), trace.prompt_context_top_token_id);
    try std.testing.expectEqualStrings(expected_routes.prompt_context.token_text, trace.prompt_context_top_token_text);
    try std.testing.expectEqual(@as(usize, expected_routes.prompt_context.projection_milli.?), trace.prompt_context_top_projection_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.prompt_context.context_bias_milli.?), trace.prompt_context_top_bias_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.prompt_context.score_milli), trace.prompt_context_top_score_milli);
    try std.testing.expectEqualStrings(llm_samples.preferred_route_name, trace.preferred_route);
    try std.testing.expectEqual(@as(usize, expected_routes.preferred.token_id), trace.preferred_top_token_id);
    try std.testing.expectEqualStrings(expected_routes.preferred.token_text, trace.preferred_top_token_text);
    try std.testing.expectEqual(@as(usize, expected_routes.preferred.score_milli), trace.preferred_top_score_milli);
    try std.testing.expectEqualStrings(expected_routes.preferred.route, trace.route_comparison.preferred.route);
    try std.testing.expectEqual(@as(usize, expected_routes.prompt_context.score_milli), trace.route_comparison.prompt_context.score_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.gains.raw_to_conditioned_milli), trace.raw_to_conditioned_gain_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.gains.conditioned_to_model_milli), trace.conditioned_to_model_gain_milli);
    try std.testing.expectEqual(@as(usize, expected_routes.gains.model_to_prompt_context_milli), trace.model_to_prompt_context_gain_milli);
    try std.testing.expectEqual(@as(usize, tokenizer.defaultTokenizer().vocab.len), trace.candidates.len);
    try std.testing.expectEqual(@as(usize, expected_candidates[1].transition_bias_milli), trace.candidates[1].transition_bias_milli);
    try std.testing.expectEqual(@as(usize, expected_candidates[1].conditioned_score_milli), trace.candidates[1].conditioned_score_milli);
    try std.testing.expectEqual(@as(usize, expected_candidates[1].model_score_milli), trace.candidates[1].model_score_milli);
    try std.testing.expectEqual(@as(usize, expected_candidates[1].prompt_context_bias_milli), trace.candidates[1].prompt_context_bias_milli);
    try std.testing.expectEqual(@as(usize, expected_candidates[1].prompt_context_score_milli), trace.candidates[1].prompt_context_score_milli);
}

fn expectSampleDecode(decode: DecoderFixtureDecode) !void {
    const expected_generated = llm_samples.generatedTokens();
    try std.testing.expectEqual(@as(usize, llm_samples.decode_step_count), decode.generated.len);
    try std.testing.expectEqualStrings(expected_generated[0].token_text, decode.generated[0].token_text);
    try std.testing.expectEqualStrings(expected_generated[1].token_text, decode.generated[1].token_text);
    try std.testing.expectEqualStrings(llm_samples.metal_token_text, decode.generated[2].token_text);
    try std.testing.expectEqual(@as(usize, expected_generated[0].source_context_token_count), decode.generated[0].source_context_token_count);
    try std.testing.expectEqual(@as(usize, expected_generated[1].source_context_token_count), decode.generated[1].source_context_token_count);
    try std.testing.expectEqual(@as(usize, llm_samples.third_step_source_context_token_count), decode.generated[2].source_context_token_count);
    try std.testing.expectEqual(@as(usize, expected_generated[0].projection_milli), decode.generated[0].projection_milli);
    try std.testing.expectEqual(@as(usize, expected_generated[0].context_bias_milli), decode.generated[0].context_bias_milli);
    try std.testing.expectEqual(@as(usize, expected_generated[1].projection_milli), decode.generated[1].projection_milli);
    try std.testing.expectEqual(@as(usize, expected_generated[1].context_bias_milli), decode.generated[1].context_bias_milli);
    try std.testing.expectEqual(@as(usize, llm_samples.metal_projection_milli), decode.generated[2].projection_milli);
    try std.testing.expectEqual(@as(usize, llm_samples.third_step_context_bias_milli), decode.generated[2].context_bias_milli);
    try std.testing.expectEqual(@as(usize, expected_generated[0].conditioned_score_milli), decode.generated[0].conditioned_score_milli);
    try std.testing.expectEqual(@as(usize, expected_generated[1].conditioned_score_milli), decode.generated[1].conditioned_score_milli);
    try std.testing.expectEqual(@as(usize, llm_samples.third_step_conditioned_score_milli), decode.generated[2].conditioned_score_milli);
}

test "summarize decoder fixture payload" {
    const allocator = std.testing.allocator;
    var parsed = try parseSamplePayload(allocator);
    defer parsed.deinit();

    const summary = try summarizeFixture(parsed.value);
    const expected = llm_samples.summary();
    try validateSummary(summary, expected);
}

test "reject non-llm payload family" {
    const allocator = std.testing.allocator;
    var parsed = try llm_samples.parseInvalidFamilyPayload(allocator);
    defer parsed.deinit();

    try std.testing.expectError(error.UnexpectedFamily, summarizeFixture(parsed.value));
}

test "validate decoder summary" {
    const expected = llm_samples.summary();
    const summary = expected;
    try validateSummary(summary, expected);
}

test "reject inconsistent decoder summary compatibility fields" {
    const summary = llm_samples.invalidPreferredTokenSummary();
    try std.testing.expectError(error.GeneratedLastTokenPreferredMismatch, validateSummaryConsistency(summary));

    const preferred_score_summary = llm_samples.invalidPreferredScoreSummary();
    try std.testing.expectError(error.PreferredNextScoreRouteComparisonMismatch, validateSummaryConsistency(preferred_score_summary));
}

test "trace decoder fixture payload" {
    const allocator = std.testing.allocator;
    var trace = try traceSampleFixture(allocator);
    defer freeTrace(allocator, &trace);

    try expectSampleTrace(trace);
    try validateTraceConsistency(trace);
}

test "decode fixture produces deterministic rollout" {
    const allocator = std.testing.allocator;
    var decode = try decodeSampleFixture(allocator, llm_samples.decode_step_count);
    defer freeDecode(allocator, &decode);

    try expectSampleDecode(decode);
}
