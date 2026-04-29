const std = @import("std");
const fixtures = @import("../fixtures/manifest.zig");
const decoder = @import("../models/decoder.zig");
const llm_assets = @import("../runtime/llm_assets.zig");
const tokenizer = @import("../tokenizer.zig");
const weights = @import("../weights.zig");

pub const family_name = "llm";
pub const fixture_name = "llm-smoke";
pub const payload_file_name = "llm-smoke.json";
pub const expected_file_name = "llm-smoke.expected.json";
pub const runtime_bundle_file_name = "llm-smoke.runtime.json";
pub const model_file_name = "llm-smoke.model.json";
pub const description = "Deterministic smoke fixture for the small decoder-only path.";
pub const backend_name = "metal";
pub const loader_name = llm_assets.sample_loader_name;
pub const model_name = llm_assets.sample_model_name;
pub const model_architecture = llm_assets.sample_architecture_name;
pub const model_vocab_size = 4;
pub const model_context_length = 8;
pub const weights_format = "binary-f32-le";
pub const raw_route_name = "raw";
pub const conditioned_route_name = "conditioned";
pub const model_route_name = "model";
pub const preferred_route_name = decoder.preferred_route_label;
pub const prompt_text = "zig metal bolt";
pub const zig_token_id = 1;
pub const metal_token_id = 2;
pub const bolt_token_id = 3;
pub const prompt_token_count = 3;
pub const generated_token_count = prompt_token_count + 1;
pub const second_step_source_context_token_count = prompt_token_count + 1;
pub const decode_step_count = 3;
pub const prompt_token_sum = zig_token_id + metal_token_id + bolt_token_id;
pub const zig_token_text = "zig";
pub const metal_token_text = "metal";
pub const bolt_token_text = "bolt";
pub const transition_bias_value_count = weights.defaultWeights().transition_bias.len;
pub const preferred_score_milli = 2400;
pub const preferred_projection_milli = 1500;
pub const preferred_context_bias_milli = 900;
pub const model_score_milli = 2300;
pub const conditioned_score_milli = 1000;
pub const conditioned_probability_milli = 343;
pub const raw_score_milli = 900;
pub const conditioned_to_model_gain_milli = 1300;
pub const prompt_context_over_conditioned_gain_milli = 1400;
pub const prompt_context_over_model_gain_milli = 100;
pub const raw_to_conditioned_gain_milli = 100;
pub const metal_projection_milli = 2000;
pub const conditioning_bias_milli = 800;
pub const zig_raw_logit_milli = 200;
pub const pad_raw_logit_milli = 100;
pub const pad_projection_score_milli = 1000;
pub const repeat_step_conditioned_score_milli = 2500;
pub const third_step_source_context_token_count = 5;
pub const third_step_context_bias_milli = 700;
pub const third_step_conditioned_score_milli = 2700;

pub fn summary() decoder.DecoderFixtureSummary {
    return .{
        .family = family_name,
        .fixture_name = fixture_name,
        .backend = backend_name,
        .loader = loader_name,
        .model_name = model_name,
        .model_architecture = model_architecture,
        .model_vocab_size = model_vocab_size,
        .model_context_length = model_context_length,
        .weights_format = weights_format,
        .dispatched_kernels = .{
            .bias_add_f32 = true,
            .softmax_f32 = true,
        },
        .prompt_token_count = prompt_token_count,
        .prompt_sum = prompt_token_sum,
        .logits_count = tokenizer.defaultTokenizer().vocab.len,
        .next_token_id = metal_token_id,
        .generated_token_count = generated_token_count,
        .generated_last_token = zig_token_id,
        .prompt_tail_token_id = bolt_token_id,
        .prompt_tail_token_text = bolt_token_text,
        .next_token_text = metal_token_text,
        .raw_next_score_milli = raw_score_milli,
        .conditioned_next_token_id = zig_token_id,
        .conditioned_next_token_text = zig_token_text,
        .model_next_token_id = zig_token_id,
        .model_next_token_text = zig_token_text,
        .prompt_context_next_token_id = zig_token_id,
        .prompt_context_next_token_text = zig_token_text,
        .preferred_route = preferred_route_name,
        .preferred_next_token_id = zig_token_id,
        .preferred_next_token_text = zig_token_text,
        .preferred_next_score_milli = preferred_score_milli,
        .route_comparison = routeComparison(),
        .generated_projection_milli = preferred_projection_milli,
        .generated_context_bias_milli = preferred_context_bias_milli,
        .prompt_context_over_raw_gain_milli = preferred_projection_milli,
        .prompt_context_over_conditioned_gain_milli = prompt_context_over_conditioned_gain_milli,
        .prompt_context_over_model_gain_milli = prompt_context_over_model_gain_milli,
        .raw_conditioning_flipped = true,
        .conditioned_matches_model = true,
        .top_token_weight_milli = metal_projection_milli,
        .weighted_logit_sum_milli = preferred_score_milli,
        .prompt_condition_bias_milli = conditioning_bias_milli,
        .prompt_context_bias_milli = preferred_context_bias_milli,
        .conditioned_next_score_milli = conditioned_score_milli,
        .conditioned_next_probability_milli = conditioned_probability_milli,
        .model_next_score_milli = model_score_milli,
        .prompt_context_next_score_milli = preferred_score_milli,
        .model_condition_gap_milli = conditioned_to_model_gain_milli,
        .logit_margin_milli = 500,
    };
}

pub fn invalidPreferredTokenSummary() decoder.DecoderFixtureSummary {
    var value = summary();
    value.preferred_next_token_id = metal_token_id;
    return value;
}

pub fn invalidPreferredScoreSummary() decoder.DecoderFixtureSummary {
    var value = summary();
    value.preferred_next_score_milli = repeat_step_conditioned_score_milli;
    return value;
}

pub fn manifest() fixtures.FixtureManifest {
    return .{
        .fixture_version = fixtures.current_fixture_version,
        .family = family_name,
        .fixture_name = fixture_name,
        .payload_file = payload_file_name,
        .expected_file = expected_file_name,
        .runtime_bundle_file = runtime_bundle_file_name,
        .description = description,
    };
}

pub fn runtimeAssets(allocator: std.mem.Allocator) !llm_assets.LoadedLlmRuntimeAssets {
    const bundle = try llm_assets.parseBundleFromSlice(
        allocator,
        llm_assets.sample_bundle_input,
    );
    const model_config = try llm_assets.parseModelFromSlice(
        allocator,
        llm_assets.sample_model_input,
    );
    const tokenizer_config = try tokenizer.parseConfigFromSlice(
        allocator,
        tokenizer.sample_config_input,
    );

    return .{
        .bundle = bundle,
        .model_config = model_config,
        .tokenizer_config = tokenizer_config,
        .weights_config = null,
        .owned_binary_values = null,
        .assets = .{
            .tokenizer = tokenizer.defaultTokenizer(),
            .weights = weights.defaultWeights(),
            .model = model_config.value,
            .weights_format = weights_format,
        },
    };
}

pub fn routeComparison() decoder.DecoderRouteComparison {
    return .{
        .raw = .{ .route = raw_route_name, .token_id = metal_token_id, .token_text = metal_token_text, .score_milli = raw_score_milli },
        .conditioned = .{ .route = conditioned_route_name, .token_id = zig_token_id, .token_text = zig_token_text, .score_milli = conditioned_score_milli },
        .model = .{ .route = model_route_name, .token_id = zig_token_id, .token_text = zig_token_text, .score_milli = model_score_milli },
        .prompt_context = .{
            .route = preferred_route_name,
            .token_id = zig_token_id,
            .token_text = zig_token_text,
            .score_milli = preferred_score_milli,
            .projection_milli = preferred_projection_milli,
            .context_bias_milli = preferred_context_bias_milli,
        },
        .preferred = .{
            .route = preferred_route_name,
            .token_id = zig_token_id,
            .token_text = zig_token_text,
            .score_milli = preferred_score_milli,
            .projection_milli = preferred_projection_milli,
            .context_bias_milli = preferred_context_bias_milli,
        },
        .gains = .{
            .raw_to_conditioned_milli = raw_to_conditioned_gain_milli,
            .conditioned_to_model_milli = conditioned_to_model_gain_milli,
            .model_to_prompt_context_milli = prompt_context_over_model_gain_milli,
            .prompt_context_over_raw_milli = preferred_projection_milli,
            .prompt_context_over_conditioned_milli = prompt_context_over_conditioned_gain_milli,
            .prompt_context_over_model_milli = prompt_context_over_model_gain_milli,
        },
    };
}

pub fn debugCandidatesIncludes() [2]decoder.DecoderCandidateScore {
    return .{
        .{
            .token_id = zig_token_id,
            .token_text = zig_token_text,
            .raw_logit_milli = zig_raw_logit_milli,
            .output_projection_milli = preferred_projection_milli,
            .transition_bias_milli = conditioning_bias_milli,
            .conditioned_score_milli = conditioned_score_milli,
            .conditioned_probability_milli = conditioned_probability_milli,
            .model_score_milli = model_score_milli,
            .prompt_context_bias_milli = preferred_context_bias_milli,
            .prompt_context_score_milli = preferred_score_milli,
        },
        .{
            .token_id = metal_token_id,
            .token_text = metal_token_text,
            .raw_logit_milli = raw_score_milli,
            .output_projection_milli = metal_projection_milli,
            .transition_bias_milli = 0,
            .conditioned_score_milli = raw_score_milli,
            .conditioned_probability_milli = 0,
            .model_score_milli = metal_projection_milli,
            .prompt_context_bias_milli = 300,
            .prompt_context_score_milli = model_score_milli,
        },
    };
}

pub fn debugCandidatesSemantic() [2]decoder.DecoderCandidateScore {
    return .{
        .{
            .token_id = 0,
            .token_text = "<pad>",
            .raw_logit_milli = pad_raw_logit_milli,
            .output_projection_milli = pad_projection_score_milli,
            .transition_bias_milli = 0,
            .conditioned_score_milli = pad_raw_logit_milli,
            .conditioned_probability_milli = 0,
            .model_score_milli = pad_projection_score_milli,
            .prompt_context_bias_milli = 0,
            .prompt_context_score_milli = pad_projection_score_milli,
        },
        .{
            .token_id = zig_token_id,
            .token_text = zig_token_text,
            .raw_logit_milli = zig_raw_logit_milli,
            .output_projection_milli = preferred_projection_milli,
            .transition_bias_milli = conditioning_bias_milli,
            .conditioned_score_milli = conditioned_score_milli,
            .conditioned_probability_milli = conditioned_probability_milli,
            .model_score_milli = model_score_milli,
            .prompt_context_bias_milli = preferred_context_bias_milli,
            .prompt_context_score_milli = preferred_score_milli,
        },
    };
}

pub fn generatedTokens() [2]decoder.DecoderGeneratedToken {
    return .{
        .{
            .step_index = 0,
            .source_context_token_count = prompt_token_count,
            .source_token_id = bolt_token_id,
            .source_token_text = bolt_token_text,
            .token_id = zig_token_id,
            .token_text = zig_token_text,
            .projection_milli = preferred_projection_milli,
            .context_bias_milli = preferred_context_bias_milli,
            .conditioned_score_milli = preferred_score_milli,
        },
        .{
            .step_index = 1,
            .source_context_token_count = second_step_source_context_token_count,
            .source_token_id = zig_token_id,
            .source_token_text = zig_token_text,
            .token_id = zig_token_id,
            .token_text = zig_token_text,
            .projection_milli = preferred_projection_milli,
            .context_bias_milli = 1000,
            .conditioned_score_milli = repeat_step_conditioned_score_milli,
        },
    };
}

pub fn debugTrace(candidates: []decoder.DecoderCandidateScore) decoder.DecoderFixtureTrace {
    return .{
        .family = family_name,
        .fixture_name = fixture_name,
        .backend = backend_name,
        .loader = loader_name,
        .model_name = model_name,
        .model_architecture = model_architecture,
        .model_vocab_size = model_vocab_size,
        .model_context_length = model_context_length,
        .weights_format = weights_format,
        .dispatched_kernels = .{
            .bias_add_f32 = true,
            .softmax_f32 = true,
        },
        .prompt_text = prompt_text,
        .prompt_tail_token_id = bolt_token_id,
        .prompt_tail_token_text = bolt_token_text,
        .raw_top_token_id = metal_token_id,
        .raw_top_token_text = metal_token_text,
        .raw_top_logit_milli = raw_score_milli,
        .conditioned_top_token_id = zig_token_id,
        .conditioned_top_token_text = zig_token_text,
        .conditioned_top_score_milli = conditioned_score_milli,
        .conditioned_top_probability_milli = conditioned_probability_milli,
        .model_top_token_id = zig_token_id,
        .model_top_token_text = zig_token_text,
        .model_top_score_milli = model_score_milli,
        .prompt_context_top_token_id = zig_token_id,
        .prompt_context_top_token_text = zig_token_text,
        .prompt_context_top_projection_milli = preferred_projection_milli,
        .prompt_context_top_bias_milli = preferred_context_bias_milli,
        .prompt_context_top_score_milli = preferred_score_milli,
        .preferred_route = preferred_route_name,
        .preferred_top_token_id = zig_token_id,
        .preferred_top_token_text = zig_token_text,
        .preferred_top_score_milli = preferred_score_milli,
        .raw_to_conditioned_gain_milli = raw_to_conditioned_gain_milli,
        .conditioned_to_model_gain_milli = conditioned_to_model_gain_milli,
        .model_to_prompt_context_gain_milli = prompt_context_over_model_gain_milli,
        .route_comparison = routeComparison(),
        .candidates = candidates,
    };
}

pub fn debugDecode(generated: []decoder.DecoderGeneratedToken) decoder.DecoderFixtureDecode {
    return .{
        .family = family_name,
        .fixture_name = fixture_name,
        .prompt_text = prompt_text,
        .prompt_token_ids = &.{ zig_token_id, metal_token_id, bolt_token_id },
        .generated = generated,
    };
}

fn payloadPrefix(comptime payload_family: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        \\{{
        \\  "family": "{s}",
        \\  "fixture_name": "{s}",
        \\  "prompt_text": "{s}",
        \\  "token_ids": [{d}, {d}, {d}],
        \\  "logits": [0.1, 0.2, 0.9, 0.4],
        \\  "expected_prompt_sum": {d},
        \\  "expected_top_token": {d}
        \\}}
    , .{ payload_family, fixture_name, prompt_text, zig_token_id, metal_token_id, bolt_token_id, prompt_token_sum, metal_token_id });
}

pub const payload_input =
    payloadPrefix(family_name);

pub const invalid_family_payload_input =
    payloadPrefix("mnist");

pub fn parsePayload(allocator: std.mem.Allocator) !std.json.Parsed(decoder.DecoderFixturePayload) {
    return decoder.parseFixturePayloadFromSlice(allocator, payload_input);
}

pub fn parseInvalidFamilyPayload(allocator: std.mem.Allocator) !std.json.Parsed(decoder.DecoderFixturePayload) {
    return decoder.parseFixturePayloadFromSlice(allocator, invalid_family_payload_input);
}
