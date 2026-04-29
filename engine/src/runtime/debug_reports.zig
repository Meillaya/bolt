const std = @import("std");
const fixtures = @import("../fixtures/manifest.zig");
const mnist = @import("../models/mnist.zig");
const decoder = @import("../models/decoder.zig");
const llm_assets = @import("llm_assets.zig");
const llm_assertions = @import("../testing/llm_assertions.zig");
const llm_samples = @import("../testing/llm_samples.zig");
const mnist_samples = @import("../testing/mnist_samples.zig");

const stringify_options: std.json.Stringify.Options = .{
    .whitespace = .indent_2,
    .emit_null_optional_fields = false,
};

const DebugManifestReport = struct {
    payload_file: []const u8,
    expected_file: []const u8,
    runtime_bundle_file: ?[]const u8 = null,
    fixture_version: usize,
};

fn manifestReport(manifest: fixtures.FixtureManifest) DebugManifestReport {
    return .{
        .payload_file = manifest.payload_file,
        .expected_file = manifest.expected_file,
        .runtime_bundle_file = manifest.runtime_bundle_file,
        .fixture_version = manifest.fixture_version,
    };
}

pub fn writeMnistDebugReport(
    writer: anytype,
    family_label: []const u8,
    manifest: fixtures.FixtureManifest,
    trace: mnist.MnistFixtureTrace,
) !void {
    const TraceReport = struct {
        rows: usize,
        cols: usize,
        pixel_sum: usize,
        non_zero_count: usize,
        predicted_label: usize,
        top_pixel_index: usize,
        top_pixel_row: usize,
        top_pixel_col: usize,
        top_pixel_value_milli: usize,
        non_zero_pixels: []const mnist.MnistNonZeroPixel,
    };
    try std.json.Stringify.value(.{
        .family = family_label,
        .fixture_name = manifest.fixture_name,
        .manifest = manifestReport(manifest),
        .trace = TraceReport{
            .rows = trace.rows,
            .cols = trace.cols,
            .pixel_sum = trace.pixel_sum,
            .non_zero_count = trace.non_zero_count,
            .predicted_label = trace.predicted_label,
            .top_pixel_index = trace.top_pixel_index,
            .top_pixel_row = trace.top_pixel_row,
            .top_pixel_col = trace.top_pixel_col,
            .top_pixel_value_milli = trace.top_pixel_value_milli,
            .non_zero_pixels = trace.non_zero_pixels,
        },
    }, stringify_options, writer);
    try writer.writeByte('\n');
}

pub fn writeLlmDebugReport(
    writer: anytype,
    family_label: []const u8,
    manifest: fixtures.FixtureManifest,
    runtime_assets: llm_assets.LoadedLlmRuntimeAssets,
    trace: decoder.DecoderFixtureTrace,
    decode: decoder.DecoderFixtureDecode,
) !void {
    try decoder.validateTraceConsistency(trace);
    const route_comparison = trace.route_comparison;
    const raw = route_comparison.raw;
    const conditioned = route_comparison.conditioned;
    const model = route_comparison.model;
    const prompt_context = route_comparison.prompt_context;
    const preferred = route_comparison.preferred;
    const gains = route_comparison.gains;
    const RuntimeReport = struct {
        model_file: []const u8,
        tokenizer_file: []const u8,
        weights_file: []const u8,
        loader: []const u8,
        model_name: []const u8,
        architecture: []const u8,
        context_length: usize,
        weights_format: []const u8,
        tokenizer_vocab_size: usize,
        weights_vocab_size: usize,
        transition_bias_value_count: usize,
    };
    const RouteWinnerReport = struct {
        route: []const u8,
        token_id: usize,
        token_text: []const u8,
        score_milli: usize,
    };
    const PromptContextWinnerReport = struct {
        route: []const u8,
        token_id: usize,
        token_text: []const u8,
        projection_milli: usize,
        context_bias_milli: usize,
        score_milli: usize,
    };
    const RouteGainsReport = struct {
        raw_to_conditioned_milli: usize,
        conditioned_to_model_milli: usize,
        model_to_prompt_context_milli: usize,
    };
    const RouteComparisonReport = struct {
        raw: RouteWinnerReport,
        conditioned: RouteWinnerReport,
        model: RouteWinnerReport,
        prompt_context: PromptContextWinnerReport,
        preferred: RouteWinnerReport,
        gains: RouteGainsReport,
    };
    const TraceReport = struct {
        prompt_text: []const u8,
        prompt_tail_token_id: usize,
        prompt_tail_token_text: []const u8,
        raw_top_token_id: usize,
        raw_top_token_text: []const u8,
        raw_top_logit_milli: usize,
        conditioned_top_token_id: usize,
        conditioned_top_token_text: []const u8,
        conditioned_top_score_milli: usize,
        conditioned_top_probability_milli: usize,
        backend: []const u8,
        dispatched_kernels: decoder.DecoderKernelEvidence,
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
        route_comparison: RouteComparisonReport,
        candidates: []const decoder.DecoderCandidateScore,
    };
    const DecodeReport = struct {
        step_count: usize,
        prompt_token_ids: []const usize,
        generated: []const decoder.DecoderGeneratedToken,
    };
    try std.json.Stringify.value(.{
        .family = family_label,
        .fixture_name = manifest.fixture_name,
        .manifest = manifestReport(manifest),
        .runtime = RuntimeReport{
            .model_file = runtime_assets.bundle.value.model_file,
            .tokenizer_file = runtime_assets.bundle.value.tokenizer_file,
            .weights_file = runtime_assets.bundle.value.weights_file,
            .loader = runtime_assets.assets.model.loader,
            .model_name = runtime_assets.assets.model.model_name,
            .architecture = runtime_assets.assets.model.architecture,
            .context_length = runtime_assets.assets.model.context_length,
            .weights_format = runtime_assets.assets.weights_format,
            .tokenizer_vocab_size = runtime_assets.assets.tokenizer.vocabSize(),
            .weights_vocab_size = runtime_assets.assets.weights.vocabSize(),
            .transition_bias_value_count = runtime_assets.assets.weights.transition_bias.len,
        },
        .trace = TraceReport{
            .prompt_text = trace.prompt_text,
            .prompt_tail_token_id = trace.prompt_tail_token_id,
            .prompt_tail_token_text = trace.prompt_tail_token_text,
            .raw_top_token_id = raw.token_id,
            .raw_top_token_text = raw.token_text,
            .raw_top_logit_milli = raw.score_milli,
            .conditioned_top_token_id = conditioned.token_id,
            .conditioned_top_token_text = conditioned.token_text,
            .conditioned_top_score_milli = conditioned.score_milli,
            .conditioned_top_probability_milli = trace.conditioned_top_probability_milli,
            .backend = trace.backend,
            .dispatched_kernels = trace.dispatched_kernels,
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
            .route_comparison = .{
                .raw = .{
                    .route = raw.route,
                    .token_id = raw.token_id,
                    .token_text = raw.token_text,
                    .score_milli = raw.score_milli,
                },
                .conditioned = .{
                    .route = conditioned.route,
                    .token_id = conditioned.token_id,
                    .token_text = conditioned.token_text,
                    .score_milli = conditioned.score_milli,
                },
                .model = .{
                    .route = model.route,
                    .token_id = model.token_id,
                    .token_text = model.token_text,
                    .score_milli = model.score_milli,
                },
                .prompt_context = .{
                    .route = prompt_context.route,
                    .token_id = prompt_context.token_id,
                    .token_text = prompt_context.token_text,
                    .projection_milli = prompt_context.projection_milli.?,
                    .context_bias_milli = prompt_context.context_bias_milli.?,
                    .score_milli = prompt_context.score_milli,
                },
                .preferred = .{
                    .route = preferred.route,
                    .token_id = preferred.token_id,
                    .token_text = preferred.token_text,
                    .score_milli = preferred.score_milli,
                },
                .gains = .{
                    .raw_to_conditioned_milli = gains.raw_to_conditioned_milli,
                    .conditioned_to_model_milli = gains.conditioned_to_model_milli,
                    .model_to_prompt_context_milli = gains.model_to_prompt_context_milli,
                },
            },
            .candidates = trace.candidates,
        },
        .decode = DecodeReport{
            .step_count = decode.generated.len,
            .prompt_token_ids = decode.prompt_token_ids,
            .generated = decode.generated,
        },
    }, stringify_options, writer);
    try writer.writeByte('\n');
}

const SampleLlmDebugRuntime = struct {
    manifest: fixtures.FixtureManifest,
    runtime_assets: llm_assets.LoadedLlmRuntimeAssets,

    fn load(allocator: std.mem.Allocator) !SampleLlmDebugRuntime {
        return .{
            .manifest = llm_samples.manifest(),
            .runtime_assets = try llm_samples.runtimeAssets(allocator),
        };
    }

    fn deinit(self: *SampleLlmDebugRuntime, allocator: std.mem.Allocator) void {
        self.runtime_assets.deinit(allocator);
    }
};

const SampleLlmDebugCase = struct {
    candidates: [2]decoder.DecoderCandidateScore,
    generated: [2]decoder.DecoderGeneratedToken,
    trace: decoder.DecoderFixtureTrace,
    decode: decoder.DecoderFixtureDecode,

    fn fromCandidates(candidates: [2]decoder.DecoderCandidateScore) SampleLlmDebugCase {
        var generated = llm_samples.generatedTokens();
        return .{
            .candidates = candidates,
            .generated = generated,
            .trace = llm_samples.debugTrace(candidates[0..]),
            .decode = llm_samples.debugDecode(generated[0..]),
        };
    }

    fn includes() SampleLlmDebugCase {
        return fromCandidates(llm_samples.debugCandidatesIncludes());
    }

    fn semantic() SampleLlmDebugCase {
        return fromCandidates(llm_samples.debugCandidatesSemantic());
    }
};

test "writeMnistDebugReport includes manifest and pixel trace" {
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();

    const non_zero_pixels = mnist_samples.debugPixelsIncludes();
    const manifest = mnist_samples.manifest();
    const trace = mnist_samples.trace(non_zero_pixels[0..]);

    try writeMnistDebugReport(&aw.writer, "mnist", manifest, trace);

    try llm_assertions.expectContainsAll(aw.written(), &.{
        std.fmt.comptimePrint("\"payload_file\": \"{s}\"", .{manifest.payload_file}),
        std.fmt.comptimePrint("\"predicted_label\": {d}", .{trace.predicted_label}),
        std.fmt.comptimePrint("\"top_pixel_index\": {d}", .{trace.top_pixel_index}),
        std.fmt.comptimePrint("\"value_milli\": {d}", .{trace.top_pixel_value_milli}),
    });
}

test "writeMnistDebugReport parses as semantic JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();

    const non_zero_pixels = mnist_samples.debugPixelsSemantic();
    const manifest = mnist_samples.manifest();
    const trace = mnist_samples.trace(non_zero_pixels[0..]);

    try writeMnistDebugReport(&aw.writer, "mnist", manifest, trace);

    var parsed = try llm_assertions.parseJsonObject(allocator, aw.written());
    defer parsed.deinit();

    const root = parsed.value.object;
    const manifest_root = root.get("manifest").?.object;
    const trace_root = root.get("trace").?.object;
    const trace_non_zero_pixels = trace_root.get("non_zero_pixels").?.array.items;
    const final_expected_non_zero_pixel = trace.non_zero_pixels[2];
    const final_trace_non_zero_pixel = trace_non_zero_pixels[2].object;
    try std.testing.expectEqualStrings(mnist_samples.family_name, root.get("family").?.string);
    try std.testing.expectEqualStrings(mnist_samples.fixture_name, root.get("fixture_name").?.string);
    try std.testing.expectEqual(@as(i64, @intCast(manifest.fixture_version)), manifest_root.get("fixture_version").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(trace.predicted_label)), trace_root.get("predicted_label").?.integer);
    try std.testing.expectEqual(@as(i64, @intCast(trace.top_pixel_index)), trace_root.get("top_pixel_index").?.integer);
    try std.testing.expectEqual(@as(usize, trace.non_zero_pixels.len), trace_non_zero_pixels.len);
    try std.testing.expectEqual(@as(i64, @intCast(final_expected_non_zero_pixel.row)), final_trace_non_zero_pixel.get("row").?.integer);
}

test "writeLlmDebugReport includes runtime trace and decode sections" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var sample = try SampleLlmDebugRuntime.load(allocator);
    defer sample.deinit(allocator);
    const debug_case = SampleLlmDebugCase.includes();

    try writeLlmDebugReport(&aw.writer, "llm", sample.manifest, sample.runtime_assets, debug_case.trace, debug_case.decode);

    try llm_assertions.expectDebugReportContainsCore(aw.written());
}

test "writeLlmDebugReport parses as semantic JSON" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var sample = try SampleLlmDebugRuntime.load(allocator);
    defer sample.deinit(allocator);
    const debug_case = SampleLlmDebugCase.semantic();

    try writeLlmDebugReport(&aw.writer, "llm", sample.manifest, sample.runtime_assets, debug_case.trace, debug_case.decode);

    var parsed = try llm_assertions.parseJsonObject(allocator, aw.written());
    defer parsed.deinit();

    const root = parsed.value.object;
    try llm_assertions.expectDebugReportSemanticRoot(root);
}

test "writeLlmDebugReport rejects inconsistent trace compatibility fields" {
    const allocator = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    var sample = try SampleLlmDebugRuntime.load(allocator);
    defer sample.deinit(allocator);

    var payload = try llm_samples.parsePayload(allocator);
    defer payload.deinit();

    var trace = try decoder.traceFixture(allocator, payload.value);
    defer decoder.freeTrace(allocator, &trace);

    var decode = try decoder.decodeFixture(allocator, payload.value, 2);
    defer decoder.freeDecode(allocator, &decode);

    trace.preferred_top_score_milli = 2500;

    try std.testing.expectError(
        error.TracePreferredScoreRouteComparisonMismatch,
        writeLlmDebugReport(&aw.writer, "llm", sample.manifest, sample.runtime_assets, trace, decode),
    );
}
