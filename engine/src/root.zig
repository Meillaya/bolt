pub const context = @import("metal/context.zig");
pub const fixtures = @import("fixtures/manifest.zig");
pub const payload_identity = @import("fixtures/payload_identity.zig");
pub const layout = @import("tensor/layout.zig");
pub const buffer = @import("tensor/buffer.zig");
pub const mnist = @import("models/mnist.zig");
pub const decoder = @import("models/decoder.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const weights = @import("weights.zig");
pub const runtime = struct {
    pub const debug_reports = @import("runtime/debug_reports.zig");
    pub const llm_assets = @import("runtime/llm_assets.zig");
    pub const reporting = @import("runtime/reporting.zig");
    pub const proof_checks = @import("runtime/proof_checks.zig");
    pub const proof_runs = @import("runtime/proof_runs.zig");
};

test {
    @import("std").testing.refAllDecls(@This());
}
