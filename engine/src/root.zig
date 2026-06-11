pub const context = @import("metal/context.zig");
pub const metal = struct {
    pub const context = @import("metal/context.zig");
    pub const shader_catalog = @import("metal/shader_catalog.zig");
};
pub const fixtures = @import("fixtures/manifest.zig");
pub const payload_identity = @import("fixtures/payload_identity.zig");
pub const layout = @import("tensor/layout.zig");
pub const buffer = @import("tensor/buffer.zig");
pub const mnist = @import("models/mnist.zig");
pub const decoder = @import("models/decoder.zig");
pub const tokenizer = @import("tokenizer.zig");
pub const weights = @import("weights.zig");
pub const safetensors = @import("safetensors.zig");
pub const q4 = @import("q4.zig");
pub const network = @import("network.zig");
pub const runtime = struct {
    pub const artifact_metadata = @import("runtime/artifact_metadata.zig");
    pub const debug_reports = @import("runtime/debug_reports.zig");
    pub const llm_assets = @import("runtime/llm_assets.zig");
    pub const mnist_assets = @import("runtime/mnist_assets.zig");
    pub const mnist_idx = @import("runtime/mnist_idx.zig");
    pub const mnist_quant = @import("runtime/mnist_quant.zig");
    pub const metal_mlp = @import("runtime/metal_mlp.zig");
    pub const reporting = @import("runtime/reporting.zig");
    pub const proof_checks = @import("runtime/proof_checks.zig");
    pub const proof_runs = @import("runtime/proof_runs.zig");
};

test {
    @import("std").testing.refAllDecls(@This());
}

pub const transformer = @import("transformer.zig");
pub const bonsai_model = @import("bonsai_model.zig");
