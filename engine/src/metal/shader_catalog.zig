const std = @import("std");

pub const specialized_hidden_k: u32 = 512;
pub const specialized_inter_k: u32 = 1024;
pub const specialized_group_size: u32 = 32;

pub const ShaderGroup = enum {
    active_primitives,
    compute,
    transformer,
    qmv_specialized,
    q4mv_bf16_specialized,
};

pub const ShaderGroupInfo = struct {
    group: ShaderGroup,
    reference_file: []const u8,
    bolt_file: []const u8,
    kernel_count: usize,
    requires_specialization_macros: bool = false,
};

pub const groups = [_]ShaderGroupInfo{
    .{
        .group = .active_primitives,
        .reference_file = "reference/nnzap/nnmetal/src/shaders/compute.metal subset",
        .bolt_file = "engine/src/metal/kernels.metal",
        .kernel_count = 7,
    },
    .{
        .group = .compute,
        .reference_file = "reference/nnzap/nnmetal/src/shaders/compute.metal",
        .bolt_file = "engine/src/metal/shaders/compute.metal",
        .kernel_count = 60,
    },
    .{
        .group = .transformer,
        .reference_file = "reference/nnzap/nnmetal/src/shaders/transformer.metal",
        .bolt_file = "engine/src/metal/shaders/transformer.metal",
        .kernel_count = 23,
    },
    .{
        .group = .qmv_specialized,
        .reference_file = "reference/nnzap/nnmetal/src/shaders/qmv_specialized.metal",
        .bolt_file = "engine/src/metal/shaders/qmv_specialized.metal",
        .kernel_count = 9,
        .requires_specialization_macros = true,
    },
    .{
        .group = .q4mv_bf16_specialized,
        .reference_file = "reference/nnzap/nnmetal/src/shaders/q4mv_bf16_specialized.metal",
        .bolt_file = "engine/src/metal/shaders/q4mv_bf16_specialized.metal",
        .kernel_count = 9,
        .requires_specialization_macros = true,
    },
};

pub fn totalReferenceKernelCount() usize {
    var count: usize = 0;
    for (groups) |group| {
        if (group.group != .active_primitives) count += group.kernel_count;
    }
    return count;
}

pub fn findGroup(group: ShaderGroup) ?ShaderGroupInfo {
    for (groups) |info| {
        if (info.group == group) return info;
    }
    return null;
}

test "shader catalog records every Milestone 1 library group" {
    try std.testing.expectEqual(@as(usize, 5), groups.len);
    try std.testing.expectEqual(@as(usize, 101), totalReferenceKernelCount());
    try std.testing.expect(findGroup(.compute) != null);
    try std.testing.expect(findGroup(.transformer) != null);
    try std.testing.expect(findGroup(.qmv_specialized).?.requires_specialization_macros);
    try std.testing.expect(findGroup(.q4mv_bf16_specialized).?.requires_specialization_macros);
}
