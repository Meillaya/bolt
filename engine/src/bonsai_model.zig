const std = @import("std");

pub const BonsaiTensorReport = struct {
    expected_tensors: u32,
    matched_tensors: u32,
    missing_tensors: u32,
    invalid_tensors: u32,
    extra_tensors: u32,
    tensor_count: u32,

    pub fn pass(self: BonsaiTensorReport) bool {
        return self.expected_tensors == self.matched_tensors and self.missing_tensors == 0 and self.invalid_tensors == 0 and self.extra_tensors == 0;
    }
};

const TensorSpec = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const u32,
};

pub fn inspectBonsai1_7BFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !BonsaiTensorReport {
    const header = try readSafetensorsHeader(io, allocator, path);
    defer allocator.free(header);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;

    var report = BonsaiTensorReport{
        .expected_tensors = 2 + 28 * 11,
        .matched_tensors = 0,
        .missing_tensors = 0,
        .invalid_tensors = 0,
        .extra_tensors = 0,
        .tensor_count = 0,
    };

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        if (!std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) report.tensor_count += 1;
    }

    try checkTensor(&report, parsed.value, .{ .name = "model.embed_tokens.weight", .dtype = "F16", .shape = &.{ 151669, 2048 } });
    try checkTensor(&report, parsed.value, .{ .name = "model.norm.weight", .dtype = "F16", .shape = &.{2048} });
    for (0..28) |layer| {
        var name_buf: [128]u8 = undefined;
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.input_layernorm.weight", .{layer}), .dtype = "F16", .shape = &.{2048} });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.post_attention_layernorm.weight", .{layer}), .dtype = "F16", .shape = &.{2048} });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_norm.weight", .{layer}), .dtype = "F16", .shape = &.{128} });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.k_norm.weight", .{layer}), .dtype = "F16", .shape = &.{128} });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.q_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 2048, 2048 } });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.k_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 1024, 2048 } });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.v_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 1024, 2048 } });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.self_attn.o_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 2048, 2048 } });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.gate_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 6144, 2048 } });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.up_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 6144, 2048 } });
        try checkTensor(&report, parsed.value, .{ .name = try std.fmt.bufPrint(&name_buf, "model.layers.{d}.mlp.down_proj.weight", .{layer}), .dtype = "F16", .shape = &.{ 2048, 6144 } });
    }
    if (report.tensor_count > report.expected_tensors) report.extra_tensors = report.tensor_count - report.expected_tensors;
    return report;
}

fn readSafetensorsHeader(io: std.Io, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var len_bytes: [8]u8 = undefined;
    if (try file.readPositionalAll(io, &len_bytes, 0) != 8) return error.TruncatedFile;
    const header_len = std.mem.readInt(u64, &len_bytes, .little);
    if (header_len == 0 or header_len > 16 * 1024 * 1024) return error.InvalidHeaderLength;
    const header = try allocator.alloc(u8, @intCast(header_len));
    errdefer allocator.free(header);
    if (try file.readPositionalAll(io, header, 8) != header.len) return error.TruncatedFile;
    return header;
}

fn checkTensor(report: *BonsaiTensorReport, root: std.json.Value, spec: TensorSpec) !void {
    const value = root.object.get(spec.name) orelse {
        report.missing_tensors += 1;
        return;
    };
    if (value != .object) {
        report.invalid_tensors += 1;
        return;
    }
    const dtype_value = value.object.get("dtype") orelse {
        report.invalid_tensors += 1;
        return;
    };
    if (dtype_value != .string or !std.mem.eql(u8, dtype_value.string, spec.dtype)) {
        report.invalid_tensors += 1;
        return;
    }
    const shape_value = value.object.get("shape") orelse {
        report.invalid_tensors += 1;
        return;
    };
    if (shape_value != .array or shape_value.array.items.len != spec.shape.len) {
        report.invalid_tensors += 1;
        return;
    }
    for (spec.shape, 0..) |expected, index| {
        const actual = shape_value.array.items[index];
        if (actual != .integer or actual.integer != expected) {
            report.invalid_tensors += 1;
            return;
        }
    }
    report.matched_tensors += 1;
}

test "Bonsai expected tensor count" {
    try std.testing.expectEqual(@as(u32, 310), 2 + 28 * 11);
}

pub const BonsaiTensorDtype = enum { f16, bf16, u32 };

pub const BonsaiTensorLocation = struct {
    name: []const u8,
    dtype: BonsaiTensorDtype,
    rank: usize,
    dims: [4]usize,
    element_count: usize,
    data_start_absolute: u64,
    data_end_absolute: u64,
};

pub fn locateTensor(io: std.Io, allocator: std.mem.Allocator, path: []const u8, tensor_name: []const u8) !BonsaiTensorLocation {
    const header = try readSafetensorsHeader(io, allocator, path);
    defer allocator.free(header);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidSafetensorsHeader;
    const tensor = parsed.value.object.get(tensor_name) orelse return error.TensorNotFound;
    return parseTensorLocation(tensor_name, tensor, 8 + @as(u64, @intCast(header.len)));
}

pub fn readF16TensorPrefix(io: std.Io, allocator: std.mem.Allocator, path: []const u8, tensor_name: []const u8, out: []f32) !usize {
    return readF16TensorSlice(io, allocator, path, tensor_name, 0, out);
}

pub fn readF16LocationSlice(io: std.Io, allocator: std.mem.Allocator, path: []const u8, location: BonsaiTensorLocation, element_offset: usize, out: []f32) !usize {
    if (location.dtype != .f16) return error.ExpectedF16Tensor;
    if (element_offset > location.element_count) return error.TensorSliceOutOfBounds;
    const count = @min(out.len, location.element_count - element_offset);
    if (count == 0) return 0;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const raw = try allocator.alloc(u8, count * 2);
    defer allocator.free(raw);
    const byte_offset = location.data_start_absolute + @as(u64, @intCast(element_offset * 2));
    if (try file.readPositionalAll(io, raw, byte_offset) != raw.len) return error.TruncatedFile;
    for (out[0..count], 0..) |*dst, index| {
        const bits = std.mem.readInt(u16, raw[index * 2 ..][0..2], .little);
        const half: f16 = @bitCast(bits);
        dst.* = @floatCast(half);
    }
    return count;
}

pub fn readF16TensorSlice(io: std.Io, allocator: std.mem.Allocator, path: []const u8, tensor_name: []const u8, element_offset: usize, out: []f32) !usize {
    const location = try locateTensor(io, allocator, path, tensor_name);
    if (location.dtype != .f16) return error.ExpectedF16Tensor;
    if (element_offset > location.element_count) return error.TensorSliceOutOfBounds;
    const count = @min(out.len, location.element_count - element_offset);
    if (count == 0) return 0;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const raw = try allocator.alloc(u8, count * 2);
    defer allocator.free(raw);
    const byte_offset = location.data_start_absolute + @as(u64, @intCast(element_offset * 2));
    if (try file.readPositionalAll(io, raw, byte_offset) != raw.len) return error.TruncatedFile;
    for (out[0..count], 0..) |*dst, index| {
        const bits = std.mem.readInt(u16, raw[index * 2 ..][0..2], .little);
        const half: f16 = @bitCast(bits);
        dst.* = @floatCast(half);
    }
    return count;
}

pub fn readBf16TensorPrefix(io: std.Io, allocator: std.mem.Allocator, path: []const u8, tensor_name: []const u8, out: []f32) !usize {
    const location = try locateTensor(io, allocator, path, tensor_name);
    return readBf16LocationSlice(io, allocator, path, location, 0, out);
}

pub fn readBf16LocationSlice(io: std.Io, allocator: std.mem.Allocator, path: []const u8, location: BonsaiTensorLocation, element_offset: usize, out: []f32) !usize {
    if (location.dtype != .bf16) return error.ExpectedBF16Tensor;
    if (element_offset > location.element_count) return error.TensorSliceOutOfBounds;
    const count = @min(out.len, location.element_count - element_offset);
    if (count == 0) return 0;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const raw = try allocator.alloc(u8, count * 2);
    defer allocator.free(raw);
    const byte_offset = location.data_start_absolute + @as(u64, @intCast(element_offset * 2));
    if (try file.readPositionalAll(io, raw, byte_offset) != raw.len) return error.TruncatedFile;
    for (out[0..count], 0..) |*dst, index| {
        const raw_bits = std.mem.readInt(u16, raw[index * 2 ..][0..2], .little);
        const bits: u32 = @as(u32, raw_bits) << 16;
        dst.* = @bitCast(bits);
    }
    return count;
}

pub fn readBf16RawLocationSlice(io: std.Io, allocator: std.mem.Allocator, path: []const u8, location: BonsaiTensorLocation, element_offset: usize, out: []u16) !usize {
    if (location.dtype != .bf16) return error.ExpectedBF16Tensor;
    if (element_offset > location.element_count) return error.TensorSliceOutOfBounds;
    const count = @min(out.len, location.element_count - element_offset);
    if (count == 0) return 0;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const raw = try allocator.alloc(u8, count * 2);
    defer allocator.free(raw);
    const byte_offset = location.data_start_absolute + @as(u64, @intCast(element_offset * 2));
    if (try file.readPositionalAll(io, raw, byte_offset) != raw.len) return error.TruncatedFile;
    for (out[0..count], 0..) |*dst, index| {
        dst.* = std.mem.readInt(u16, raw[index * 2 ..][0..2], .little);
    }
    return count;
}

pub fn readU32LocationSlice(io: std.Io, allocator: std.mem.Allocator, path: []const u8, location: BonsaiTensorLocation, element_offset: usize, out: []u32) !usize {
    if (location.dtype != .u32) return error.ExpectedU32Tensor;
    if (element_offset > location.element_count) return error.TensorSliceOutOfBounds;
    const count = @min(out.len, location.element_count - element_offset);
    if (count == 0) return 0;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const raw = try allocator.alloc(u8, count * 4);
    defer allocator.free(raw);
    const byte_offset = location.data_start_absolute + @as(u64, @intCast(element_offset * 4));
    if (try file.readPositionalAll(io, raw, byte_offset) != raw.len) return error.TruncatedFile;
    for (out[0..count], 0..) |*dst, index| {
        dst.* = std.mem.readInt(u32, raw[index * 4 ..][0..4], .little);
    }
    return count;
}

fn parseTensorLocation(name: []const u8, value: std.json.Value, data_base: u64) !BonsaiTensorLocation {
    if (value != .object) return error.InvalidTensorDescriptor;
    const dtype_value = value.object.get("dtype") orelse return error.MissingDtype;
    if (dtype_value != .string) return error.InvalidDtype;
    const dtype: BonsaiTensorDtype = if (std.mem.eql(u8, dtype_value.string, "F16")) .f16 else if (std.mem.eql(u8, dtype_value.string, "BF16")) .bf16 else if (std.mem.eql(u8, dtype_value.string, "U32")) .u32 else return error.UnsupportedDtype;
    const shape_value = value.object.get("shape") orelse return error.MissingShape;
    if (shape_value != .array or shape_value.array.items.len == 0 or shape_value.array.items.len > 4) return error.InvalidShape;
    var dims: [4]usize = .{ 0, 0, 0, 0 };
    var element_count: usize = 1;
    for (shape_value.array.items, 0..) |dim_value, index| {
        if (dim_value != .integer or dim_value.integer <= 0) return error.InvalidShape;
        dims[index] = @intCast(dim_value.integer);
        element_count = try std.math.mul(usize, element_count, dims[index]);
    }
    const offsets_value = value.object.get("data_offsets") orelse return error.MissingDataOffsets;
    if (offsets_value != .array or offsets_value.array.items.len != 2) return error.InvalidDataOffsets;
    const start_value = offsets_value.array.items[0];
    const end_value = offsets_value.array.items[1];
    if (start_value != .integer or end_value != .integer or start_value.integer < 0 or end_value.integer < start_value.integer) return error.InvalidDataOffsets;
    return .{
        .name = name,
        .dtype = dtype,
        .rank = shape_value.array.items.len,
        .dims = dims,
        .element_count = element_count,
        .data_start_absolute = data_base + @as(u64, @intCast(start_value.integer)),
        .data_end_absolute = data_base + @as(u64, @intCast(end_value.integer)),
    };
}

test "read F16 tensor prefix from safetensors bytes" {
    const allocator = std.testing.allocator;
    const header =
        \\{"x":{"dtype":"F16","shape":[2],"data_offsets":[0,4]}}
    ;
    var bytes = std.array_list.Managed(u8).init(allocator);
    defer bytes.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(&len_buf);
    try bytes.appendSlice(header);
    try bytes.appendSlice(&.{ 0x00, 0x3c, 0x00, 0x40 });
    const path = ".zig-cache/bonsai-prefix-test.safetensors";
    var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{ .truncate = true });
    defer file.close(std.testing.io);
    var write_buf: [256]u8 = undefined;
    var writer = file.writer(std.testing.io, &write_buf);
    try writer.interface.writeAll(bytes.items);
    try writer.interface.flush();
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};
    var out: [2]f32 = undefined;
    const n = try readF16TensorPrefix(std.testing.io, allocator, path, "x", &out);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqual(@as(f32, 1.0), out[0]);
    try std.testing.expectEqual(@as(f32, 2.0), out[1]);
    var one: [1]f32 = undefined;
    const m = try readF16TensorSlice(std.testing.io, allocator, path, "x", 1, &one);
    try std.testing.expectEqual(@as(usize, 1), m);
    try std.testing.expectEqual(@as(f32, 2.0), one[0]);
}

test "read BF16 and U32 tensor slices from safetensors bytes" {
    const allocator = std.testing.allocator;
    const header =
        \\{"bf":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},"u":{"dtype":"U32","shape":[2],"data_offsets":[4,12]}}
    ;
    var bytes = std.array_list.Managed(u8).init(allocator);
    defer bytes.deinit();
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try bytes.appendSlice(&len_buf);
    try bytes.appendSlice(header);
    try bytes.appendSlice(&.{ 0x80, 0x3f, 0x00, 0x40 });
    try bytes.appendSlice(&.{ 0x78, 0x56, 0x34, 0x12, 0xf0, 0xde, 0xbc, 0x9a });
    const path = ".zig-cache/bonsai-bf16-u32-test.safetensors";
    var file = try std.Io.Dir.cwd().createFile(std.testing.io, path, .{ .truncate = true });
    defer file.close(std.testing.io);
    var write_buf: [512]u8 = undefined;
    var writer = file.writer(std.testing.io, &write_buf);
    try writer.interface.writeAll(bytes.items);
    try writer.interface.flush();
    defer std.Io.Dir.cwd().deleteFile(std.testing.io, path) catch {};

    var bf: [2]f32 = undefined;
    const bf_count = try readBf16TensorPrefix(std.testing.io, allocator, path, "bf", &bf);
    try std.testing.expectEqual(@as(usize, 2), bf_count);
    try std.testing.expectEqual(@as(f32, 1.0), bf[0]);
    try std.testing.expectEqual(@as(f32, 2.0), bf[1]);

    const loc = try locateTensor(std.testing.io, allocator, path, "u");
    var words: [2]u32 = undefined;
    const word_count = try readU32LocationSlice(std.testing.io, allocator, path, loc, 0, &words);
    try std.testing.expectEqual(@as(usize, 2), word_count);
    try std.testing.expectEqual(@as(u32, 0x12345678), words[0]);
    try std.testing.expectEqual(@as(u32, 0x9abcdef0), words[1]);
}
