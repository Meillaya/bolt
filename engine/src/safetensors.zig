const std = @import("std");

pub const max_header_bytes: u64 = 16 * 1024 * 1024;
pub const default_max_file_bytes: u64 = 4 * 1024 * 1024 * 1024;
pub const max_rank: usize = 8;

pub const Dtype = enum {
    u8,
    u32,
    f16,
    f32,
    bf16,

    pub fn sizeBytes(self: Dtype) usize {
        return switch (self) {
            .u8 => 1,
            .u32 => 4,
            .f16, .bf16 => 2,
            .f32 => 4,
        };
    }

    pub fn fromString(value: []const u8) !Dtype {
        if (std.mem.eql(u8, value, "U8")) return .u8;
        if (std.mem.eql(u8, value, "U32")) return .u32;
        if (std.mem.eql(u8, value, "F16")) return .f16;
        if (std.mem.eql(u8, value, "F32")) return .f32;
        if (std.mem.eql(u8, value, "BF16")) return .bf16;
        return error.UnsupportedDtype;
    }

    pub fn label(self: Dtype) []const u8 {
        return switch (self) {
            .u8 => "U8",
            .u32 => "U32",
            .f16 => "F16",
            .f32 => "F32",
            .bf16 => "BF16",
        };
    }
};

pub const TensorDescriptor = struct {
    name: []const u8,
    dtype: Dtype,
    rank: usize,
    dims: [max_rank]usize,
    data_start: usize,
    data_end: usize,
    data: []const u8,

    pub fn elementCount(self: TensorDescriptor) !usize {
        if (self.rank == 0 or self.rank > max_rank) return error.InvalidRank;
        var count: usize = 1;
        for (self.dims[0..self.rank]) |dim| {
            if (dim == 0) return error.InvalidShape;
            count = try std.math.mul(usize, count, dim);
        }
        return count;
    }

    pub fn expectedSizeBytes(self: TensorDescriptor) !usize {
        return try std.math.mul(usize, try self.elementCount(), self.dtype.sizeBytes());
    }
};

pub const TensorMetadata = struct {
    name: []const u8,
    dtype: Dtype,
    rank: usize,
    dims: [max_rank]usize,
    data_start: usize,
    data_end: usize,

    pub fn elementCount(self: TensorMetadata) !usize {
        if (self.rank == 0 or self.rank > max_rank) return error.InvalidRank;
        var count: usize = 1;
        for (self.dims[0..self.rank]) |dim| {
            if (dim == 0) return error.InvalidShape;
            count = try std.math.mul(usize, count, dim);
        }
        return count;
    }

    pub fn expectedSizeBytes(self: TensorMetadata) !usize {
        return try std.math.mul(usize, try self.elementCount(), self.dtype.sizeBytes());
    }
};

pub const SafetensorsMetadataReport = struct {
    tensor_count: usize,
    u8_count: usize = 0,
    u32_count: usize = 0,
    f32_count: usize = 0,
    f16_count: usize = 0,
    bf16_count: usize = 0,
};

pub const SafetensorsFile = struct {
    allocator: std.mem.Allocator,
    bytes: []u8,
    header: std.json.Parsed(std.json.Value),
    tensors: []TensorDescriptor,

    pub fn initFromBytes(allocator: std.mem.Allocator, source: []const u8) !SafetensorsFile {
        const owned = try allocator.dupe(u8, source);
        errdefer allocator.free(owned);
        return parseOwned(allocator, owned);
    }

    pub fn initFromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !SafetensorsFile {
        return initFromFileLimited(io, allocator, path, default_max_file_bytes);
    }

    pub fn initFromFileLimited(io: std.Io, allocator: std.mem.Allocator, path: []const u8, max_file_bytes: u64) !SafetensorsFile {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_file_bytes));
        errdefer allocator.free(bytes);
        return parseOwned(allocator, bytes);
    }

    pub fn deinit(self: *SafetensorsFile) void {
        self.allocator.free(self.tensors);
        self.header.deinit();
        self.allocator.free(self.bytes);
        self.* = undefined;
    }

    pub fn getTensor(self: SafetensorsFile, name: []const u8) ?TensorDescriptor {
        for (self.tensors) |tensor| {
            if (std.mem.eql(u8, tensor.name, name)) return tensor;
        }
        return null;
    }

    pub fn tensorCount(self: SafetensorsFile) usize {
        return self.tensors.len;
    }
};

pub fn inspectMetadataFromFile(io: std.Io, allocator: std.mem.Allocator, path: []const u8) !SafetensorsMetadataReport {
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (stat.kind != .file) return error.InvalidSafetensorsFile;
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var header_len_bytes: [8]u8 = undefined;
    const prefix_read = try file.readPositionalAll(io, &header_len_bytes, 0);
    if (prefix_read != header_len_bytes.len) return error.TruncatedFile;
    const header_len_u64 = std.mem.readInt(u64, &header_len_bytes, .little);
    if (header_len_u64 == 0 or header_len_u64 > max_header_bytes) return error.InvalidHeaderLength;
    const data_start_u64 = 8 + header_len_u64;
    if (data_start_u64 > stat.size) return error.InvalidHeaderLength;

    const header_len: usize = @intCast(header_len_u64);
    const header = try allocator.alloc(u8, header_len);
    defer allocator.free(header);
    const header_read = try file.readPositionalAll(io, header, 8);
    if (header_read != header.len) return error.TruncatedFile;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, header, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidHeader;

    const data_region_len: usize = @intCast(stat.size - data_start_u64);
    var report = SafetensorsMetadataReport{ .tensor_count = 0 };
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) continue;
        const meta = try parseTensorMetadata(name, entry.value_ptr.*, data_region_len);
        report.tensor_count += 1;
        switch (meta.dtype) {
            .u8 => report.u8_count += 1,
            .u32 => report.u32_count += 1,
            .f32 => report.f32_count += 1,
            .f16 => report.f16_count += 1,
            .bf16 => report.bf16_count += 1,
        }
    }
    if (report.tensor_count == 0) return error.EmptySafetensorsFile;
    return report;
}

fn parseOwned(allocator: std.mem.Allocator, bytes: []u8) !SafetensorsFile {
    if (bytes.len < 10) return error.TruncatedFile;
    const header_len_u64 = std.mem.readInt(u64, bytes[0..8], .little);
    if (header_len_u64 == 0 or header_len_u64 > max_header_bytes) return error.InvalidHeaderLength;
    const header_len: usize = @intCast(header_len_u64);
    const data_start = 8 + header_len;
    if (data_start > bytes.len) return error.InvalidHeaderLength;

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes[8..data_start], .{ .allocate = .alloc_always });
    errdefer parsed.deinit();
    if (parsed.value != .object) return error.InvalidHeader;

    var descriptors = std.array_list.Managed(TensorDescriptor).init(allocator);
    errdefer descriptors.deinit();

    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) continue;
        const desc = try parseTensorDescriptor(name, entry.value_ptr.*, bytes[data_start..]);
        try descriptors.append(desc);
    }
    if (descriptors.items.len == 0) return error.EmptySafetensorsFile;

    return .{
        .allocator = allocator,
        .bytes = bytes,
        .header = parsed,
        .tensors = try descriptors.toOwnedSlice(),
    };
}

fn parseTensorDescriptor(name: []const u8, value: std.json.Value, data_region: []const u8) !TensorDescriptor {
    const meta = try parseTensorMetadata(name, value, data_region.len);
    const start = meta.data_start;
    const end = meta.data_end;

    var desc = TensorDescriptor{
        .name = meta.name,
        .dtype = meta.dtype,
        .rank = meta.rank,
        .dims = meta.dims,
        .data_start = start,
        .data_end = end,
        .data = data_region[start..end],
    };
    if (try desc.expectedSizeBytes() != desc.data.len) return error.TensorByteSizeMismatch;
    return desc;
}

fn parseTensorMetadata(name: []const u8, value: std.json.Value, data_region_len: usize) !TensorMetadata {
    if (name.len == 0) return error.EmptyTensorName;
    if (value != .object) return error.InvalidTensorDescriptor;
    const object = value.object;

    const dtype_value = object.get("dtype") orelse return error.MissingDtype;
    if (dtype_value != .string) return error.InvalidDtype;
    const dtype = try Dtype.fromString(dtype_value.string);

    const shape_value = object.get("shape") orelse return error.MissingShape;
    if (shape_value != .array) return error.InvalidShape;
    if (shape_value.array.items.len == 0 or shape_value.array.items.len > max_rank) return error.InvalidRank;
    var dims: [max_rank]usize = undefined;
    @memset(&dims, 0);
    for (shape_value.array.items, 0..) |dim_value, index| {
        if (dim_value != .integer) return error.InvalidShape;
        if (dim_value.integer <= 0) return error.InvalidShape;
        dims[index] = @intCast(dim_value.integer);
    }

    const offsets_value = object.get("data_offsets") orelse return error.MissingDataOffsets;
    if (offsets_value != .array or offsets_value.array.items.len != 2) return error.InvalidDataOffsets;
    const start_value = offsets_value.array.items[0];
    const end_value = offsets_value.array.items[1];
    if (start_value != .integer or end_value != .integer) return error.InvalidDataOffsets;
    if (start_value.integer < 0 or end_value.integer < 0) return error.InvalidDataOffsets;
    const start: usize = @intCast(start_value.integer);
    const end: usize = @intCast(end_value.integer);
    if (end < start or end > data_region_len) return error.DataOffsetOutOfBounds;

    var desc = TensorMetadata{
        .name = name,
        .dtype = dtype,
        .rank = shape_value.array.items.len,
        .dims = dims,
        .data_start = start,
        .data_end = end,
    };
    if (try desc.expectedSizeBytes() != end - start) return error.TensorByteSizeMismatch;
    return desc;
}

pub const ModelManifest = struct {
    model_id: []const u8,
    config_file: []const u8,
    tokenizer_id: ?[]const u8 = null,
    safetensors_files: []const []const u8,
};

pub fn parseModelManifestFromSlice(allocator: std.mem.Allocator, input: []const u8) !std.json.Parsed(ModelManifest) {
    return std.json.parseFromSlice(ModelManifest, allocator, input, .{ .allocate = .alloc_always });
}

pub const ModelTensorReport = struct {
    model_id: []const u8,
    tokenizer_id: ?[]const u8,
    file_count: usize,
    tensor_count: usize,
    f32_count: usize,
    f16_count: usize,
    bf16_count: usize,
};

pub fn inspectManifest(io: std.Io, allocator: std.mem.Allocator, base_dir: []const u8, manifest: ModelManifest) !ModelTensorReport {
    if (manifest.model_id.len == 0) return error.EmptyModelId;
    if (manifest.config_file.len == 0) return error.MissingModelConfig;
    if (manifest.safetensors_files.len == 0) return error.MissingSafetensorsFiles;

    var report = ModelTensorReport{
        .model_id = manifest.model_id,
        .tokenizer_id = manifest.tokenizer_id,
        .file_count = manifest.safetensors_files.len,
        .tensor_count = 0,
        .f32_count = 0,
        .f16_count = 0,
        .bf16_count = 0,
    };

    for (manifest.safetensors_files) |relative_path| {
        const path = try std.fs.path.join(allocator, &.{ base_dir, relative_path });
        defer allocator.free(path);
        const metadata = try inspectMetadataFromFile(io, allocator, path);
        report.tensor_count += metadata.tensor_count;
        report.f32_count += metadata.f32_count;
        report.f16_count += metadata.f16_count;
        report.bf16_count += metadata.bf16_count;
    }
    return report;
}

test "parse safetensors header from bytes" {
    const json =
        \\{"tensor_a":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}
    ;
    const file_bytes = buildTestFile(json, 24);
    var file = try SafetensorsFile.initFromBytes(std.testing.allocator, &file_bytes);
    defer file.deinit();
    try std.testing.expectEqual(@as(usize, 1), file.tensorCount());
    const tensor = file.getTensor("tensor_a") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(Dtype.f32, tensor.dtype);
    try std.testing.expectEqual(@as(usize, 2), tensor.rank);
    try std.testing.expectEqual(@as(usize, 2), tensor.dims[0]);
    try std.testing.expectEqual(@as(usize, 3), tensor.dims[1]);
    try std.testing.expectEqual(@as(usize, 24), tensor.data.len);
    try std.testing.expectEqual(@as(usize, 6), try tensor.elementCount());
}

test "metadata is skipped and dtype coverage matches reference subset" {
    const json =
        \\{"__metadata__":{"format":"pt"},"a":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"b":{"dtype":"F16","shape":[2],"data_offsets":[4,8]},"c":{"dtype":"BF16","shape":[2],"data_offsets":[8,12]}}
    ;
    const file_bytes = buildTestFile(json, 12);
    var file = try SafetensorsFile.initFromBytes(std.testing.allocator, &file_bytes);
    defer file.deinit();
    try std.testing.expectEqual(@as(usize, 3), file.tensorCount());
    try std.testing.expect(file.getTensor("__metadata__") == null);
    try std.testing.expectEqual(Dtype.u8, (file.getTensor("a") orelse return error.TestUnexpectedResult).dtype);
    try std.testing.expectEqual(Dtype.f16, (file.getTensor("b") orelse return error.TestUnexpectedResult).dtype);
    try std.testing.expectEqual(Dtype.bf16, (file.getTensor("c") orelse return error.TestUnexpectedResult).dtype);
}

test "malformed safetensors fail with named errors" {
    var too_short = [_]u8{0} ** 9;
    try std.testing.expectError(error.TruncatedFile, SafetensorsFile.initFromBytes(std.testing.allocator, &too_short));

    const bad_dtype_json =
        \\{"bad":{"dtype":"I32","shape":[1],"data_offsets":[0,4]}}
    ;
    const bad_dtype = buildTestFile(bad_dtype_json, 4);
    try std.testing.expectError(error.UnsupportedDtype, SafetensorsFile.initFromBytes(std.testing.allocator, &bad_dtype));

    const bad_offsets_json =
        \\{"bad":{"dtype":"F32","shape":[2],"data_offsets":[0,99]}}
    ;
    const bad_offsets = buildTestFile(bad_offsets_json, 8);
    try std.testing.expectError(error.DataOffsetOutOfBounds, SafetensorsFile.initFromBytes(std.testing.allocator, &bad_offsets));
}

fn buildTestFile(comptime json: []const u8, comptime data_size: usize) [8 + json.len + data_size]u8 {
    var out: [8 + json.len + data_size]u8 = undefined;
    std.mem.writeInt(u64, out[0..8], json.len, .little);
    @memcpy(out[8..][0..json.len], json);
    @memset(out[8 + json.len ..], 0);
    return out;
}
