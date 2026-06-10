const std = @import("std");

extern fn bolt_metal_context_is_available() bool;
extern fn bolt_metal_string_destroy(message: ?[*:0]u8) void;
extern fn bolt_metal_context_create(
    source: [*]const u8,
    source_len: usize,
    error_out: *?[*:0]u8,
) ?*anyopaque;
extern fn bolt_metal_context_destroy(handle: *anyopaque) void;
extern fn bolt_metal_buffer_create(
    context_handle: *anyopaque,
    count: usize,
    error_out: *?[*:0]u8,
) ?*anyopaque;
extern fn bolt_metal_buffer_create_bytes(
    context_handle: *anyopaque,
    byte_count: usize,
    error_out: *?[*:0]u8,
) ?*anyopaque;
extern fn bolt_metal_buffer_destroy(handle: *anyopaque) void;
extern fn bolt_metal_buffer_contents_f32(handle: *anyopaque) ?[*]f32;
extern fn bolt_metal_buffer_contents_u8(handle: *anyopaque) ?[*]u8;
extern fn bolt_metal_buffer_byte_length(handle: *anyopaque) usize;
extern fn bolt_metal_buffer_validate_bytes(
    handle: *anyopaque,
    byte_count: usize,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_context_has_kernel(
    context_handle: *anyopaque,
    name: [*]const u8,
    name_len: usize,
) bool;
extern fn bolt_metal_validate_dispatch_dimensions(
    grid_x: usize,
    grid_y: usize,
    grid_z: usize,
    threads_x: usize,
    threads_y: usize,
    threads_z: usize,
    max_threads_per_threadgroup: usize,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_copy_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_add_buffer_f32(
    context_handle: *anyopaque,
    left_handle: *anyopaque,
    right_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_matmul_buffer_f32(
    context_handle: *anyopaque,
    left_handle: *anyopaque,
    right_handle: *anyopaque,
    rows: usize,
    cols: usize,
    inner: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_qmv_buffer_f32(
    context_handle: *anyopaque,
    packed_bits_handle: *anyopaque,
    scales_handle: *anyopaque,
    input_handle: *anyopaque,
    rows: usize,
    cols: usize,
    group_size: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_q4mv_buffer_f32(
    context_handle: *anyopaque,
    packed_nibbles_handle: *anyopaque,
    scales_handle: *anyopaque,
    biases_handle: *anyopaque,
    input_handle: *anyopaque,
    rows: usize,
    cols: usize,
    group_size: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_bias_add_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    bias_handle: *anyopaque,
    rows: usize,
    cols: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_relu_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_sigmoid_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_tanh_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_reduce_sum_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;
extern fn bolt_metal_softmax_buffer_f32(
    context_handle: *anyopaque,
    input_handle: *anyopaque,
    count: usize,
    output_handle: *anyopaque,
    error_out: *?[*:0]u8,
) bool;

const embedded_kernels = @embedFile("kernels.metal");

pub const DispatchDimensions = struct {
    grid_x: usize,
    grid_y: usize = 1,
    grid_z: usize = 1,
    threads_x: usize,
    threads_y: usize = 1,
    threads_z: usize = 1,
};

pub const SharedBufferF32 = struct {
    handle: ?*anyopaque = null,
    len: usize = 0,

    pub fn deinit(self: *SharedBufferF32) void {
        if (self.handle) |handle| {
            bolt_metal_buffer_destroy(handle);
            self.handle = null;
            self.len = 0;
        }
    }

    pub fn asSlice(self: SharedBufferF32) []f32 {
        const values = self.slice() catch @panic("metal shared buffer has no CPU-visible contents");
        return values[0..self.len];
    }

    pub fn slice(self: SharedBufferF32) ![]f32 {
        const handle = self.handle orelse return error.UninitializedBuffer;
        const values = bolt_metal_buffer_contents_f32(handle) orelse return error.MetalBufferContentsUnavailable;
        return values[0..self.len];
    }

    pub fn write(self: SharedBufferF32, values: []const f32) !void {
        if (values.len != self.len) return error.LengthMismatch;
        @memcpy(try self.slice(), values);
    }

    pub fn materialize(self: SharedBufferF32, allocator: std.mem.Allocator) ![]f32 {
        const owned = try allocator.alloc(f32, self.len);
        errdefer allocator.free(owned);
        @memcpy(owned, try self.slice());
        return owned;
    }
};

pub const SharedBufferBytes = struct {
    handle: ?*anyopaque = null,
    byte_len: usize = 0,

    pub fn deinit(self: *SharedBufferBytes) void {
        if (self.handle) |handle| {
            bolt_metal_buffer_destroy(handle);
            self.handle = null;
            self.byte_len = 0;
        }
    }

    pub fn slice(self: SharedBufferBytes) ![]u8 {
        const handle = self.handle orelse return error.UninitializedBuffer;
        const values = bolt_metal_buffer_contents_u8(handle) orelse return error.MetalBufferContentsUnavailable;
        const actual_byte_len = bolt_metal_buffer_byte_length(handle);
        if (actual_byte_len < self.byte_len) return error.LengthMismatch;
        return values[0..self.byte_len];
    }

    pub fn write(self: SharedBufferBytes, values: []const u8) !void {
        if (values.len != self.byte_len) return error.LengthMismatch;
        @memcpy(try self.slice(), values);
    }

    pub fn validateBinding(self: SharedBufferBytes, byte_count: usize) !void {
        const handle = self.handle orelse return error.UninitializedBuffer;
        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_buffer_validate_bytes(handle, byte_count, &error_message);
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalArgumentValidationFailed;
    }
};

pub const HalfBuffer = struct {
    bytes: SharedBufferBytes,
    len: usize,

    pub fn deinit(self: *HalfBuffer) void {
        self.bytes.deinit();
        self.len = 0;
    }

    pub fn writeRaw(self: HalfBuffer, values: []const u16) !void {
        if (values.len != self.len) return error.LengthMismatch;
        const raw = std.mem.sliceAsBytes(values);
        try self.bytes.write(raw);
    }

    pub fn materializeRaw(self: HalfBuffer, allocator: std.mem.Allocator) ![]u16 {
        const out = try allocator.alloc(u16, self.len);
        errdefer allocator.free(out);
        const raw = try self.bytes.slice();
        @memcpy(std.mem.sliceAsBytes(out), raw);
        return out;
    }
};

pub const PackedBuffer = struct {
    bytes: SharedBufferBytes,
    word_count: usize,

    pub fn deinit(self: *PackedBuffer) void {
        self.bytes.deinit();
        self.word_count = 0;
    }

    pub fn writeWords(self: PackedBuffer, values: []const u32) !void {
        if (values.len != self.word_count) return error.LengthMismatch;
        try self.bytes.write(std.mem.sliceAsBytes(values));
    }

    pub fn materializeWords(self: PackedBuffer, allocator: std.mem.Allocator) ![]u32 {
        const out = try allocator.alloc(u32, self.word_count);
        errdefer allocator.free(out);
        @memcpy(std.mem.sliceAsBytes(out), try self.bytes.slice());
        return out;
    }
};

pub const Q4Buffer = struct {
    packed_nibbles: SharedBufferBytes,
    scales_bf16: HalfBuffer,
    biases_bf16: HalfBuffer,
    rows: usize,
    cols: usize,
    group_size: usize,

    pub fn deinit(self: *Q4Buffer) void {
        self.packed_nibbles.deinit();
        self.scales_bf16.deinit();
        self.biases_bf16.deinit();
        self.rows = 0;
        self.cols = 0;
        self.group_size = 0;
    }
};

pub const Context = struct {
    backend_name: []const u8 = "metal",
    handle: ?*anyopaque = null,

    pub fn isMetalBackend(self: Context) bool {
        return std.mem.eql(u8, self.backend_name, "metal");
    }

    pub fn isAvailable() bool {
        return bolt_metal_context_is_available();
    }

    pub fn init() !Context {
        if (!isAvailable()) return error.MetalUnavailable;

        var error_message: ?[*:0]u8 = null;
        const handle = bolt_metal_context_create(
            embedded_kernels.ptr,
            embedded_kernels.len,
            &error_message,
        ) orelse {
            defer bolt_metal_string_destroy(error_message);
            return error.MetalInitializationFailed;
        };

        return .{
            .backend_name = "metal",
            .handle = handle,
        };
    }

    pub fn deinit(self: *Context) void {
        if (self.handle) |handle| {
            bolt_metal_context_destroy(handle);
            self.handle = null;
        }
    }

    pub fn createSharedBufferF32(self: Context, len: usize) !SharedBufferF32 {
        if (self.handle == null) return error.UninitializedContext;
        if (len == 0) return error.EmptyBuffer;

        var error_message: ?[*:0]u8 = null;
        const handle = bolt_metal_buffer_create(self.handle.?, len, &error_message) orelse {
            defer bolt_metal_string_destroy(error_message);
            return error.MetalBufferAllocationFailed;
        };
        return .{
            .handle = handle,
            .len = len,
        };
    }

    pub fn createSharedBufferBytes(self: Context, byte_len: usize) !SharedBufferBytes {
        if (self.handle == null) return error.UninitializedContext;
        if (byte_len == 0) return error.EmptyBuffer;

        var error_message: ?[*:0]u8 = null;
        const handle = bolt_metal_buffer_create_bytes(self.handle.?, byte_len, &error_message) orelse {
            defer bolt_metal_string_destroy(error_message);
            return error.MetalBufferAllocationFailed;
        };
        return .{
            .handle = handle,
            .byte_len = byte_len,
        };
    }

    pub fn createHalfBuffer(self: Context, len: usize) !HalfBuffer {
        return .{
            .bytes = try self.createSharedBufferBytes(len * @sizeOf(u16)),
            .len = len,
        };
    }

    pub fn createPackedBuffer(self: Context, word_count: usize) !PackedBuffer {
        return .{
            .bytes = try self.createSharedBufferBytes(word_count * @sizeOf(u32)),
            .word_count = word_count,
        };
    }

    pub fn createQ4Buffer(self: Context, rows: usize, cols: usize, group_size: usize) !Q4Buffer {
        if (rows == 0 or cols == 0 or group_size == 0) return error.EmptyBuffer;
        if (cols % group_size != 0) return error.InvalidGroupSize;
        const packed_bytes = rows * ((cols + 1) / 2);
        const group_count = rows * (cols / group_size);
        var packed_nibbles = try self.createSharedBufferBytes(packed_bytes);
        errdefer packed_nibbles.deinit();
        var scales_bf16 = try self.createHalfBuffer(group_count);
        errdefer scales_bf16.deinit();
        var biases_bf16 = try self.createHalfBuffer(group_count);
        errdefer biases_bf16.deinit();
        return .{
            .packed_nibbles = packed_nibbles,
            .scales_bf16 = scales_bf16,
            .biases_bf16 = biases_bf16,
            .rows = rows,
            .cols = cols,
            .group_size = group_size,
        };
    }

    pub fn hasKernel(self: Context, name: []const u8) bool {
        const handle = self.handle orelse return false;
        return bolt_metal_context_has_kernel(handle, name.ptr, name.len);
    }

    pub fn validateDispatchDimensions(_: Context, dims: DispatchDimensions, max_threads_per_threadgroup: usize) !void {
        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_validate_dispatch_dimensions(
            dims.grid_x,
            dims.grid_y,
            dims.grid_z,
            dims.threads_x,
            dims.threads_y,
            dims.threads_z,
            max_threads_per_threadgroup,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.InvalidDispatchDimensions;
    }

    pub fn copySharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_copy_buffer_f32(
            self.handle.?,
            input.handle.?,
            input.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn addSharedBufferF32(
        self: Context,
        left: SharedBufferF32,
        right: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (left.handle == null or right.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (left.len != right.len or left.len != output.len) return error.LengthMismatch;
        if (left.len == 0) return;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_add_buffer_f32(
            self.handle.?,
            left.handle.?,
            right.handle.?,
            left.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn matMulSharedBufferF32(
        self: Context,
        left: SharedBufferF32,
        right: SharedBufferF32,
        output: SharedBufferF32,
        rows: usize,
        cols: usize,
        inner: usize,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (left.handle == null or right.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (rows == 0 or cols == 0 or inner == 0) return error.EmptyBuffer;
        if (left.len != rows * inner or right.len != inner * cols or output.len != rows * cols) {
            return error.LengthMismatch;
        }

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_matmul_buffer_f32(
            self.handle.?,
            left.handle.?,
            right.handle.?,
            rows,
            cols,
            inner,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn qmvSharedBufferF32(
        self: Context,
        packed_bits: SharedBufferBytes,
        scales: HalfBuffer,
        input: SharedBufferF32,
        output: SharedBufferF32,
        rows: usize,
        cols: usize,
        group_size: usize,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (packed_bits.handle == null or scales.bytes.handle == null or input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (rows == 0 or cols == 0 or group_size == 0) return error.EmptyBuffer;
        if (cols % group_size != 0 or cols % 32 != 0) return error.InvalidGroupSize;
        if (packed_bits.byte_len < rows * (cols / 8) or scales.len < rows * (cols / group_size) or input.len != cols or output.len != rows) {
            return error.LengthMismatch;
        }

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_qmv_buffer_f32(
            self.handle.?,
            packed_bits.handle.?,
            scales.bytes.handle.?,
            input.handle.?,
            rows,
            cols,
            group_size,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn q4mvSharedBufferF32(
        self: Context,
        q4: Q4Buffer,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (q4.packed_nibbles.handle == null or q4.scales_bf16.bytes.handle == null or q4.biases_bf16.bytes.handle == null or input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (q4.rows == 0 or q4.cols == 0 or q4.group_size == 0) return error.EmptyBuffer;
        if (q4.cols % q4.group_size != 0 or q4.cols % 2 != 0) return error.InvalidGroupSize;
        if (input.len != q4.cols or output.len != q4.rows) return error.LengthMismatch;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_q4mv_buffer_f32(
            self.handle.?,
            q4.packed_nibbles.handle.?,
            q4.scales_bf16.bytes.handle.?,
            q4.biases_bf16.bytes.handle.?,
            input.handle.?,
            q4.rows,
            q4.cols,
            q4.group_size,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn biasAddSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        bias: SharedBufferF32,
        output: SharedBufferF32,
        rows: usize,
        cols: usize,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or bias.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (rows == 0 or cols == 0) return error.EmptyBuffer;
        if (input.len != rows * cols or bias.len != cols or output.len != rows * cols) {
            return error.LengthMismatch;
        }

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_bias_add_buffer_f32(
            self.handle.?,
            input.handle.?,
            bias.handle.?,
            rows,
            cols,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn reluSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_relu_buffer_f32(
            self.handle.?,
            input.handle.?,
            input.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn sigmoidSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_sigmoid_buffer_f32(
            self.handle.?,
            input.handle.?,
            input.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn tanhSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_tanh_buffer_f32(
            self.handle.?,
            input.handle.?,
            input.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn reduceSumSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (input.len == 0) return error.EmptyBuffer;
        if (output.len != 1) return error.LengthMismatch;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_reduce_sum_buffer_f32(
            self.handle.?,
            input.handle.?,
            input.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn softmaxSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
        if (input.handle == null or output.handle == null) return error.UninitializedBuffer;
        if (input.len == 0) return error.EmptyBuffer;
        if (input.len != output.len) return error.LengthMismatch;

        var error_message: ?[*:0]u8 = null;
        const ok = bolt_metal_softmax_buffer_f32(
            self.handle.?,
            input.handle.?,
            input.len,
            output.handle.?,
            &error_message,
        );
        defer bolt_metal_string_destroy(error_message);
        if (!ok) return error.MetalExecutionFailed;
    }

    pub fn roundTripF32(
        self: Context,
        allocator: std.mem.Allocator,
        input: []const f32,
    ) ![]f32 {
        if (input.len == 0) return allocator.alloc(f32, 0);

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(input.len);
        defer output_buffer.deinit();
        try self.copySharedBufferF32(input_buffer, output_buffer);
        return output_buffer.materialize(allocator);
    }

    pub fn copyF32(
        self: Context,
        input: []const f32,
        output: []f32,
    ) !void {
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return;

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.copySharedBufferF32(input_buffer, output_buffer);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn addF32(
        self: Context,
        left: []const f32,
        right: []const f32,
        output: []f32,
    ) !void {
        if (left.len != right.len or left.len != output.len) return error.LengthMismatch;
        if (left.len == 0) return;

        var left_buffer = try self.createSharedBufferF32(left.len);
        defer left_buffer.deinit();
        try left_buffer.write(left);

        var right_buffer = try self.createSharedBufferF32(right.len);
        defer right_buffer.deinit();
        try right_buffer.write(right);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.addSharedBufferF32(left_buffer, right_buffer, output_buffer);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn matMulF32(
        self: Context,
        left: []const f32,
        right: []const f32,
        output: []f32,
        rows: usize,
        cols: usize,
        inner: usize,
    ) !void {
        if (rows == 0 or cols == 0 or inner == 0) return error.EmptyBuffer;
        if (left.len != rows * inner or right.len != inner * cols or output.len != rows * cols) {
            return error.LengthMismatch;
        }

        var left_buffer = try self.createSharedBufferF32(left.len);
        defer left_buffer.deinit();
        try left_buffer.write(left);

        var right_buffer = try self.createSharedBufferF32(right.len);
        defer right_buffer.deinit();
        try right_buffer.write(right);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.matMulSharedBufferF32(left_buffer, right_buffer, output_buffer, rows, cols, inner);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn biasAddF32(
        self: Context,
        input: []const f32,
        bias: []const f32,
        output: []f32,
        rows: usize,
        cols: usize,
    ) !void {
        if (rows == 0 or cols == 0) return error.EmptyBuffer;
        if (input.len != rows * cols or bias.len != cols or output.len != rows * cols) {
            return error.LengthMismatch;
        }

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var bias_buffer = try self.createSharedBufferF32(bias.len);
        defer bias_buffer.deinit();
        try bias_buffer.write(bias);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.biasAddSharedBufferF32(input_buffer, bias_buffer, output_buffer, rows, cols);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn reluF32(
        self: Context,
        input: []const f32,
        output: []f32,
    ) !void {
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.reluSharedBufferF32(input_buffer, output_buffer);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn sigmoidF32(
        self: Context,
        input: []const f32,
        output: []f32,
    ) !void {
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.sigmoidSharedBufferF32(input_buffer, output_buffer);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn tanhF32(
        self: Context,
        input: []const f32,
        output: []f32,
    ) !void {
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.tanhSharedBufferF32(input_buffer, output_buffer);
        @memcpy(output, output_buffer.asSlice());
    }

    pub fn reduceSumF32(
        self: Context,
        input: []const f32,
    ) !f32 {
        if (input.len == 0) return error.EmptyBuffer;

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(1);
        defer output_buffer.deinit();
        try self.reduceSumSharedBufferF32(input_buffer, output_buffer);
        return output_buffer.asSlice()[0];
    }

    pub fn softmaxF32(
        self: Context,
        input: []const f32,
        output: []f32,
    ) !void {
        if (input.len != output.len) return error.LengthMismatch;
        if (input.len == 0) return error.EmptyBuffer;

        var input_buffer = try self.createSharedBufferF32(input.len);
        defer input_buffer.deinit();
        try input_buffer.write(input);

        var output_buffer = try self.createSharedBufferF32(output.len);
        defer output_buffer.deinit();
        try self.softmaxSharedBufferF32(input_buffer, output_buffer);
        @memcpy(output, output_buffer.asSlice());
    }
};

test "context defaults to metal backend" {
    const context = Context{};
    try std.testing.expect(context.isMetalBackend());
}

test "metal context allocates shared buffers with CPU-visible slices" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var buffer = try context.createSharedBufferF32(3);
    defer buffer.deinit();

    try buffer.write(&.{ 1.0, 2.5, 4.0 });
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.5, 4.0 }, buffer.asSlice());
}

test "metal context copies floats through shared buffers and a real compute pipeline" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var input_buffer = try context.createSharedBufferF32(3);
    defer input_buffer.deinit();
    try input_buffer.write(&.{ 1.0, 2.5, 4.0 });

    var output_buffer = try context.createSharedBufferF32(3);
    defer output_buffer.deinit();
    try context.copySharedBufferF32(input_buffer, output_buffer);

    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.5, 4.0 }, output_buffer.asSlice());
}

test "metal context adds float buffers through shared buffers and a real compute pipeline" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var left_buffer = try context.createSharedBufferF32(4);
    defer left_buffer.deinit();
    try left_buffer.write(&.{ 0.1, 0.2, 0.9, 0.4 });

    var right_buffer = try context.createSharedBufferF32(4);
    defer right_buffer.deinit();
    try right_buffer.write(&.{ 0.0, 0.8, 0.0, 0.1 });

    var output_buffer = try context.createSharedBufferF32(4);
    defer output_buffer.deinit();
    try context.addSharedBufferF32(left_buffer, right_buffer, output_buffer);

    try std.testing.expectEqualSlices(f32, &.{ 0.1, 1.0, 0.9, 0.5 }, output_buffer.asSlice());
}

test "metal context runs matrix multiply through shared buffers" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var left_buffer = try context.createSharedBufferF32(6);
    defer left_buffer.deinit();
    try left_buffer.write(&.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    var right_buffer = try context.createSharedBufferF32(6);
    defer right_buffer.deinit();
    try right_buffer.write(&.{ 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 });

    var output_buffer = try context.createSharedBufferF32(4);
    defer output_buffer.deinit();
    try context.matMulSharedBufferF32(left_buffer, right_buffer, output_buffer, 2, 2, 3);

    try std.testing.expectEqualSlices(f32, &.{ 58.0, 64.0, 139.0, 154.0 }, output_buffer.asSlice());
}

test "metal context runs bias add and relu through shared buffers" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var input_buffer = try context.createSharedBufferF32(6);
    defer input_buffer.deinit();
    try input_buffer.write(&.{ -1.0, 2.0, -3.0, 4.0, 5.0, -6.0 });

    var bias_buffer = try context.createSharedBufferF32(3);
    defer bias_buffer.deinit();
    try bias_buffer.write(&.{ 1.0, -2.0, 3.0 });

    var biased_buffer = try context.createSharedBufferF32(6);
    defer biased_buffer.deinit();
    try context.biasAddSharedBufferF32(input_buffer, bias_buffer, biased_buffer, 2, 3);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 0.0, 0.0, 5.0, 3.0, -3.0 }, biased_buffer.asSlice());

    var relu_buffer = try context.createSharedBufferF32(6);
    defer relu_buffer.deinit();
    try context.reluSharedBufferF32(biased_buffer, relu_buffer);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 0.0, 0.0, 5.0, 3.0, 0.0 }, relu_buffer.asSlice());
}

test "metal context runs sigmoid and tanh through shared buffers" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var input_buffer = try context.createSharedBufferF32(3);
    defer input_buffer.deinit();
    try input_buffer.write(&.{ -1.0, 0.0, 1.0 });

    var sigmoid_buffer = try context.createSharedBufferF32(3);
    defer sigmoid_buffer.deinit();
    try context.sigmoidSharedBufferF32(input_buffer, sigmoid_buffer);
    const sigmoid = sigmoid_buffer.asSlice();
    try std.testing.expectApproxEqAbs(@as(f32, 0.268941), sigmoid[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.731059), sigmoid[2], 0.0001);

    var tanh_buffer = try context.createSharedBufferF32(3);
    defer tanh_buffer.deinit();
    try context.tanhSharedBufferF32(input_buffer, tanh_buffer);
    const tanh_values = tanh_buffer.asSlice();
    try std.testing.expectApproxEqAbs(@as(f32, -0.761594), tanh_values[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), tanh_values[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.761594), tanh_values[2], 0.0001);
}

test "metal context runs reduction and softmax through shared buffers" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var input_buffer = try context.createSharedBufferF32(4);
    defer input_buffer.deinit();
    try input_buffer.write(&.{ 1.0, 2.0, 3.0, 4.0 });

    var sum_buffer = try context.createSharedBufferF32(1);
    defer sum_buffer.deinit();
    try context.reduceSumSharedBufferF32(input_buffer, sum_buffer);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), sum_buffer.asSlice()[0], 0.0001);

    var softmax_buffer = try context.createSharedBufferF32(4);
    defer softmax_buffer.deinit();
    try context.softmaxSharedBufferF32(input_buffer, softmax_buffer);
    const softmax = softmax_buffer.asSlice();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0320586), softmax[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0871443), softmax[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.236883), softmax[2], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.643914), softmax[3], 0.0001);
}

test "metal shared buffer invalid states are recoverable errors" {
    const empty = SharedBufferF32{};
    try std.testing.expectError(error.UninitializedBuffer, empty.slice());
    try std.testing.expectError(error.UninitializedBuffer, empty.materialize(std.testing.allocator));

    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    var buffer = try context.createSharedBufferF32(2);
    defer buffer.deinit();
    try std.testing.expectError(error.LengthMismatch, buffer.write(&.{ 1.0, 2.0, 3.0 }));
}

test "metal byte, half, packed, and q4 buffers round-trip shared storage" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var context = try Context.init();
    defer context.deinit();

    var bytes = try context.createSharedBufferBytes(4);
    defer bytes.deinit();
    try bytes.write(&.{ 0xaa, 0xbb, 0xcc, 0xdd });
    try bytes.validateBinding(4);
    try std.testing.expectError(error.MetalArgumentValidationFailed, bytes.validateBinding(5));
    try std.testing.expectEqualSlices(u8, &.{ 0xaa, 0xbb, 0xcc, 0xdd }, try bytes.slice());

    var half = try context.createHalfBuffer(3);
    defer half.deinit();
    try half.writeRaw(&.{ 0x3c00, 0x4000, 0x4200 });
    const half_copy = try half.materializeRaw(allocator);
    defer allocator.free(half_copy);
    try std.testing.expectEqualSlices(u16, &.{ 0x3c00, 0x4000, 0x4200 }, half_copy);

    var packed_buffer = try context.createPackedBuffer(2);
    defer packed_buffer.deinit();
    try packed_buffer.writeWords(&.{ 0x01234567, 0x89abcdef });
    const packed_copy = try packed_buffer.materializeWords(allocator);
    defer allocator.free(packed_copy);
    try std.testing.expectEqualSlices(u32, &.{ 0x01234567, 0x89abcdef }, packed_copy);

    var q4 = try context.createQ4Buffer(2, 8, 4);
    defer q4.deinit();
    try q4.packed_nibbles.write(&.{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe });
    try q4.scales_bf16.writeRaw(&.{ 0x3f80, 0x4000, 0x4040, 0x4080 });
    try q4.biases_bf16.writeRaw(&.{ 0x0000, 0x3f00, 0x3f80, 0x4000 });
    try std.testing.expectEqual(@as(usize, 2), q4.rows);
    try std.testing.expectEqual(@as(usize, 8), q4.cols);
    try std.testing.expectEqual(@as(usize, 4), q4.group_size);
}

test "metal bridge validates kernel lookup and dispatch dimensions" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();

    try std.testing.expect(context.hasKernel("add_f32"));
    try std.testing.expect(context.hasKernel("matmul_f32"));
    try std.testing.expect(!context.hasKernel("missing_kernel"));

    try context.validateDispatchDimensions(.{
        .grid_x = 16,
        .threads_x = 16,
    }, 1024);
    try std.testing.expectError(error.InvalidDispatchDimensions, context.validateDispatchDimensions(.{
        .grid_x = 0,
        .threads_x = 16,
    }, 1024));
    try std.testing.expectError(error.InvalidDispatchDimensions, context.validateDispatchDimensions(.{
        .grid_x = 16,
        .threads_x = 2048,
    }, 1024));
}

test "metal qmv dispatch executes reference q1 contract" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();
    try std.testing.expect(context.hasKernel("qmv"));

    var packed_bits = try context.createSharedBufferBytes(2 * (128 / 8));
    defer packed_bits.deinit();
    var packed_data = [_]u8{0} ** (2 * (128 / 8));
    @memset(packed_data[0 .. 128 / 8], 0xff);
    @memset(packed_data[128 / 8 ..], 0x00);
    try packed_bits.write(&packed_data);

    var scales = try context.createHalfBuffer(2);
    defer scales.deinit();
    try scales.writeRaw(&.{ 0x3c00, 0x3c00 }); // f16(1.0), f16(1.0)

    var input = try context.createSharedBufferF32(128);
    defer input.deinit();
    var input_values = [_]f32{1.0} ** 128;
    try input.write(&input_values);

    var output = try context.createSharedBufferF32(2);
    defer output.deinit();
    try context.qmvSharedBufferF32(packed_bits, scales, input, output, 2, 128, 128);
    const values = output.asSlice();
    try std.testing.expectApproxEqAbs(@as(f32, 128.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -128.0), values[1], 0.001);
}

test "metal q4mv dispatch executes MLX affine q4 contract" {
    if (!Context.isAvailable()) return error.SkipZigTest;

    var context = try Context.init();
    defer context.deinit();
    try std.testing.expect(context.hasKernel("q4mv_f32"));

    var q4 = try context.createQ4Buffer(2, 8, 4);
    defer q4.deinit();
    // Row 0 nibbles: 0..7. Row 1 nibbles: 8..15.
    try q4.packed_nibbles.write(&.{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe });
    // BF16 raw: row0 groups scale 1 and 2; row1 groups scale 3 and 4.
    try q4.scales_bf16.writeRaw(&.{ 0x3f80, 0x4000, 0x4040, 0x4080 });
    // BF16 raw: row0 biases 0 and 0.5; row1 biases 1 and 2.
    try q4.biases_bf16.writeRaw(&.{ 0x0000, 0x3f00, 0x3f80, 0x4000 });

    var input = try context.createSharedBufferF32(8);
    defer input.deinit();
    try input.write(&.{ 1, 1, 1, 1, 1, 1, 1, 1 });

    var output = try context.createSharedBufferF32(2);
    defer output.deinit();
    try context.q4mvSharedBufferF32(q4, input, output);
    const values = output.asSlice();
    try std.testing.expectApproxEqAbs(@as(f32, 52.0), values[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 342.0), values[1], 0.001);
}
