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
extern fn bolt_metal_buffer_destroy(handle: *anyopaque) void;
extern fn bolt_metal_buffer_contents_f32(handle: *anyopaque) ?[*]f32;
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
        const handle = self.handle orelse @panic("metal shared buffer is not initialized");
        const values = bolt_metal_buffer_contents_f32(handle) orelse @panic("metal shared buffer has no CPU-visible contents");
        return values[0..self.len];
    }

    pub fn write(self: SharedBufferF32, values: []const f32) !void {
        if (values.len != self.len) return error.LengthMismatch;
        @memcpy(self.asSlice(), values);
    }

    pub fn materialize(self: SharedBufferF32, allocator: std.mem.Allocator) ![]f32 {
        const owned = try allocator.alloc(f32, self.len);
        errdefer allocator.free(owned);
        @memcpy(owned, self.asSlice());
        return owned;
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

    pub fn copySharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
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

    pub fn biasAddSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        bias: SharedBufferF32,
        output: SharedBufferF32,
        rows: usize,
        cols: usize,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
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

    pub fn reduceSumSharedBufferF32(
        self: Context,
        input: SharedBufferF32,
        output: SharedBufferF32,
    ) !void {
        if (self.handle == null) return error.UninitializedContext;
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
