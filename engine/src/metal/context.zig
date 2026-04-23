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
