const std = @import("std");
const layout = @import("layout.zig");
const metal = @import("../metal/context.zig");

pub const Storage = enum {
    host,
    metal_shared,
};

pub const OwnedTensorF32 = struct {
    allocator: std.mem.Allocator,
    shape: layout.Shape,
    values: []f32,
    storage: Storage,
    host_values: ?[]f32 = null,
    metal_buffer: ?metal.SharedBufferF32 = null,

    pub fn initCopy(
        allocator: std.mem.Allocator,
        shape: layout.Shape,
        values: []const f32,
        storage: Storage,
    ) !OwnedTensorF32 {
        if (shape.elementCount() != values.len) return error.LengthMismatch;

        return switch (storage) {
            .host => initHostCopy(allocator, shape, values),
            .metal_shared => initMetalSharedCopy(allocator, shape, values),
        };
    }

    pub fn deinit(self: *OwnedTensorF32) void {
        if (self.host_values) |host_values| {
            self.allocator.free(host_values);
        }
        if (self.metal_buffer) |*metal_buffer| {
            metal_buffer.deinit();
        }
        self.* = undefined;
    }

    pub fn elementCount(self: OwnedTensorF32) usize {
        return self.shape.elementCount();
    }

    pub fn materializeToHost(self: OwnedTensorF32) !OwnedTensorF32 {
        return initHostCopy(self.allocator, self.shape, self.values);
    }

    pub fn roundTripThroughMetal(self: OwnedTensorF32) !OwnedTensorF32 {
        var context = try metal.Context.init();
        defer context.deinit();

        var input_buffer = try context.createSharedBufferF32(self.values.len);
        errdefer input_buffer.deinit();
        try input_buffer.write(self.values);

        var output_buffer = try context.createSharedBufferF32(self.values.len);
        errdefer output_buffer.deinit();
        try context.copySharedBufferF32(input_buffer, output_buffer);
        input_buffer.deinit();

        return .{
            .allocator = self.allocator,
            .shape = self.shape,
            .values = output_buffer.asSlice(),
            .storage = .metal_shared,
            .host_values = null,
            .metal_buffer = output_buffer,
        };
    }

    fn initHostCopy(
        allocator: std.mem.Allocator,
        shape: layout.Shape,
        values: []const f32,
    ) !OwnedTensorF32 {
        const owned = try allocator.alloc(f32, values.len);
        errdefer allocator.free(owned);
        @memcpy(owned, values);

        return .{
            .allocator = allocator,
            .shape = shape,
            .values = owned,
            .storage = .host,
            .host_values = owned,
            .metal_buffer = null,
        };
    }

    fn initMetalSharedCopy(
        allocator: std.mem.Allocator,
        shape: layout.Shape,
        values: []const f32,
    ) !OwnedTensorF32 {
        var context = try metal.Context.init();
        defer context.deinit();

        var shared_buffer = try context.createSharedBufferF32(values.len);
        errdefer shared_buffer.deinit();
        try shared_buffer.write(values);

        return .{
            .allocator = allocator,
            .shape = shape,
            .values = shared_buffer.asSlice(),
            .storage = .metal_shared,
            .host_values = null,
            .metal_buffer = shared_buffer,
        };
    }
};

test "owned tensor preserves layout and storage across a Metal round-trip" {
    if (!metal.Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var host_tensor = try OwnedTensorF32.initCopy(
        allocator,
        .{ .rows = 2, .cols = 2 },
        &.{ 1.0, 2.0, 3.0, 4.0 },
        .host,
    );
    defer host_tensor.deinit();

    var metal_tensor = try host_tensor.roundTripThroughMetal();
    defer metal_tensor.deinit();

    try std.testing.expectEqual(@as(usize, 4), metal_tensor.elementCount());
    try std.testing.expectEqual(Storage.metal_shared, metal_tensor.storage);
    try std.testing.expectEqualSlices(f32, host_tensor.values, metal_tensor.values);
}

test "metal-shared tensor materializes back to host on demand" {
    if (!metal.Context.isAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var metal_tensor = try OwnedTensorF32.initCopy(
        allocator,
        .{ .rows = 1, .cols = 3 },
        &.{ 0.5, 1.5, 2.5 },
        .metal_shared,
    );
    defer metal_tensor.deinit();

    var host_tensor = try metal_tensor.materializeToHost();
    defer host_tensor.deinit();

    try std.testing.expectEqual(Storage.host, host_tensor.storage);
    try std.testing.expectEqualSlices(f32, metal_tensor.values, host_tensor.values);
}
