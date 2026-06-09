const std = @import("std");

pub const image_rows: u32 = 28;
pub const image_cols: u32 = 28;
pub const image_size: u32 = image_rows * image_cols;
pub const num_classes: u32 = 10;
pub const train_count: u32 = 60_000;
pub const test_count: u32 = 10_000;

pub const Split = enum {
    train,
    test_set,
};

pub const Mnist = struct {
    train_images: []f32,
    train_labels: []f32,
    train_labels_raw: []u8,
    test_images: []f32,
    test_labels: []f32,
    test_labels_raw: []u8,

    pub fn load(io: std.Io, allocator: std.mem.Allocator, base_dir: []const u8) !Mnist {
        const train_imgs = try loadImages(io, allocator, base_dir, "train-images-idx3-ubyte", train_count);
        errdefer allocator.free(train_imgs);
        const train_raw = try loadLabelsRaw(io, allocator, base_dir, "train-labels-idx1-ubyte", train_count);
        errdefer allocator.free(train_raw);
        const train_oh = try oneHot(allocator, train_raw);
        errdefer allocator.free(train_oh);

        const test_imgs = try loadImages(io, allocator, base_dir, "t10k-images-idx3-ubyte", test_count);
        errdefer allocator.free(test_imgs);
        const test_raw = try loadLabelsRaw(io, allocator, base_dir, "t10k-labels-idx1-ubyte", test_count);
        errdefer allocator.free(test_raw);
        const test_oh = try oneHot(allocator, test_raw);
        errdefer allocator.free(test_oh);

        return .{
            .train_images = train_imgs,
            .train_labels = train_oh,
            .train_labels_raw = train_raw,
            .test_images = test_imgs,
            .test_labels = test_oh,
            .test_labels_raw = test_raw,
        };
    }

    pub fn deinit(self: *Mnist, allocator: std.mem.Allocator) void {
        allocator.free(self.test_labels_raw);
        allocator.free(self.test_labels);
        allocator.free(self.test_images);
        allocator.free(self.train_labels_raw);
        allocator.free(self.train_labels);
        allocator.free(self.train_images);
        self.* = undefined;
    }

    pub fn image(self: Mnist, split: Split, index: usize) []const f32 {
        return switch (split) {
            .train => self.train_images[index * image_size ..][0..image_size],
            .test_set => self.test_images[index * image_size ..][0..image_size],
        };
    }
};

pub fn readU32BE(bytes: []const u8) u32 {
    return std.mem.readInt(u32, bytes[0..4], .big);
}

fn loadImages(io: std.Io, allocator: std.mem.Allocator, base_dir: []const u8, filename: []const u8, expected_count: u32) ![]f32 {
    const raw = try readFile(io, allocator, base_dir, filename, 64 * 1024 * 1024);
    defer allocator.free(raw);
    return parseImages(allocator, raw, expected_count);
}

pub fn parseImages(allocator: std.mem.Allocator, raw: []const u8, expected_count: u32) ![]f32 {
    if (raw.len < 16) return error.InvalidHeader;
    if (readU32BE(raw[0..4]) != 0x0803) return error.BadMagicNumber;
    const count = readU32BE(raw[4..8]);
    if (count != expected_count) return error.UnexpectedCount;
    const rows = readU32BE(raw[8..12]);
    const cols = readU32BE(raw[12..16]);
    if (rows != image_rows or cols != image_cols) return error.UnexpectedDimensions;

    const pixel_count = @as(usize, count) * image_size;
    if (raw.len < 16 + pixel_count) return error.TruncatedFile;
    const out = try allocator.alloc(f32, pixel_count);
    errdefer allocator.free(out);
    for (raw[16..][0..pixel_count], 0..) |byte, i| {
        out[i] = @as(f32, @floatFromInt(byte)) / 255.0;
    }
    return out;
}

fn loadLabelsRaw(io: std.Io, allocator: std.mem.Allocator, base_dir: []const u8, filename: []const u8, expected_count: u32) ![]u8 {
    const raw = try readFile(io, allocator, base_dir, filename, 1024 * 1024);
    defer allocator.free(raw);
    return parseLabelsRaw(allocator, raw, expected_count);
}

pub fn parseLabelsRaw(allocator: std.mem.Allocator, raw: []const u8, expected_count: u32) ![]u8 {
    if (raw.len < 8) return error.InvalidHeader;
    if (readU32BE(raw[0..4]) != 0x0801) return error.BadMagicNumber;
    const count = readU32BE(raw[4..8]);
    if (count != expected_count) return error.UnexpectedCount;
    if (raw.len < 8 + @as(usize, count)) return error.TruncatedFile;
    const out = try allocator.alloc(u8, count);
    errdefer allocator.free(out);
    @memcpy(out, raw[8..][0..count]);
    for (out) |label| if (label >= num_classes) return error.LabelOutOfRange;
    return out;
}

pub fn oneHot(allocator: std.mem.Allocator, raw: []const u8) ![]f32 {
    const out = try allocator.alloc(f32, raw.len * num_classes);
    errdefer allocator.free(out);
    @memset(out, 0.0);
    for (raw, 0..) |label, i| out[i * num_classes + label] = 1.0;
    return out;
}

fn readFile(io: std.Io, allocator: std.mem.Allocator, base_dir: []const u8, filename: []const u8, max_size: usize) ![]u8 {
    const path = try std.fs.path.join(allocator, &.{ base_dir, filename });
    defer allocator.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(max_size));
}

test "IDX image and label parsers validate headers, dimensions, and one-hot labels" {
    var image_raw = [_]u8{
        0, 0, 8, 3, 0, 0, 0, 1, 0, 0, 0, 28, 0, 0, 0, 28,
    } ++ [_]u8{0} ** image_size;
    image_raw[16 + 7] = 255;
    const images = try parseImages(std.testing.allocator, &image_raw, 1);
    defer std.testing.allocator.free(images);
    try std.testing.expectEqual(@as(usize, image_size), images.len);
    try std.testing.expectEqual(@as(f32, 1.0), images[7]);

    const label_raw = [_]u8{ 0, 0, 8, 1, 0, 0, 0, 1, 7 };
    const labels = try parseLabelsRaw(std.testing.allocator, &label_raw, 1);
    defer std.testing.allocator.free(labels);
    try std.testing.expectEqual(@as(u8, 7), labels[0]);
    const oh = try oneHot(std.testing.allocator, labels);
    defer std.testing.allocator.free(oh);
    try std.testing.expectEqual(@as(f32, 1.0), oh[7]);
}
