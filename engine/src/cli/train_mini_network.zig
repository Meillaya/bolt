const std = @import("std");
const bolt = @import("bolt");

pub fn main(init: std.process.Init) !void {
    const metrics = try bolt.network.trainSyntheticMiniClassifier(init.arena.allocator(), 500, 0.1);
    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "{{\n" ++
            "  \"schema_version\": {d},\n" ++
            "  \"backend\": \"{s}\",\n" ++
            "  \"dataset\": \"synthetic-mini-classifier\",\n" ++
            "  \"epochs\": {d},\n" ++
            "  \"samples\": {d},\n" ++
            "  \"final_loss\": {d:.6},\n" ++
            "  \"accuracy_milli\": {d}\n" ++
            "}}\n",
        .{
            metrics.schema_version,
            metrics.backend,
            metrics.epochs,
            metrics.samples,
            metrics.final_loss,
            metrics.accuracy_milli,
        },
    );
    try stdout.flush();
}
