const std = @import("std");

pub const Context = struct {
    backend_name: []const u8 = "metal",

    pub fn isMetalBackend(self: Context) bool {
        return std.mem.eql(u8, self.backend_name, "metal");
    }
};

test "context defaults to metal backend" {
    const context = Context{};
    try std.testing.expect(context.isMetalBackend());
}
