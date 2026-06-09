const std = @import("std");
const bolt = @import("bolt");

const default_manifest_path = "../artifacts/assets/nnzap-parity/asset-manifest.json";

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);
    const manifest_path = if (args.len > 1) args[1] else default_manifest_path;

    const input = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        manifest_path,
        allocator,
        .limited(16 * 1024 * 1024),
    );
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, input, .{ .allocate = .alloc_always });
    defer parsed.deinit();

    const repo_root = try repoRootForManifest(allocator, manifest_path);
    const report = try validateManifest(init.io, allocator, repo_root, parsed.value);
    try writeReport(init, manifest_path, repo_root, report);
    if (!report.pass) return error.AssetGateBlocked;
}

const AssetKind = enum { dataset, model };

const AssetStatus = struct {
    present: bool = false,
    complete: bool = false,
    missing: bool = false,
    contract_count: usize = 0,
    contract_match_count: usize = 0,
    file_count: usize = 0,
    existing_count: usize = 0,
    checksum_count: usize = 0,
    checksum_match_count: usize = 0,
    size_declared_count: usize = 0,
    size_match_count: usize = 0,
    config_count: usize = 0,
    tokenizer_count: usize = 0,
    safetensors_count: usize = 0,
    safetensors_parse_count: usize = 0,
    error_count: usize = 0,
};

const ValidationReport = struct {
    pass: bool,
    mnist: AssetStatus,
    bonsai: AssetStatus,
    qwen_q4_gs64: AssetStatus,
    qwen_q4: AssetStatus,
};

fn validateManifest(io: std.Io, allocator: std.mem.Allocator, repo_root: []const u8, root: std.json.Value) !ValidationReport {
    if (root != .object) return error.InvalidAssetManifest;
    const assets_value = root.object.get("assets") orelse return error.MissingAssets;
    if (assets_value != .array) return error.InvalidAssets;

    var report = ValidationReport{
        .pass = true,
        .mnist = .{},
        .bonsai = .{},
        .qwen_q4_gs64 = .{},
        .qwen_q4 = .{},
    };

    for (assets_value.array.items) |asset| {
        if (asset != .object) return error.InvalidAsset;
        const id_value = asset.object.get("id") orelse return error.MissingAssetId;
        if (id_value != .string) return error.InvalidAssetId;
        const id = id_value.string;
        if (std.mem.eql(u8, id, "mnist-raw")) {
            report.mnist = try validateAsset(io, allocator, repo_root, asset, .dataset);
        } else if (std.mem.eql(u8, id, "bonsai-1.7b")) {
            report.bonsai = try validateAsset(io, allocator, repo_root, asset, .model);
        } else if (std.mem.eql(u8, id, "qwen3-1.7b-q4-gs64")) {
            report.qwen_q4_gs64 = try validateAsset(io, allocator, repo_root, asset, .model);
        } else if (std.mem.eql(u8, id, "qwen3-1.7b-q4")) {
            report.qwen_q4 = try validateAsset(io, allocator, repo_root, asset, .model);
        }
    }

    report.pass = report.mnist.complete and report.bonsai.complete and report.qwen_q4_gs64.complete;
    return report;
}

fn validateAsset(io: std.Io, allocator: std.mem.Allocator, repo_root: []const u8, asset: std.json.Value, kind: AssetKind) !AssetStatus {
    var status = AssetStatus{ .present = true };
    try validateAssetContract(asset, kind, &status);

    const status_value = asset.object.get("status");
    if (status_value) |value| {
        if (value == .string and (std.mem.indexOf(u8, value.string, "missing") != null or std.mem.indexOf(u8, value.string, "blocked") != null)) {
            status.missing = true;
        }
    }

    if (asset.object.get("files")) |files| {
        if (files == .array) {
            status.file_count = files.array.items.len;
            for (files.array.items) |file| {
                if (file != .object) {
                    status.error_count += 1;
                    continue;
                }
                const path_value = file.object.get("path");
                if (path_value == null or path_value.? != .string or path_value.?.string.len == 0) {
                    status.error_count += 1;
                    continue;
                }
                const path = path_value.?.string;
                if (!pathIsAllowed(path)) {
                    status.error_count += 1;
                    continue;
                }
                countPathKind(path, &status);

                if (hasConcreteString(file.object.get("role"))) {
                    status.contract_count += 1;
                    status.contract_match_count += 1;
                } else {
                    status.error_count += 1;
                }

                const resolved_path = try resolveManifestPath(allocator, repo_root, path);
                defer allocator.free(resolved_path);

                const stat = std.Io.Dir.cwd().statFile(io, resolved_path, .{}) catch |err| {
                    if (err == error.FileNotFound) {
                        status.error_count += 1;
                        continue;
                    }
                    status.error_count += 1;
                    continue;
                };
                if (stat.kind != .file) {
                    status.error_count += 1;
                    continue;
                }
                status.existing_count += 1;

                if (file.object.get("size")) |size_value| {
                    if (size_value == .integer and size_value.integer >= 0) {
                        status.size_declared_count += 1;
                        if (stat.size == @as(u64, @intCast(size_value.integer))) status.size_match_count += 1;
                    } else {
                        status.error_count += 1;
                    }
                }

                if (file.object.get("sha256")) |sha| {
                    if (sha == .string and isConcreteSha256(sha.string)) {
                        status.checksum_count += 1;
                        if (try fileSha256Matches(io, resolved_path, sha.string)) status.checksum_match_count += 1 else status.error_count += 1;
                    } else {
                        status.error_count += 1;
                    }
                } else {
                    status.error_count += 1;
                }

                if (isSafetensorsPath(path)) {
                    const metadata = bolt.safetensors.inspectMetadataFromFile(io, allocator, resolved_path) catch {
                        status.error_count += 1;
                        continue;
                    };
                    if (metadata.tensor_count > 0) status.safetensors_parse_count += 1 else status.error_count += 1;
                }
            }
        } else {
            status.error_count += 1;
        }
    }

    status.complete = switch (kind) {
        .dataset => !status.missing and status.file_count > 0 and status.existing_count == status.file_count and status.checksum_count == status.file_count and status.checksum_match_count == status.file_count and status.error_count == 0,
        .model => !status.missing and status.file_count > 0 and status.existing_count == status.file_count and status.checksum_count == status.file_count and status.checksum_match_count == status.file_count and status.config_count > 0 and status.tokenizer_count > 0 and status.safetensors_count > 0 and status.safetensors_parse_count == status.safetensors_count and status.error_count == 0,
    };
    return status;
}

fn validateAssetContract(asset: std.json.Value, kind: AssetKind, status: *AssetStatus) !void {
    const required = [_][]const u8{ "id", "status", "localPath", "source", "provenance", "licenseOrAccess" };
    inline for (required) |field| {
        status.contract_count += 1;
        if (hasConcreteString(asset.object.get(field))) {
            status.contract_match_count += 1;
        } else {
            status.error_count += 1;
        }
    }

    status.contract_count += 1;
    const acceptance = asset.object.get("acceptanceUse");
    if (acceptance) |value| {
        if (value == .array and value.array.items.len > 0) {
            var all_concrete = true;
            for (value.array.items) |item| {
                if (item != .string or !isConcreteText(item.string)) all_concrete = false;
            }
            if (all_concrete) status.contract_match_count += 1 else status.error_count += 1;
        } else if (value == .string and isConcreteText(value.string)) {
            status.contract_match_count += 1;
        } else {
            status.error_count += 1;
        }
    } else {
        status.error_count += 1;
    }

    if (kind == .model) {
        status.contract_count += 1;
        if (asset.object.get("modelTokenizerPairing")) |pairing| {
            if (pairing == .object and
                hasConcreteString(pairing.object.get("revision")) and
                hasConcreteString(pairing.object.get("sourceRepositoryOrUrl")) and
                hasConcreteString(pairing.object.get("rule")))
            {
                status.contract_match_count += 1;
            } else {
                status.error_count += 1;
            }
        } else {
            status.error_count += 1;
        }
    }
}

fn countPathKind(path: []const u8, status: *AssetStatus) void {
    const base = std.fs.path.basename(path);
    if (std.mem.eql(u8, base, "config.json") or std.mem.endsWith(u8, base, ".model.json")) status.config_count += 1;
    if (std.mem.indexOf(u8, base, "tokenizer") != null) status.tokenizer_count += 1;
    if (isSafetensorsPath(path)) status.safetensors_count += 1;
}

fn isSafetensorsPath(path: []const u8) bool {
    return std.mem.eql(u8, std.fs.path.extension(path), ".safetensors");
}

fn isConcreteSha256(value: []const u8) bool {
    if (value.len != 64) return false;
    if (std.mem.eql(u8, value, "TODO")) return false;
    for (value) |byte| switch (byte) {
        '0'...'9', 'a'...'f', 'A'...'F' => {},
        else => return false,
    };
    return true;
}

fn hasConcreteString(value: ?std.json.Value) bool {
    if (value) |v| {
        return v == .string and isConcreteText(v.string);
    }
    return false;
}

fn isConcreteText(value: []const u8) bool {
    if (value.len == 0) return false;
    const forbidden = [_][]const u8{ "TODO", "PLACEHOLDER", "UNKNOWN", "TBD", "FIXME" };
    for (forbidden) |token| {
        if (std.mem.indexOf(u8, value, token) != null) return false;
    }
    return true;
}

fn pathIsAllowed(path: []const u8) bool {
    if (path.len == 0) return false;
    if (std.mem.indexOf(u8, path, "\x00") != null) return false;
    if (std.mem.indexOf(u8, path, "..") != null) return false;
    if (!std.fs.path.isAbsolute(path)) return true;
    return std.mem.startsWith(u8, path, "/Users/mei/models/");
}

fn fileSha256Matches(io: std.Io, path: []const u8, expected_hex: []const u8) !bool {
    const Sha256 = std.crypto.hash.sha2.Sha256;
    var hasher = Sha256.init(.{});
    var file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);

    var offset: u64 = 0;
    var buffer: [1024 * 1024]u8 = undefined;
    while (true) {
        const read_len = try file.readPositionalAll(io, &buffer, offset);
        if (read_len == 0) break;
        hasher.update(buffer[0..read_len]);
        offset += read_len;
    }
    var digest: [Sha256.digest_length]u8 = undefined;
    hasher.final(&digest);
    const actual_hex = std.fmt.bytesToHex(digest, .lower);
    return std.ascii.eqlIgnoreCase(expected_hex, &actual_hex);
}

fn repoRootForManifest(allocator: std.mem.Allocator, manifest_path: []const u8) ![]const u8 {
    const marker = "artifacts/";
    if (std.mem.indexOf(u8, manifest_path, marker)) |index| {
        if (index == 0) return allocator.dupe(u8, ".");
        var prefix = manifest_path[0..index];
        while (prefix.len > 0 and (prefix[prefix.len - 1] == '/' or prefix[prefix.len - 1] == std.fs.path.sep)) {
            prefix = prefix[0 .. prefix.len - 1];
        }
        if (prefix.len == 0) return allocator.dupe(u8, ".");
        return allocator.dupe(u8, prefix);
    }
    return allocator.dupe(u8, ".");
}

fn resolveManifestPath(allocator: std.mem.Allocator, repo_root: []const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path)) return allocator.dupe(u8, path);
    return std.fs.path.join(allocator, &.{ repo_root, path });
}

fn writeAsset(stdout: anytype, name: []const u8, status: AssetStatus, trailing_comma: bool) !void {
    try stdout.print(
        "    \"{s}\": {{\"complete\":{},\"missing\":{},\"contract_fields\":{d},\"contract_matches\":{d},\"files\":{d},\"existing\":{d},\"checksums\":{d},\"checksum_matches\":{d},\"size_declarations\":{d},\"size_matches\":{d},\"configs\":{d},\"tokenizers\":{d},\"safetensors\":{d},\"safetensors_parsed\":{d},\"errors\":{d}}}{s}\n",
        .{
            name,
            status.complete,
            status.missing,
            status.contract_count,
            status.contract_match_count,
            status.file_count,
            status.existing_count,
            status.checksum_count,
            status.checksum_match_count,
            status.size_declared_count,
            status.size_match_count,
            status.config_count,
            status.tokenizer_count,
            status.safetensors_count,
            status.safetensors_parse_count,
            status.error_count,
            if (trailing_comma) "," else "",
        },
    );
}

fn writeReport(init: std.process.Init, manifest_path: []const u8, repo_root: []const u8, report: ValidationReport) !void {
    var stdout_buffer: [8192]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;
    try stdout.print(
        "{{\n" ++
            "  \"schema_version\": 2,\n" ++
            "  \"gate\": \"nnzap-real-asset-manifest\",\n" ++
            "  \"manifest_path\": \"{s}\",\n" ++
            "  \"repo_root\": \"{s}\",\n" ++
            "  \"status\": \"{s}\",\n" ++
            "  \"assets\": {{\n",
        .{ manifest_path, repo_root, if (report.pass) "pass" else "blocked" },
    );
    try writeAsset(stdout, "mnist-raw", report.mnist, true);
    try writeAsset(stdout, "bonsai-1.7b", report.bonsai, true);
    try writeAsset(stdout, "qwen3-1.7b-q4-gs64", report.qwen_q4_gs64, true);
    try writeAsset(stdout, "qwen3-1.7b-q4", report.qwen_q4, false);
    try stdout.print(
        "  }},\n" ++
            "  \"required_to_pass\": \"MNIST, Bonsai full-precision, and the reference-compatible qwen3-1.7b-q4-gs64 asset must each have existing files, concrete matching SHA256 values, and model assets must include config, tokenizer, and parseable safetensors metadata. The legacy qwen3-1.7b-q4 asset is reported for diagnostics but is not sufficient for Milestone 7 acceptance.\"\n" ++
            "}}\n",
        .{},
    );
    try stdout.flush();
}
