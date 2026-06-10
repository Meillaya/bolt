const core = @import("agent_cli_core.zig");

pub fn main() !void {
    try core.runAgent(.{
        .lane = "bonsai",
        .prompt_path = "programs/bonsai_system.md",
        .researcher_step = "bonsai-researcher",
        .offline_artifact = "../artifacts/labrat-bonsai-agent.json",
        .blocked_artifact = "../artifacts/blockers/labrat-bonsai-agent-live-blocked.json",
        .audit_artifact = "../artifacts/labrat-bonsai-agent-audit.jsonl",
    });
}
