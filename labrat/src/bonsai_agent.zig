const core = @import("agent_cli_core.zig");

pub fn main() !void {
    try core.runAgent(.{
        .lane = "bonsai",
        .prompt_path = "programs/bonsai_system.md",
        .researcher_step = "bonsai-researcher",
        .offline_artifact = "../artifacts/labrat-m5-bonsai-agent.json",
        .blocked_artifact = "../artifacts/blockers/labrat-m5-bonsai-agent-live-blocked.json",
        .audit_artifact = "../artifacts/labrat-m5-bonsai-agent-audit.jsonl",
    });
}
