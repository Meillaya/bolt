const core = @import("agent_cli_core.zig");

pub fn main() !void {
    try core.runAgent(.{
        .lane = "bonsai_q4",
        .prompt_path = "programs/q4_system.md",
        .researcher_step = "bonsai-q4-researcher",
        .offline_artifact = "../artifacts/labrat-m5-bonsai-q4-agent.json",
        .blocked_artifact = "../artifacts/blockers/labrat-m5-bonsai-q4-agent-live-blocked.json",
        .audit_artifact = "../artifacts/labrat-m5-bonsai-q4-agent-audit.jsonl",
    });
}
