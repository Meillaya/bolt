const core = @import("agent_cli_core.zig");

pub fn main() !void {
    try core.runAgent(.{
        .lane = "mnist",
        .prompt_path = "programs/mnist_system.md",
        .researcher_step = "mnist-researcher",
        .offline_artifact = "../artifacts/labrat-mnist-agent.json",
        .blocked_artifact = "../artifacts/blockers/labrat-mnist-agent-live-blocked.json",
        .audit_artifact = "../artifacts/labrat-mnist-agent-audit.jsonl",
    });
}
