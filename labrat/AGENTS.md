# LABRAT KNOWLEDGE BASE

## OVERVIEW

`labrat/` is the sandboxed research-agent package. It runs deterministic offline
agent scenarios, blocks live provider paths by default, and summarizes engine
artifacts through lane-specific researcher binaries.

## STRUCTURE

```text
labrat/
├── build.zig          # Agent, researcher, test, and cross-engine gates
├── build.zig.zon      # Package manifest; no external deps
├── programs/          # Agent system prompts and experiment protocols
└── src/               # Toolbox, API parser, agent/researcher cores, lanes
```

## WHERE TO LOOK

| Task | Location | Notes |
| --- | --- | --- |
| Agent gate | `src/agent_cli_core.zig` | Offline scenarios, live blocking, audit artifacts |
| API parsing | `src/api_client.zig` | Offline response kinds and secret redaction |
| Tool dispatcher | `src/agent_core.zig` | API tool names to toolbox names |
| Sandbox tools | `src/toolbox.zig` | Path guards, command allowlist, experiment history |
| Shared helpers | `src/tools.zig` | File, JSON, path, and benchmark helpers |
| Research summaries | `src/researcher_core.zig` | Lane report modes and history summaries |
| Lane prompts | `programs/*.md` | Experiment protocols and hard constraints |

## CONVENTIONS

- Live provider work is opt-in only. `LABRAT_LIVE=1` without accepted provider
  setup writes a blocked artifact instead of calling a provider.
- Offline agent gates must stay deterministic: parser scenarios, toolbox mapping,
  sandbox evidence, and audit JSON are part of readiness.
- `toolbox.zig` owns read/write scopes, symlink rejection, hidden path denylists,
  and exact command allowlists. Extend tests before widening those gates.
- Agent lane binaries are thin wrappers that pass prompt path, artifact path, and
  researcher step into `agent_cli_core.runAgent`.
- `labrat/build.zig` intentionally runs `../engine` build steps for researcher
  gates; these inherit engine asset requirements.
- Prompts require one experiment hypothesis, then edit/check/test/bench, then
  `experiment_finish` keep/abandon discipline.

## ANTI-PATTERNS

- Do not bypass `run-cmd` allowlisting with shell metacharacters or ad hoc shell
  execution.
- Do not allow absolute paths, `..`, `data/`, `reference/`, `.git/`, secrets, or
  broad hidden directories through sandbox reads/writes.
- Do not log API keys, Authorization headers, cookies, or bearer tokens; keep
  redaction tests current.
- Do not make live provider access part of default `zig build test` or agent
  readiness gates.
- Do not reattempt known-regressed experiment configurations in `programs/*.md`
  workflows; use recorded history.

## COMMANDS

```sh
zig build test --summary all
zig build api-offline-test --summary all
zig build mnist-agent --summary all
zig build bonsai-agent --summary all
zig build bonsai-q4-agent --summary all
```

Researcher steps (`mnist-researcher`, `bonsai-researcher`, `bonsai-q4-researcher`)
may invoke real engine gates and therefore may require local assets.
