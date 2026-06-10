# Labrat Phase 2 parity inventory and sandbox contract

Status: Milestone 1 inventory gate for `.omx/plans/prometheus-strict/prd-nnzap-labrat-phase2.md` and `test-spec-nnzap-labrat-phase2.md`.
Reference: `reference/nnzap/labrat` (authorized porting boundary).
Baseline rule: nnmetal parity is closed. Labrat consumes `docs/nnzap-build-command-mapping.md`, `docs/nnzap-interface-contracts.md`, `artifacts/nnzap-milestone9-labrat-readiness.json`, and `.omx/ultragoal/ledger.jsonl`; it must not reopen nnmetal unless a blocking contract violation is proven.

## M1 acceptance checklist

- [x] Every Labrat reference file is listed.
- [x] Every build step and executable is listed.
- [x] Every system prompt is listed.
- [x] Side-effecting tools and command surfaces are identified.
- [x] Stable nnmetal command/artifact dependencies are mapped.
- [x] Sandbox contract defines allowed roots, denied paths, max output sizes, rollback behavior, and durable artifacts.
- [x] Later milestones are constrained to offline/mocked API behavior by default, with live API disabled unless explicit opt-in and credentials are present.

## Reference file manifest

| Reference path | Lines | Role | Port/adapt disposition |
|---|---:|---|---|
| `reference/nnzap/labrat/build.zig` | 228 | Zig build graph for modules, executables, and test step | Port to project Labrat build integration without changing nnmetal build contracts. |
| `reference/nnzap/labrat/build.zig.zon` | 12 | Zig package manifest | Port/adapt only if local Labrat package boundary needs it. |
| `reference/nnzap/labrat/programs/mnist_system.md` | 92 | MNIST agent system prompt | Port as durable prompt asset; no secrets or live-provider assumptions. |
| `reference/nnzap/labrat/programs/bonsai_system.md` | 246 | Bonsai F16 agent system prompt | Port as prompt asset; preserve engineering/safety intent but adapt paths to this repo. |
| `reference/nnzap/labrat/programs/q4_system.md` | 286 | Bonsai Q4 agent system prompt | Port as prompt asset; preserve Q4 focus and safe edit constraints. |
| `reference/nnzap/labrat/src/tools.zig` | 706 | Shared JSON/path/file/benchmark utility layer | M2 port/adapt before toolbox; preserve limits and JSON escaping behavior. |
| `reference/nnzap/labrat/src/toolbox.zig` | 3224 | Generic sandboxed toolbox CLI | M2 port/adapt core sandbox, file operations, command execution, snapshots, rollback, history. |
| `reference/nnzap/labrat/src/tools_test.zig` | 581 | Tests for shared utilities | M2 adapt into executable test gate. |
| `reference/nnzap/labrat/src/toolbox_test.zig` | 788 | Tests for toolbox parsing/safety behavior | M2 adapt/extend for this repo's sandbox contract. |
| `reference/nnzap/labrat/src/mnist_researcher.zig` | 871 | MNIST researcher CLI profile and custom config tools | M3 port/adapt to consume stable MNIST nnmetal commands/artifacts. |
| `reference/nnzap/labrat/src/bonsai_researcher.zig` | 67 | Bonsai F16 researcher CLI profile | M3 port/adapt to consume Bonsai F16 golden/bench commands. |
| `reference/nnzap/labrat/src/bonsai_q4_researcher.zig` | 67 | Bonsai Q4 researcher CLI profile | M3 port/adapt to consume Q4 golden/bench commands. |
| `reference/nnzap/labrat/src/api_client.zig` | 786 | Anthropic request/response JSON infrastructure | M4 port/adapt with offline mocks first and credential-safe live gate. |
| `reference/nnzap/labrat/src/agent_core.zig` | 2133 | Agent loop, tool dispatcher, API retry/logging/history | M4 port/adapt; live API disabled by default. |
| `reference/nnzap/labrat/src/mnist_agent.zig` | 468 | MNIST agent profile and tool schema | M5 port/adapt CLI behind explicit live opt-in. |
| `reference/nnzap/labrat/src/bonsai_agent.zig` | 674 | Bonsai F16 agent profile and tool schema | M5 port/adapt CLI behind explicit live opt-in. |
| `reference/nnzap/labrat/src/bonsai_q4_agent.zig` | 641 | Bonsai Q4 agent profile and tool schema | M5 port/adapt CLI behind explicit live opt-in. |

Total reference surface: 17 files, 11,870 lines.

## Build graph inventory

`reference/nnzap/labrat/build.zig` defines four shared modules and six executables. The local implementation must expose equivalent build/test surfaces while integrating with this repo's `engine` layout.

### Shared modules

| Module | Source | Imports | Milestone |
|---|---|---|---|
| `tools_module` | `src/tools.zig` | none | M2 |
| `toolbox_module` | `src/toolbox.zig` | `tools.zig` | M2 |
| `api_client_module` | `src/api_client.zig` | `tools.zig` | M4 |
| `agent_core_module` | `src/agent_core.zig` | `api_client.zig`, `tools.zig` | M4 |

### Executables and steps

| Build step | Executable | Source | Required behavior |
|---|---|---|---|
| `mnist-researcher` | `mnist_researcher` | `src/mnist_researcher.zig` | Research CLI for MNIST build/test/train artifacts. |
| `bonsai-researcher` | `bonsai_researcher` | `src/bonsai_researcher.zig` | Research CLI for Bonsai F16 golden and benchmark artifacts. |
| `bonsai-q4-researcher` | `bonsai_q4_researcher` | `src/bonsai_q4_researcher.zig` | Research CLI for Bonsai Q4 golden and benchmark artifacts. |
| `mnist-agent` | `mnist_agent` | `src/mnist_agent.zig` | Autonomous MNIST experiment runner; live API gated. |
| `bonsai-agent` | `bonsai_agent` | `src/bonsai_agent.zig` | Autonomous Bonsai F16 optimizer; live API gated. |
| `bonsai-q4-agent` | `bonsai_q4_agent` | `src/bonsai_q4_agent.zig` | Autonomous Bonsai Q4 optimizer; live API gated. |
| `test` | test artifacts | `src/toolbox.zig`, `src/tools_test.zig`, `src/toolbox_test.zig` | Labrat unit/regression tests. |

## System prompt inventory

| Prompt | Consumer | Porting requirement |
|---|---|---|
| `programs/mnist_system.md` | `mnist_agent` | Preserve MNIST experiment guidance; update paths/commands to current repo; forbid unsafe edits and secret output. |
| `programs/bonsai_system.md` | `bonsai_agent` | Preserve Bonsai optimization workflow; align with stable nnmetal F16 gates; make rollback mandatory. |
| `programs/q4_system.md` | `bonsai_q4_agent` | Preserve Q4 optimization workflow; align with stable Q4 gates; make rollback mandatory. |

## Tool and side-effect inventory

### Generic toolbox commands

The generic toolbox dispatches these tools:

| Tool | Side effects | Required sandboxing |
|---|---|---|
| `help` | none | Safe read-only. |
| `check` | runs configured build check command | Allowlisted command only, bounded output, fixed working directory. |
| `test` | runs configured test command | Allowlisted command only, bounded output, fixed working directory. |
| `bench` | runs configured primary benchmark and optional reference benchmark; writes last-bench history | Allowlisted command only; must consume stable nnmetal artifacts and write Labrat-owned history/artifacts only. |
| `show` | reads a scoped file | Read scope + max output guard; later port must address reference known gap where large files can be returned without a size guard. |
| `show-function` | reads/extracts scoped source function | Read scope + bounded output. |
| `read-file` | reads scoped file | Read scope + `MAX_FILE_SIZE` equivalent. |
| `write-file` | writes scoped file | Write scope only; snapshot before write; block denied paths and symlink components. |
| `copy-file` | copies scoped file | Read-scope source, write-scope destination; snapshot destination before copy; block denied paths and symlink components. |
| `edit-file` | search/replace write | Write scope only; require exact old content; snapshot before edit; block ambiguous/no-op edits. |
| `list-dir` | reads directory entries | Read scope only; bounded listing. |
| `cwd` | prints current directory | Safe read-only. |
| `run-cmd` | executes user-provided command | Must be narrowed from reference behavior to allowlisted commands/subcommands only; no arbitrary shell expansion; timeout/nonzero/spawn failures report error. |
| `experiment-start` | creates/switches experiment branch | Side effecting git branch operation; must be sandboxed or mocked offline where destructive. |
| `diff` | runs git diff | Read-only git command with bounded output. |
| `experiment-finish` | records experiment, runs cleanup/rollback/merge policy | Must write history atomically and rollback unaccepted edits. |
| `history` | reads experiment history | Read-only; bounded count and output. |

### MNIST custom researcher tools

`mnist_researcher` adds these custom tools to the generic toolbox:

| Tool | Target | Side effects |
|---|---|---|
| `config-show` | `nnmetal/examples/mnist.zig` | read/parse only. |
| `config-set` | `nnmetal/examples/mnist.zig` | writes hyperparameters/architecture; requires snapshot and rollback. |
| `config-backup` | `nnmetal/examples/mnist.zig.bak` | writes backup artifact. |
| `config-restore` | `nnmetal/examples/mnist.zig` from backup | write/restore operation; must be reversible. |

### Agent/API tools

The agent profiles expose LLM-facing schemas that dispatch to toolbox commands. Ported agents must keep schema names stable but execute with offline mocks unless live mode is explicitly enabled.

| Agent | Tool schema names |
|---|---|
| `mnist_agent` | `config_show`, `config_set`, `config_backup`, `config_restore`, `experiment_start`, `diff`, `experiment_finish`, `check`, `test`, `train`, `history`, `show`, `show_function`, `read_file`, `write_file`, `copy_file`, `edit_file`, `list_directory`, `cwd`, `run_command` |
| `bonsai_agent` | `experiment_start`, `diff`, `experiment_finish`, `check`, `test`, `bench`, `history`, `show`, `show_function`, `read_file`, `write_file`, `copy_file`, `edit_file`, `list_directory`, `cwd`, `run_command` |
| `bonsai_q4_agent` | `experiment_start`, `diff`, `experiment_finish`, `check`, `test`, `bench`, `history`, `show`, `show_function`, `read_file`, `write_file`, `copy_file`, `edit_file`, `list_directory`, `cwd`, `run_command` |

## Stable nnmetal command dependencies

Labrat must depend on stable commands from `docs/nnzap-build-command-mapping.md`, not ad hoc nnmetal internals.

| Labrat lane | Stable command(s) | Expected consumed artifact(s) | Gate |
|---|---|---|---|
| MNIST researcher/agent check | `cd engine && zig build` | build success, no new artifact required | M2/M3/M5 |
| MNIST researcher/agent test | `cd engine && zig build test` | test pass | M2/M3/M5 |
| MNIST researcher/agent train | `cd engine && zig build run` | `artifacts/nnzap-milestone3-run.json` | M3/M6 |
| MNIST 1-bit comparison | `cd engine && zig build run-1bit` | `artifacts/nnzap-milestone3-run-1bit.json` | M3/M6 |
| MNIST inference | `cd engine && zig build run-infer` | `artifacts/nnzap-milestone3-run-infer.json` | M3/M6 |
| Bonsai F16 golden | `cd engine && zig build run-bonsai-golden` | `artifacts/nnzap-milestone6-bonsai-readiness.json` | M3/M6 |
| Bonsai F16 benchmark | `cd engine && zig build run-bonsai-bench` | `artifacts/nnzap-milestone8-bonsai-bench.json` | M3/M6 |
| Bonsai Q4 golden | `cd engine && zig build run-bonsai-q4-golden` | `artifacts/nnzap-milestone7-q4-golden.json` | M3/M6 |
| Bonsai Q4 benchmark | `cd engine && zig build run-bonsai-q4-bench` | `artifacts/nnzap-milestone7-q4-bench.json` | M3/M6 |

Reference Bonsai researchers also call `../reference/.venv/bin/python ../reference/mlx_bonsai.py` for reference benchmark comparison. The local Labrat port may consume that only as an optional/reference comparison path; final gates must be based on this repo's accepted nnmetal artifacts unless an explicit PRD/test-spec update says otherwise.

## Sandbox contract for implementation milestones

### Roots and path model

- Repo root: `/Users/mei/Projects/bolt` at execution time; implementation must derive paths dynamically from CWD or build runner rather than hardcoding this absolute path.
- Labrat-owned source root: `labrat/` if implemented as a sibling Zig package, or `engine/src/labrat*` if integrated into `engine`; the chosen layout must be documented when M2 begins.
- Labrat-owned artifacts:
  - `artifacts/labrat-*.json`
  - `artifacts/labrat-*.log`
  - `artifacts/labrat-history/**` or hidden history dirs explicitly listed below.
- Read-only stable inputs:
  - `docs/nnzap-build-command-mapping.md`
  - `docs/nnzap-interface-contracts.md`
  - `artifacts/nnzap-milestone*-*.json`
  - `.omx/ultragoal/ledger.jsonl`
  - `reference/nnzap/labrat/**` for parity comparison only.

### Allowed read roots

| Lane | Allowed read roots/files |
|---|---|
| Shared toolbox | Labrat source, `engine/**`, `docs/**`, `artifacts/nnzap-*.json`, Labrat history dirs. |
| MNIST | `engine/**`, `docs/**`, `artifacts/nnzap-milestone3-*.json`, `.mnist_history/**` or Labrat equivalent. |
| Bonsai F16 | `engine/**`, `docs/**`, `artifacts/nnzap-milestone6-bonsai-readiness.json`, `artifacts/nnzap-milestone8-bonsai-bench.json`, `.bonsai_history/**` or Labrat equivalent. |
| Bonsai Q4 | `engine/**`, `docs/**`, `artifacts/nnzap-milestone7-q4-*.json`, `.bonsai_q4_history/**` or Labrat equivalent. |
| Agent prompts/API | Labrat prompts, Labrat history dirs, mock response fixtures, live response logs with redaction. |

### Allowed write roots

| Lane | Allowed writes |
|---|---|
| Shared toolbox tests | temp dirs only; test fixtures under Labrat-owned testdata if needed. |
| MNIST experiments | Only explicit MNIST experiment targets after snapshot, primarily `engine/src`/example files chosen by M2/M3 implementation; Labrat history/artifacts. |
| Bonsai F16 experiments | Only explicit engine source/example files after snapshot; Labrat history/artifacts. |
| Bonsai Q4 experiments | Only explicit engine source/example files after snapshot; Labrat history/artifacts. |
| Agent/API offline mocks | mock fixtures and Labrat history/artifacts only. |

### Denied paths

All Labrat tools must reject these paths even if a caller supplies relative traversal:

- `.git/**`, `.ssh/**`, `.gnupg/**`, `.aws/**`, `.config/**`, `.local/share/**`, `~/Library/**`, `/etc/**`, `/var/**`, `/tmp/**` outside a Labrat-created temp sandbox.
- `.omx/**` except read-only `.omx/ultragoal/ledger.jsonl` and writing explicit checkpoint/evidence through the OMX CLI, not the Labrat toolbox.
- `reference/nnzap/**` for writes. Reference is read-only.
- `data/**` for writes unless a later milestone explicitly creates a Labrat-owned fixture; real model/MNIST assets are inputs.
- Secret-bearing environment variables and files; logs must never include raw `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, tokens, cookies, or request authorization headers.

### Output, input, and runtime limits

| Limit | Required local contract |
|---|---|
| Max file read | Preserve reference `tools.MAX_FILE_SIZE` intent; cap regular file reads at <= 2 MiB unless a tool returns a structured outline instead. |
| Max process output | Preserve reference `tools.MAX_OUTPUT_BYTES`/toolbox bounded output intent; cap command output at <= 1 MiB for subprocess capture and <= 50 KiB for LLM tool-return text. |
| Max API response | M4 must cap raw mocked/live API response storage at <= 8 MiB and parse failures safely. |
| Max tool calls | M4/M5 must cap tool calls per response at <= 16. |
| Max history size | Cap per-history file read at <= 2 MiB; append history atomically. |
| Command timeout | Every subprocess must have a bounded timeout; reference curl timeout is 600 seconds, but local tests should use much shorter mock/test timeouts. |
| Command allowlist | `run-cmd` must not execute arbitrary shell. Accept only named safe commands/subcommands or a strict argv allowlist tied to stable nnmetal commands. |

### Snapshot and rollback behavior

- Before any write/edit operation, create a snapshot record containing path, original bytes hash, size, and timestamp.
- If an edit fails validation, rollback immediately from the snapshot.
- `experiment-start` creates an isolated experiment context. If implemented with git branches, it must refuse dirty worktrees unless snapshot/rollback can prove reversibility. If implemented without branches, it must use a Labrat-owned temp/snapshot directory.
- `experiment-finish` decisions:
  - `accept`: persist summary/history, leave accepted edits only after tests/benchmarks pass.
  - `reject`/`abandon`: rollback all experiment edits and persist an audit record.
- Agent crash/turn-limit behavior must attempt rollback and emit a durable blocked/abandoned artifact.

## API/live-provider safety contract

Reference `agent_core.zig` calls Anthropic Messages API via `curl`, uses `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL`, default model `claude-opus-4-6`, retries transient failures, saves `_request.json`/`_response.json`, and injects tools into the request. Local Labrat Phase 2 must change the default posture:

- Offline/mock mode is default for M4 and M5.
- Live network/API calls are disabled unless all are true:
  1. explicit live flag is passed,
  2. provider is selected,
  3. required API key env var is present,
  4. output redaction is enabled,
  5. test-spec live gate is intentionally run.
- Missing credentials must create a credential-safe blocked artifact, not a failed crash or secret-bearing log.
- Request/response logs must redact authorization headers and key-like fields.

## Durable Labrat artifacts

| Artifact | Producer | Required fields |
|---|---|---|
| `artifacts/labrat-m1-inventory-gate.json` | M1 | status, reference file count, build steps, prompts, side-effect list, sandbox summary, verification commands. |
| `artifacts/labrat-m2-toolbox-tests.json` | M2 | status, unit tests, sandbox tests, rollback tests, command allowlist tests. |
| `artifacts/labrat-m3-mnist-researcher.json` | M3 | stable commands run, consumed nnmetal artifacts, timestamp, pass/block status. |
| `artifacts/labrat-m3-bonsai-researcher.json` | M3 | F16 stable commands run, consumed artifacts, timestamp, pass/block status. |
| `artifacts/labrat-m3-bonsai-q4-researcher.json` | M3 | Q4 stable commands run, consumed artifacts, timestamp, pass/block status. |
| `artifacts/labrat-m4-offline-api-agent-core.json` | M4 | mock scenarios, parsing/dispatch results, refusal/malformed/truncation coverage. |
| `artifacts/labrat-m5-agent-cli-readiness.json` | M5 | CLIs, offline runs, live blocked artifacts, sandboxed edit/test/rollback evidence. |
| `artifacts/labrat-m6-final-closure.json` | M6 | full matrix, security/rollback review, final pass/block status. |
| `artifacts/labrat-final-quality-gate.json` | M6 | ai-slop status, verification, code-reviewer APPROVE, architect CLEAR. |

## M1 decisions carried into M2+

1. `reference/nnzap/labrat` is read-only. Port/adapt internals are authorized, but the implementation must live in this repo's non-reference source tree.
2. Arbitrary `run-cmd` behavior from the reference must be tightened to an argv allowlist tied to stable build/test/benchmark commands.
3. Large-file reads must return structured outlines/truncated content instead of unbounded full file content; the reference's documented `toolShow` size-gap is not allowed to survive local final gates.
4. Live API behavior is a late, opt-in readiness check; all acceptance through M5 must pass offline with deterministic mocks.
5. Any missing real nnmetal asset/artifact required by a final gate creates an explicit blocker artifact; it must not be replaced by synthetic-only acceptance.
