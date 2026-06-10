# bolt — Apple-Silicon nnmetal + Labrat parity port

`bolt` is the production-ready local parity port of the reference `nnzap`
project. It combines a Zig + Metal inference engine with a sandboxed Labrat
agent harness so model/research gates can run locally, deterministically, and
with auditable evidence.

The approved `local-reference-nnzap` scope is complete. Live LLM-provider
execution is intentionally not part of the final gate: it remains disabled by
default and requires explicit credentials plus opt-in.

## Status and evidence

Final parity closure passed with independent review:

```text
code-reviewer: APPROVE
architect:     CLEAR
```

Key evidence:

- `artifacts/labrat-final-quality-gate.json`
- `artifacts/labrat-m6-final-closure.json`
- `artifacts/labrat-ai-slop-cleaner-report.json`
- `artifacts/nnzap-milestone9-labrat-readiness.json`
- `.omx/ultragoal/ledger.jsonl` for durable audit history

Historical note: the ledger retains an earlier `G007 review_blocked` record for
traceability. That blocker was resolved by `G008`; the aggregate parity goal is
complete.

## Repository layout

```text
engine/      Zig + Metal runtime, kernels, model loaders, proof gates, benches
labrat/      Researcher gates, toolbox, API-shaped agent core, domain agents
python/      Reference scripts and golden-fixture tooling
research/    Local check/compare/bench/proof scripts and replayable recipes
docs/        Architecture notes, parity contracts, inventories, runbooks
artifacts/   Generated acceptance, benchmark, blocker, and closure evidence
reference/   Local source/reference project (`reference/nnzap`)
tests/       Supplemental test assets
```

## Quick start

### Verify the native engine

```bash
cd engine
zig build test --summary all
```

Accepted final gate: `96/96` engine tests passed.

### Verify Labrat

```bash
cd labrat
zig build test --summary all
zig build api-offline-test --summary all
```

Accepted final gates: `170/170` Labrat tests passed and `7/7` offline API / agent-core tests passed.

### Run offline agent gates

```bash
cd labrat
zig build mnist-agent
zig build bonsai-agent
zig build bonsai-q4-agent
```

These execute deterministic offline scenarios plus credential-safe live-block
checks. They produce artifacts such as:

```text
artifacts/labrat-m5-mnist-agent.json
artifacts/labrat-m5-bonsai-agent.json
artifacts/labrat-m5-bonsai-q4-agent.json
artifacts/blockers/labrat-m5-*-live-blocked.json
```

### Run researcher gates

```bash
cd labrat
zig build mnist-researcher
zig build bonsai-researcher
zig build bonsai-q4-researcher
```

These wrap stable `engine/` commands and emit timestamped JSON evidence for the
MNIST, Bonsai F16, and Bonsai Q4 parity surfaces.

## Engine scope

`engine/` is the nnmetal-style runtime and model-gate layer:

- Apple-Silicon Metal shared-buffer runtime
- committed Metal shader library
- matmul, bias, activation, reduction, softmax, QMV/Q4MV paths
- MNIST training/inference proof surface
- Bonsai/Qwen tokenizer, loader, golden, and benchmark gates
- benchmark and proof artifacts under `artifacts/`

Detailed references:

- `docs/architecture.md`
- `docs/development.md`
- `docs/nnzap-build-command-mapping.md`
- `docs/nnzap-interface-contracts.md`
- `docs/nnzap-real-asset-checklist.md`

## Labrat scope

`labrat/` is the autonomous experiment harness around the engine:

- researcher commands for MNIST, Bonsai F16, and Bonsai Q4
- sandboxed file/read/write/copy/edit/build/test/bench toolbox
- snapshot and rollback evidence for mutating tools
- bounded subprocess execution and failure propagation
- API-shaped request/response parsing with deterministic offline mocks
- credential-safe live-provider blocking by default
- domain CLIs: `mnist-agent`, `bonsai-agent`, `bonsai-q4-agent`

For the full sandbox contract and module inventory, see
`docs/labrat-parity-inventory.md`.

## Live API policy

Live provider execution is opt-in only. Normal verification uses offline mocks
and a blocked-live gate.

```bash
cd labrat
export LABRAT_LIVE=1
export ANTHROPIC_API_KEY=...
./zig-out/bin/bonsai_agent
```

Do not commit live credentials, raw provider transcripts, or local data/model
assets. Secret-like fields are redacted from generated blocked-live artifacts.

## Data and model assets

Real datasets and model files are local final-gate assets and are intentionally
ignored by Git. Keep them under the documented local asset paths and preserve the
accepted artifact evidence.

Useful docs:

- `docs/nnzap-real-asset-checklist.md`
- `docs/reference-gap-closure-v1-completion.md`
- `artifacts/assets/nnzap-parity/asset-manifest.json`

## Final verification summary

The accepted Labrat Phase 2 closure recorded:

```text
cd labrat && zig build test --summary all              # pass; 170/170
cd labrat && zig build api-offline-test --summary all  # pass; 7/7
cd labrat && zig build mnist-agent                     # pass
cd labrat && zig build bonsai-agent                    # pass
cd labrat && zig build bonsai-q4-agent                 # pass
cd labrat && zig build mnist-researcher                # pass; real MNIST
cd labrat && zig build bonsai-researcher               # pass; real Bonsai F16
cd labrat && zig build bonsai-q4-researcher            # pass; real Bonsai Q4
cd engine && zig build test --summary all              # pass; 96/96
git diff --check                                       # pass
credential scan                                        # pass
```

## Documentation map

- `docs/architecture.md` — engine/runtime architecture
- `docs/development.md` — local development workflow
- `docs/labrat-parity-inventory.md` — Labrat module and sandbox inventory
- `docs/nnzap-build-command-mapping.md` — stable build/test command contracts
- `docs/nnzap-interface-contracts.md` — parity interface contracts
- `docs/reference-gap-closure-v1-completion.md` — reference-gap closure summary
- `.omx/plans/prometheus-strict/` — execution plans and test specs

Reference point: `reference/nnzap/README.md`.
