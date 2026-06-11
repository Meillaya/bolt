# Release Process

Bolt v1 is released as a local Apple-Silicon CLI/workbench for `engine/` and
`labrat/`. A release candidate is valid only when fresh evidence is captured for
the current candidate and public metadata remains free of credentials, private
model paths, and generated evidence.

## Version authority

The root `VERSION` file is the product release version authority. Package
metadata in `engine/` and `labrat/` must agree with root `VERSION` unless a
future split-version policy explicitly replaces this rule.

## V1 blockers

The v1 release cannot ship until all blocker classes below are satisfied with
fresh evidence:

- CI: portable default CI runs the documented engine and Labrat health commands
  without large assets, provider credentials, live provider mode, or
  platform-specific shader tooling.
- Artifact contracts: production-gated artifacts have explicit schemas,
  freshness metadata, validation, and stale-state rejection.
- Security: default jobs use least privilege, require no secrets, preserve
  Labrat sandbox/path/redaction boundaries. The release policy phrase is:
  live provider mode is out of v1 production scope. The release policy phrase is: live provider mode
  is out of v1 production scope.
- Docs: release metadata, version authority, support scope, command surfaces,
  optional diagnostics, and runbooks are public-ready and internally
  consistent.
- CI action review: portable CI currently uses first-party `actions/checkout@v4`
  and `actions/upload-artifact@v4` with explicit first-party action exceptions.
  Third-party actions must be full-SHA pinned before adoption; first-party
  version pins require release security review before a public release tag.

## Deferred after v1

The following work is staged or deferred after v1 entry criteria are met, not a
blocker for the initial v1 readiness gate:

- Packaging: installer/package distribution beyond the local CLI/workbench
  release surface.
- Provenance: SLSA provenance, SPDX SBOM publication, and release artifact
  attestations.
- Notarization: Developer ID signing, Hardened Runtime, notarization,
  stapling, and validation.
- Performance: dashboards and long-running real-model benchmark tracking.

## Release evidence

Release evidence must be captured under the active release evidence location
using exact commands, current working directory, timestamps, exit codes,
toolchain versions when relevant, and artifact digests when applicable. Do not
use generated evidence from previous runs, ignored `artifacts/` contents, real
model paths, provider transcripts, or private datasets as public proof.

Minimum default health evidence should cover:

```sh
cd engine && zig build test --summary all
cd engine && zig build --summary all
cd labrat && zig build test --summary all
cd labrat && zig build api-offline-test --summary all
```

Optional Metal, native, and real-asset gates must be labeled opt-in and must not
make portable default CI fail when the required runner or asset is unavailable.

## Rollback

If a release candidate fails a blocker gate, stop the candidate, preserve the
fresh failing evidence, and revert or patch only the narrow change that caused
the failure. Re-run the failed gate and any adjacent release checks before
creating a replacement candidate. Do not rewrite public release notes to hide a
failed candidate; document the corrected candidate with fresh evidence.

## Staged post-v1 release lanes

- Provenance, SLSA, SPDX SBOM, and artifact attestation entry criteria are
  documented in `docs/release-provenance.md`; they are not default CI blockers
  until tooling and verification are selected.
- Performance policy is documented in `docs/performance-policy.md`; long-running
  real-model benchmarks are opt-in/self-hosted/manual, not default blockers.
- macOS Developer ID, Hardened Runtime, notarization, stapling, and validation
  readiness is documented in `docs/macos-distribution.md`; this is not App Store
  distribution and is deferred after v1.
- Production operations and troubleshooting are documented in
  `docs/production-runbook.md`.
