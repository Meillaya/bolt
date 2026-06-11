# Release provenance, attestations, and SBOM staging

Bolt v1 treats provenance, attestations, and SBOM publication as **post-v1 entry
criteria**, not default portable CI blockers. A release candidate may enter this
lane only after these entry criteria are true:

- entry criteria: all v1 blockers in `RELEASE.md` are green.
- entry criteria: `.github/workflows/ci.yml` portable CI has passed.
- entry criteria: artifact schema validation passes with
  `python3 scripts/release/validate_artifact.py --all-fixtures` and the current
  candidate artifacts.
- entry criteria: `git status --short` is empty except intentionally ignored
  run-local evidence before release packaging begins.

## Staged policies

- **SLSA**: record the chosen SLSA provenance path after a reusable workflow or
  equivalent isolated build design is selected. Reference: <https://slsa.dev/>.
- **SPDX SBOM**: generate an SPDX SBOM only after tooling is selected and
  reviewed. Reference: <https://spdx.dev/>.
- **GitHub artifact attestation**: if GitHub artifact attestations are adopted,
  require least-privilege workflow permissions, documented OIDC/provenance
  verification, and a verification command in this document before marking the
  lane complete. Reference: <https://docs.github.com/en/actions/concepts/security/artifact-attestations>.

## Placeholder commands

These commands document the intended shape only; they are not release blockers
until tooling is selected and a follow-up security review lands.

```sh
# Future: generate SPDX SBOM after tool selection.
# spdx-sbom-generator --output artifacts/release/bolt.spdx.json

# Future: generate or verify SLSA/GitHub attestation after workflow selection.
# gh attestation verify artifacts/release/bolt.tar.gz --repo Meillaya/bolt
```

## Completion rule

Do not mark SLSA, SPDX SBOM, or attestation complete unless this document names
an implemented workflow/script, a verification command, and evidence captured
under `.omo/evidence/` for the current candidate.
