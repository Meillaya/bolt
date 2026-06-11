# Security Policy

## Supported scope

Bolt v1 is a local Apple-Silicon CLI/workbench. Security support for v1 covers
the default local build, test, artifact-validation, and Labrat offline harness
surfaces documented in the root README and release process.

Default jobs must require no secrets. They must not depend on provider
credentials, private datasets, real model weights, generated evidence, or local
machine paths. Any gate that needs real assets, Metal-specific tooling, or
self-hosted runners must be opt-in and clearly labeled outside the portable
default path.

live provider mode is out of v1 production scope. Provider-backed runs
must not be required by CI, release checks, or public DoneClaim evidence.
`LABRAT_LIVE=1` may be used only to prove the fail-closed safety gate, and
provider credentials must remain redacted from logs and artifacts.

## Reporting a vulnerability

Report suspected vulnerabilities privately to the maintainers before public
disclosure. Include the affected command, platform, expected behavior, observed
behavior, and a minimal reproduction that does not include credentials, private
model paths, datasets, transcripts, or generated evidence.

## Handling sensitive material

Do not file or commit:

- provider credentials or tokens;
- private model, dataset, or cache paths;
- generated `.omo/evidence/`, `artifacts/`, `data/`, `.zig-cache/`, or
  `zig-out/` contents;
- provider transcripts or logs that may contain user data.

If sensitive material is exposed, rotate the affected secret outside this
repository and open a security report describing the repository-side cleanup
needed.
