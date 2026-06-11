# Changelog

All notable public release changes for Bolt are tracked here.

Bolt uses the root `VERSION` file as the product release version authority.
Package metadata in `engine/` and `labrat/` must agree with root `VERSION`
unless a future split-version policy explicitly says otherwise.

## Unreleased

### Added

- Public release metadata and governance documents: Apache-2.0 license, security
  policy, contribution guide, release process, and changelog.

### Release policy

- Default release evidence must be fresh and captured from commands run for the
  release candidate, not inferred from older `worklog.md` entries or ignored
  local artifacts.
- Default gates must stay portable and require no secrets, provider credentials,
  real datasets, or large model assets.
