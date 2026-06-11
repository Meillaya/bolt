# macOS distribution readiness

Bolt v1 is a local Apple-Silicon CLI/workbench. Direct macOS distribution is
**deferred after v1** and is **not App Store** distribution.

Future direct-distribution readiness work must cover:

- Developer ID certificate selection and signing identity management.
- Hardened Runtime configuration for installed binaries.
- notarization submission for signed release archives.
- stapling of notarization tickets where applicable.
- validation on a clean macOS machine before publishing.

References:

- Apple notarization guide: <https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution>
- Apple Hardened Runtime: <https://developer.apple.com/documentation/security/hardened-runtime>

## Current v1 rule

Do not require Apple developer credentials, signing secrets, notarization,
stapling, or App Store tooling in default CI. Packaging/notarization cannot be
claimed complete until a future plan adds signing/notarization automation and
verification evidence.
