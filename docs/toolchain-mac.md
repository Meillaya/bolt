# macOS Zig + Metal toolchain guide

Last verified: April 22, 2026

## What this project expects
- Xcode installed at `/Applications/Xcode.app`
- Apple Metal Toolchain downloaded through Xcode
- Zig installed
- Python 3 available

## What was set up on this Mac
- Zig installed with Homebrew: `zig 0.16.0`
- Xcode detected: `Xcode 26.0.1`
- Metal Toolchain downloaded with:
  - `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -downloadComponent MetalToolchain`
- Shell configured in `~/.zshenv` (and mirrored in `~/.zshrc` / `~/.zprofile` where available):
  - `export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"`

## Why `DEVELOPER_DIR` is set
This Mac had `Xcode.app` installed, but `xcode-select -p` was still pointing at:
- `/Library/Developer/CommandLineTools`

That was enough for general command-line tools, but not enough for Metal. Setting `DEVELOPER_DIR` makes `xcrun` use the full Xcode toolchain without requiring a system-wide `sudo xcode-select --switch ...`.

## Fresh setup on another Mac

### 1. Install Xcode
Install Xcode from Apple, then open it once.

### 2. Download the Metal Toolchain
Run:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -downloadComponent MetalToolchain
```

### 3. Install Zig
If Homebrew is available:

```bash
brew install zig
```

### 4. Export the Xcode developer directory
Add this to `~/.zshenv` (recommended for all zsh invocations):

```bash
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
```

Then reload your shell:

```bash
source ~/.zshenv
```

## Verify everything
Run:

```bash
research/scripts/doctor.sh
```

Expected checks:
- Zig version prints
- Xcode version prints
- `xcrun --find metal` succeeds
- a tiny `.metal` file compiles to `.air`
- Python 3 is available

## Useful manual checks

```bash
zig version
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild -version
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcrun --find metal
```

## If Metal still fails
1. Confirm Xcode exists at `/Applications/Xcode.app`
2. Re-run:

```bash
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer \
  xcodebuild -downloadComponent MetalToolchain
```

3. Make sure your shell has:

```bash
export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
```

4. Retry:

```bash
research/scripts/doctor.sh
```

## Optional system-wide switch
If you want the whole system to use Xcode instead of Command Line Tools, you can run this manually:

```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
```

That was not done automatically here because it requires your macOS admin password.
