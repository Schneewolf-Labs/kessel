# Release Process

## Overview

Kessel uses [Semantic Versioning](https://semver.org/). The version is defined in a single place (`kessel/_version.py`) and read by `pyproject.toml` and `kessel/__init__.py`.

- **Patch** (0.1.x): Bug fixes, no API changes
- **Minor** (0.x.0): New features, backward-compatible (new merge methods, new config options)
- **Major** (x.0.0): Breaking API changes (renamed classes, changed function signatures, removed features)

## How to Release

### 1. Prepare the release

```bash
# Make sure you're on main and up to date
git checkout main && git pull

# Run checks
ruff check .
pytest

# Update the version
# Edit kessel/_version.py — this is the single source of truth
```

### 2. Update CHANGELOG.md

Add a section for the new version at the top of `CHANGELOG.md`:

```markdown
## [0.2.0] - 2026-03-14

### Added
- New merge method: XYZ

### Fixed
- Bug in FP8 requantization
```

### 3. Commit and tag

```bash
git add kessel/_version.py CHANGELOG.md
git commit -m "Release v0.2.0"
git tag v0.2.0
git push origin main --tags
```

### 4. GitHub does the rest

Pushing the tag triggers the **Release** workflow (`.github/workflows/release.yml`), which:

1. Runs the full test suite across Python 3.10/3.11/3.12
2. Builds the sdist and wheel
3. Publishes to PyPI via trusted publishing
4. Creates a GitHub Release with notes from CHANGELOG.md

## PyPI Trusted Publishing Setup (one-time)

To enable automated PyPI publishing:

1. Go to [pypi.org/manage/account/publishing](https://pypi.org/manage/account/publishing/)
2. Add a new pending publisher:
   - **Project name**: `kessel-merge`
   - **Owner**: `Schneewolf-Labs`
   - **Repository**: `kessel`
   - **Workflow name**: `release.yml`
   - **Environment**: `pypi`

## Versioning Examples

| Change | Version bump | Example |
|--------|-------------|---------|
| Fix an FP8 requantization bug | Patch | 0.1.0 → 0.1.1 |
| Add a merge method (e.g., karcher) | Minor | 0.1.0 → 0.2.0 |
| Add new config option with default | Minor | 0.1.0 → 0.2.0 |
| Rename `MergeConfig` fields | Major | 0.1.0 → 1.0.0 |
| Remove a merge method | Major | 0.1.0 → 1.0.0 |

## Emergency Hotfix

For critical bugs in a released version:

```bash
git checkout v0.1.0
git checkout -b hotfix/v0.1.1
# fix the bug
# bump version to 0.1.1 in kessel/_version.py
git commit -am "Fix critical bug in XYZ"
git tag v0.1.1
git push origin hotfix/v0.1.1 --tags
# open PR to merge hotfix back into main
```
