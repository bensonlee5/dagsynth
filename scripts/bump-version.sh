#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

usage() {
  cat <<EOF
Usage: $(basename "$0") <major|minor|patch> [--dry-run] [--tag]

Bump the project version in pyproject.toml.

Arguments:
  major|minor|patch   Which semver component to bump.

Options:
  --dry-run           Print old/new version without making changes.
  --tag               Git-commit the change and create a vX.Y.Z tag.
  -h, --help          Show this help message.
EOF
}

# --- Parse arguments ---
BUMP_TYPE=""
DRY_RUN=false
TAG=false

for arg in "$@"; do
  case "$arg" in
    major|minor|patch) BUMP_TYPE="$arg" ;;
    --dry-run)         DRY_RUN=true ;;
    --tag)             TAG=true ;;
    -h|--help)         usage; exit 0 ;;
    *)                 echo "Error: unknown argument '$arg'" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "$BUMP_TYPE" ]]; then
  echo "Error: bump type required (major, minor, or patch)." >&2
  usage >&2
  exit 1
fi

# --- Read current version (same regex as CI) ---
CURRENT=$(grep -m1 '^version' pyproject.toml | sed 's/.*= *"\(.*\)"/\1/')
if [[ -z "$CURRENT" ]]; then
  echo "Error: could not parse version from pyproject.toml" >&2
  exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

# --- Compute new version ---
case "$BUMP_TYPE" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"

echo "$CURRENT -> $NEW_VERSION"

if $DRY_RUN; then
  exit 0
fi

# --- Update pyproject.toml ---
sed -i.bak "s/^version = \"${CURRENT}\"/version = \"${NEW_VERSION}\"/" pyproject.toml
rm -f pyproject.toml.bak

echo "Updated pyproject.toml"

# --- Optionally commit and tag ---
if $TAG; then
  # Warn about uncommitted changes in other files
  if [[ -n "$(git diff --name-only HEAD 2>/dev/null | grep -v '^pyproject.toml$' | grep -v '^CHANGELOG.md$' || true)" ]] || \
     [[ -n "$(git diff --cached --name-only HEAD 2>/dev/null | grep -v '^pyproject.toml$' | grep -v '^CHANGELOG.md$' || true)" ]]; then
    echo "Warning: other files have uncommitted changes." >&2
  fi

  # --- Stamp CHANGELOG.md ---
  CHANGELOG="${REPO_ROOT}/CHANGELOG.md"
  if [[ ! -f "$CHANGELOG" ]]; then
    echo "Error: CHANGELOG.md not found." >&2
    exit 1
  fi

  # Check that [Unreleased] has at least one entry
  UNRELEASED_BODY=$(sed -n '/^## \[Unreleased\]/,/^## \[/{/^## \[/d; p;}' "$CHANGELOG")
  if [[ -z "$(echo "$UNRELEASED_BODY" | grep -v '^[[:space:]]*$')" ]]; then
    echo "Error: [Unreleased] section in CHANGELOG.md is empty. Add entries before tagging." >&2
    exit 1
  fi

  TODAY=$(date +%Y-%m-%d)
  # Replace "## [Unreleased]" with stamped header, preceded by a fresh Unreleased section
  sed -i.bak "s/^## \[Unreleased\]/## [Unreleased]\n\n## [${NEW_VERSION}] - ${TODAY}/" "$CHANGELOG"
  rm -f "${CHANGELOG}.bak"
  echo "Stamped CHANGELOG.md for ${NEW_VERSION}"

  git add pyproject.toml CHANGELOG.md
  git commit -m "chore: bump version to ${NEW_VERSION}"
  git tag "v${NEW_VERSION}"
  echo "Created tag v${NEW_VERSION}"
  echo "Run: git push origin main --tags"
fi
