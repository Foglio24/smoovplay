#!/bin/bash
# Builds the macOS release zip: the source SmoovPlay needs on a Mac, the installer, and a double-click launcher for it.
#   bash mac/package_release.sh <version>      ->  dist/SmoovPlay-macOS-<version>.zip
# Upload that zip as an asset of a GitHub release.
set -euo pipefail
VERSION="${1:?usage: package_release.sh <version, e.g. 0.1.0>}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
NAME="SmoovPlay-macOS-$VERSION"
OUT="$REPO/dist"
STAGE="$(mktemp -d)/$NAME"
mkdir -p "$STAGE/smoovplay" "$OUT"

# Tracked and new (not ignored) files; left out: Windows installer and native code, CI, tests, local notes and results.
git ls-files -co --exclude-standard -z \
  | grep -zvE '^(\.github/|installer/|native/|tests/|Claude outputs/|dist/|CLAUDE\.md$|.*result\.txt$|.*\.DS_Store$|mac/ISSUE_DRAFT\.md$)' \
  | while IFS= read -r -d '' f; do printf '%s\0' "$f"; done \
  | rsync -a --from0 --files-from=- ./ "$STAGE/smoovplay/"

cat > "$STAGE/Install SmoovPlay.command" <<'CMD'
#!/bin/bash
cd "$(dirname "$0")" || exit 1
bash smoovplay/mac/install.sh
status=$?
echo
read -r -p "Press Return to close this window. " _
exit $status
CMD
chmod +x "$STAGE/Install SmoovPlay.command" "$STAGE/smoovplay/mac/"*.sh
cp mac/README.md "$STAGE/README.md"
echo "$VERSION" > "$STAGE/smoovplay/VERSION"

rm -f "$OUT/$NAME.zip"
(cd "$(dirname "$STAGE")" && ditto -c -k --keepParent "$NAME" "$OUT/$NAME.zip")
echo "built: $OUT/$NAME.zip ($(du -h "$OUT/$NAME.zip" | cut -f1))"
