#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PRESENTATION_DIR="$REPO_ROOT/docs/presentation"
FIGURES_SRC="$REPO_ROOT/docs/figures"
ASSETS_LINK="$PRESENTATION_DIR/assets"
DIST_DIR="$PRESENTATION_DIR/dist"
DIST_ASSETS="$DIST_DIR/assets"
HTML_OUT="$DIST_DIR/slides.html"
ZIP_OUT="$DIST_DIR/omnihc-presentation-html.zip"

if [ ! -d "$FIGURES_SRC" ]; then
  echo "Missing documentation figures: $FIGURES_SRC" >&2
  exit 1
fi

if [ -L "$ASSETS_LINK" ]; then
  rm "$ASSETS_LINK"
elif [ -e "$ASSETS_LINK" ]; then
  echo "Refusing to replace non-symlink path: $ASSETS_LINK" >&2
  exit 1
fi
ln -s ../figures "$ASSETS_LINK"

rm -rf "$DIST_ASSETS"
mkdir -p "$DIST_ASSETS"

grep -Eoh 'assets/[^" )>]+' "$PRESENTATION_DIR/slides.md" | sort -u |
while IFS= read -r asset_ref; do
  rel_path="${asset_ref#assets/}"
  source_path="$FIGURES_SRC/$rel_path"
  target_path="$DIST_ASSETS/$rel_path"
  if [ ! -f "$source_path" ]; then
    echo "Missing referenced figure: $source_path" >&2
    exit 1
  fi
  mkdir -p "$(dirname "$target_path")"
  cp "$source_path" "$target_path"
done

if [ -n "${MARP_BIN:-}" ]; then
  MARP="$MARP_BIN"
elif [ -x "$PRESENTATION_DIR/node_modules/.bin/marp" ]; then
  MARP="$PRESENTATION_DIR/node_modules/.bin/marp"
elif command -v marp >/dev/null 2>&1; then
  MARP="$(command -v marp)"
else
  echo "Missing Marp CLI. Install it locally or set MARP_BIN=/path/to/marp." >&2
  exit 1
fi

"$MARP" "$PRESENTATION_DIR/slides.md" \
  --theme "$PRESENTATION_DIR/theme.css" \
  --html \
  --allow-local-files \
  -o "$HTML_OUT"

rm -f "$ZIP_OUT"
(
  cd "$DIST_DIR"
  zip -qr "$ZIP_OUT" slides.html assets
)

echo "Wrote $HTML_OUT"
echo "Wrote $ZIP_OUT"
