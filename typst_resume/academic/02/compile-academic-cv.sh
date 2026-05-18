#!/usr/bin/env bash
# Build academic CV + academic cover letter, then combine into one PDF.
# Requires: Typst (typst compile), pdfunite from Poppler (brew install poppler).
set -euo pipefail

cd "$(dirname "$0")"

TYPST="${TYPST:-$HOME/.cargo/bin/typst}"
OUT_COMBINED="academic-cv-cover-combined.pdf"

if ! command -v "$TYPST" >/dev/null 2>&1; then
  echo "error: Typst not found at $TYPST — set TYPST to your typst binary." >&2
  exit 1
fi

python3 gen_publications_typ.py

"$TYPST" compile academic-cv.typ academic-cv.pdf
"$TYPST" compile cover-letter-academic.typ cover-letter-academic.pdf
"$TYPST" compile academic-supplement.typ academic-supplement.pdf

if ! command -v pdfunite >/dev/null 2>&1; then
  echo "error: pdfunite not found (install Poppler: brew install poppler)." >&2
  exit 1
fi

# Cover letter → CV → references & portfolio supplement
pdfunite cover-letter-academic.pdf academic-cv.pdf academic-supplement.pdf "$OUT_COMBINED"
echo "Wrote $OUT_COMBINED (cover letter + CV + supplement)"
