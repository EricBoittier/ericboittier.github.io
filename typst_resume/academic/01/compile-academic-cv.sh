#!/usr/bin/env bash
# Build academic CV + cover letter + supplement + transcripts from ../etc/, then pdfunite.
# Requires: Typst, pdfunite (brew install poppler).
set -euo pipefail

cd "$(dirname "$0")"

TYPST="${TYPST:-$HOME/.cargo/bin/typst}"
OUT_COMBINED="academic-cv-cover-combined.pdf"

TRANSCRIPTS=(
  "../etc/Boittier U.pdf"
  "../etc/Transcript_EricBoittier.pdf"
)

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

for f in "${TRANSCRIPTS[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "error: transcript PDF not found: $f (cwd $(pwd))" >&2
    exit 1
  fi
done

# Cover letter → CV → supplement → transcripts
pdfunite \
  cover-letter-academic.pdf \
  academic-cv.pdf \
  academic-supplement.pdf \
  "${TRANSCRIPTS[@]}" \
  "$OUT_COMBINED"
echo "Wrote $OUT_COMBINED (cover + CV + supplement + ${#TRANSCRIPTS[@]} transcript file(s))"
