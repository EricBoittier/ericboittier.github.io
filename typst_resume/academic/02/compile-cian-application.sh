#!/usr/bin/env bash
# CIAN PostDoc bundle: motivation letter + CV (+ publications via CSV) + supplement +
# transcripts from ../etc/ (Boittier U.pdf, Transcript_EricBoittier.pdf).
# Builds cian-application-combined.pdf.
set -euo pipefail

cd "$(dirname "$0")"

TYPST="${TYPST:-$HOME/.cargo/bin/typst}"
OUT="cian-application-combined.pdf"

# Official scans (edit order/names here if filenames change).
TRANSCRIPTS=(
  "../etc/Boittier U.pdf"
  "../etc/Transcript_EricBoittier.pdf"
)

if ! command -v "$TYPST" >/dev/null 2>&1; then
  echo "error: Typst not found at \$TYPST=$TYPST" >&2
  exit 1
fi

python3 gen_publications_typ.py

"$TYPST" compile motivation-letter-cian.typ motivation-letter-cian.pdf
"$TYPST" compile academic-cv.typ academic-cv.pdf
"$TYPST" compile academic-supplement.typ academic-supplement.pdf

if ! command -v pdfunite >/dev/null 2>&1; then
  echo "error: pdfunite not found (brew install poppler)" >&2
  exit 1
fi

for f in "${TRANSCRIPTS[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "error: transcript PDF not found: $f (cwd $(pwd))" >&2
    exit 1
  fi
done

pdfunite \
  motivation-letter-cian.pdf \
  academic-cv.pdf \
  academic-supplement.pdf \
  "${TRANSCRIPTS[@]}" \
  "$OUT"
echo "wrote $(pwd)/$OUT  (motivation + CV + supplement + ${#TRANSCRIPTS[@]} transcript file(s))"
