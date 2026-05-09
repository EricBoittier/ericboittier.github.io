#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 gen_publications_typ.py
~/.cargo/bin/typst compile academic-cv.typ academic-cv.pdf
