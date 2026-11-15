#!/bin/sh
# Builds docs/*.png from these sources. Run from docs/src after
#   python experiments/summarize.py results/events --figures docs/src
# Needs pdflatex (TikZ, pgfplots) and Python with PyMuPDF. Each PNG is 1660 px wide.
set -e
for name in system audio network timing turns labels; do
  pdflatex -interaction=nonstopmode -halt-on-error "f_$name.tex" > /dev/null
  python -c "import fitz; p = fitz.open('f_$name.pdf')[0]; p.get_pixmap(dpi=round(1660 / (p.rect.width / 72))).save('../$name.png')"
  rm -f "f_$name.aux" "f_$name.log" "f_$name.pdf"
done
