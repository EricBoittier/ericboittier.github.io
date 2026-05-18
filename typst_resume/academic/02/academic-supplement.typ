// CIAN PostDoc application supplement: referee contacts + portfolio QR grid
#import "@preview/zebra:0.1.0": qrcode

#set document(
  title: "References and links --- CIAN application",
  author: "Eric Boittier",
)

#set page(
  margin: (
    top: 1.25cm,
    bottom: 1.25cm,
    left: 1.5cm,
    right: 1.5cm,
  ),
)

#set text(size: 11pt, lang: "en")
#set par(justify: true)

#set heading(numbering: none)
#show heading.where(level: 1): it => [
  #v(-4pt)
  #text(size: 12pt, weight: "semibold")[#smallcaps(it.body)]
  #v(10pt)
]

= Professional references (2)

Listed below are the two referees requested at this recruitment stage (#emph[reference letters were not solicited]); both supervised or co-supervised substantive parts of my dissertation research at the University of Basel Department of Chemistry.

#par(justify: false)[
  #text(weight: "bold")[Dr. Michael Devereux] \
  Scientific Staff/Research IT, Computational Physical Chemistry --- Group Meuwly \
  Department of Chemistry, University of Basel \
  Klingelbergstrasse 80, CH-4056 Basel, Switzerland \
  Email:
  #link("mailto:michael.devereux@unibas.ch")[michael.devereux\@unibas.ch] \
  Telephone: #link("tel:+41612073820")[+41 61 207 38 20]
]

#v(14pt)

#par(justify: false)[
  #text(weight: "bold")[Prof. Dr. Markus Meuwly] \
  Professor of Physical Chemistry; Group Leader (Group Meuwly) \
  Department of Chemistry, University of Basel \
  Klingelbergstrasse 80, CH-4056 Basel, Switzerland \
  Email:
  #link("mailto:m.meuwly@unibas.ch")[m.meuwly\@unibas.ch] \
  Telephone: #link("tel:+41612073821")[+41 61 207 38 21]
]

#pagebreak()

= Online portfolios and technical work

#set par(justify: false)

#let qr-link(url, title, caption) = align(center)[
  #qrcode(url, quiet-zone: true, background-fill: white, width: 3.55cm)
  #v(8pt)
  #strong[#title] \
  #link(url)[#caption]
]

#grid(
  columns: (1fr, 1fr),
  column-gutter: 22pt,
  row-gutter: 26pt,
  qr-link(
    "https://scholar.google.com/citations?user=pAQXUFcAAAAJ",
    "Google Scholar",
    [Scholar profile (publications \& citations)],
  ),
  qr-link(
    "https://github.com/EricBoittier",
    "GitHub",
    [github.com/EricBoittier],
  ),
  qr-link(
    "https://ericboittier.github.io/",
    "Personal website",
    [ericboittier.github.io],
  ),
  qr-link(
    "https://huggingface.co/EricBoi",
    "Hugging Face",
    [huggingface.co/EricBoi],
  ),
)

#v(18pt)

#par(justify: true)[These profiles summarise publications alongside code artefacts, demos, datasets, and recent LLM-era tooling explorations.] 
