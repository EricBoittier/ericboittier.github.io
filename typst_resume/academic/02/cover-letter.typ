// Cover Letter Template — matches CV theme (main.typ)
// Compile: typst compile cover-letter.typ

#set document(
  title: "Cover Letter",
  author: "Eric Boittier",
  date: datetime.today(),
)

#let page-margin-right = 35pt
#set page(
  margin: (left: 47pt, right: page-margin-right),
)

#set text(
  font: "Inria Sans",
  size: 12pt,
)

#set par(
  justify: true,
  spacing: 1.2em,
)

// Theme colors (match main.typ)
#let accent-color = rgb("005198")

// --- Header ---
#block[
  #text(size: 1.4em, weight: "bold", fill: accent-color)[Eric Boittier, Ph.D.]
  #v(4pt)
  #text(size: 0.95em)[Scientific Software Architect — Deep Learning Applications]
  #v(12pt)
  #text(size: 10pt, fill: rgb("666"))[
    Arlesheim, BL 4144 \u{00B7}
    +4176 234 35 81 \u{00B7}
    #link("mailto:eric.boittier\u{40}icloud.com", [eric.boittier\u{40}icloud.com])
  ]
]

#v(24pt)

// --- Date ---
#text(fill: rgb("666"))[March 16, 2025]

#v(12pt)

// --- Recipient ---
#block[
  #text(weight: "bold")[Hiring Manager Name]
  #text(weight: "bold")[Company Name]
  #text[123 Street Address]
  #text[City, Postal Code]
]

#v(24pt)

// --- Salutation ---
Dear Hiring Manager,

#v(12pt)

...

#v(12pt)

...

#v(12pt)

...

#v(12pt)

Thank you for considering my application. I look forward to hearing from you.

#v(24pt)

// --- Closing ---
Sincerely,

#v(24pt)

#text(weight: "bold")[Eric Boittier]
