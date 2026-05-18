// Motivation letter (CIAN / DBE PostDoc) — typography matches academic CV (gloat)
// Target: 1 page; compile: typst compile motivation-letter-cian.typ motivation-letter-cian.pdf

#set document(
  title: "Motivation letter --- CIAN Postdoctoral position",
  author: "Eric Boittier, Ph.D.",
  date: datetime.today(),
)

#set page(
  margin: (
    top: 1.1cm,
    bottom: 1.1cm,
    left: 1.45cm,
    right: 1.45cm,
  ),
)

#set text(size: 10.8pt, lang: "en")
#set par(justify: true, leading: 0.62em)

#let author-name = "Eric Boittier, Ph.D."
#let address-line = "Arlesheim, BL 4144, Switzerland"

#align(center)[
  #block(text(size: 13.5pt, weight: 700)[#smallcaps(author-name)])
]

#pad(top: 1pt)[
  #align(center)[
    #smallcaps[
      #link("mailto:eric.boittier\@icloud.com")[eric.boittier\@icloud.com]
      #h(0.35em) | #h(0.35em)
      #link("tel:+41762343581")[+41 76 234 35 81]
      #h(0.35em) | #h(0.35em)
      #link("https://github.com/EricBoittier")[github.com/EricBoittier]
    ]
  ]
]

#align(center)[#smallcaps(address-line)]

#v(6pt)

#align(right)[#text(weight: "medium", size: 10pt)[#datetime.today().display("[month repr:long] [day], [year]")]]

#v(8pt)

#block(above: 0pt, below: 0pt)[
  #text(weight: "bold")[Prof. Dr. Philippe C. Cattin] \
  Professor \& Head --- Center for medical Image Analysis and Navigation (CIAN) \
  Department of Biomedical Engineering, University of Basel \
  Hegenheimermattweg 167B/C, CH-4123 Allschwil, Switzerland \
  #link("mailto:philippe.cattin@unibas.ch")[philippe.cattin\@unibas.ch] ---
  #link("https://dbe.unibas.ch/en/cian")[dbe.unibas.ch/en/cian]
]
#v(10pt)
#v(6pt)
#v(6pt)

Dear Prof. Dr. Cattin,

#v(8pt)

I am applying for CIAN's Postdoctoral role at the interface of artificial intelligence, natural language processing, and medical informatics.

My trajectory began in medicinal chemistry with a straightforward aim: contribute to science that improves patients' quality of life. I was always drawn to programming, statistics, and how computation helps researchers and clinicians handle complex data. My first laboratory role sat inside a multidisciplinary team focused on cancer and age-related disease.

The rise of deep learning drew me to Basel for doctoral study - graph neural networks and allied methods for physics-based molecular simulation.
In addition, my research focused on calibrated uncertainty modelling, using strategies for statistical learning theory to find mechanistic understanding inside the black box of deep learning.

Although the domains are significantly different, I believe my background in machine learning and software engineering can be easily transferred.
I am usually a bottom-up thinker, and my experience in encorporating informed priors into machine learning models to prevent out-of-distribution behaviour (e.g. hallucinations) is a natural fit.


As I am already experienced with the local infrastructure (such as SciCore), I strongly believe we can hit the ground running and make a meaningful contribution to reliable, agentic clinical reporting.

#v(10pt)

Sincerely,

#v(8pt)

#text(weight: 700)[#smallcaps(author-name)]
