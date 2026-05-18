// Cover letter — typography matches academic-cv.typ (gloat)
// Build: typst compile cover-letter-academic.typ cover-letter-academic.pdf

#set document(
  title: "Cover Letter",
  author: "Eric Boittier, Ph.D.",
  date: datetime.today(),
)

// Match gloat CV (src/core.typ): page + body text
#set page(
  margin: (
    top: 1.25cm,
    bottom: 1.25cm,
    left: 1.5cm,
    right: 1.5cm,
  ),
)

#set text(
  size: 11pt,
  lang: "en",
)

#set par(justify: true)

#let author-name = "Eric Boittier, Ph.D."
#let address-line = "Arlesheim, BL 4144, Switzerland"
#let tagline = [
  Post Doctoral Scientist in Artificial Intelligence building Machine Learning Models and Scientific Software Solutions for Research in the Natural Sciences.
]

#align(center)[
  #block(text(size: 14pt, weight: 700)[#smallcaps(author-name)])
]

#pad(top: 2pt)[
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

#v(10pt)


// --- Letter meta ---
#align(right)[#text(weight: "medium")[#datetime.today().display("[month repr:long] [day], [year]")]]

#v(10pt)

// --- Recipient ---
#block(above: 0pt, below: 0pt)[
  #text(weight: "bold")[Prof. Dr. Ece Özkan Elsen]

  Department of Biomedical Engineering \
  Universität Basel \
  Hegenheimermattweg 167C \
  CH-4123 Allschwil, Switzerland
]

#v(20pt)

Dear Prof. Dr. Ece Özkan Elsen,

#v(12pt)

I am writing to express my interest in the Postdoctoral Researcher position with the Analytics \& Informatics for Child Health group at the Basel Research Center for Child Health. I believe I have a lot to offer you and your team, and that this role would be a natural extension of my existing research interests and competencies.

The story behind my early research career started in medicinal chemistry, motivated by the goal of improving the quality of life of patients. I have always had an interest in programming and statistics, and my first research position was done in a multidisciplinary team fighting cancer and geriatric disease. It was the deep learning revolution around 7 years ago that drew me to Basel to begin my doctoral studies applying Graph Neural Networks to physics-based (bio)molecular modelling.

In addition to deep learning, I have also developed kernel-based methods and applied Bayesian optimisation. Uncertainty quantification and explainable AI are areas of overlap between our research. I am interested in digital health problems where machine learning can support clinical research and improve pediatric healthcare. I am interested in learning more about data governance, security, pseudonymization, and collaboration with clinical IT partners.

Working at the intersection of AI/ML and data-intensive research, I have experience with computational workflows, version control, containerized environments, and reproducible research setups. Through the BioZentrum, I have attended courses and seminars on HPC; and I am an experienced user of the SciCore cluster. I would be happy to support the group's IT administration needs, including coordination with university IT and HPC services, management of local infrastructure, website updates, software licensing, hardware procurement, and inventory. I have experience hosting websites (static and server based), as well as benchmarking GPU hardware for scientific applications.

Overall, I believe my combination of research experience, computational skills, interest in applied AI, and willingness to support shared infrastructure would allow me to contribute effectively to the AICH group. I would be grateful for the opportunity to hear more about how I can assist your research.

#v(24pt)

Sincerely,

#v(20pt)

#text(weight: 700)[#smallcaps(author-name)]
