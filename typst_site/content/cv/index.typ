#import "../../config.typ": template, tufted
#show: template
#import "@preview/citegeist:0.2.2": load-bibliography

= Eric Boittier, Ph.D.

#link("/assets/EricBoittier_CV.pdf")[
  #box(
    fill: rgb("#0A66C2"),
    inset: (x: 0.7em, y: 0.35em),
    radius: 4pt,
    [#text(fill: white, weight: "semibold")[Download CV (PDF)]],
  )
]

#tufted.margin-note[
  #image("../../assets/portrait-placeholder.svg", width: 92%, alt: "Profile image placeholder")
  #linebreak()
  #linebreak()
  Scientific Software Architect -- Deep Learning Applications \
  Location: Basel region, Switzerland \
  Phone: +41 76 234 35 81 \
  Email: #link("mailto:eric.boittier@icloud.com")[`eric.boittier@icloud.com`] \
  Languages: English (native), German (B2) \
  #link("https://scholar.google.ch/citations?hl=en&user=pAQXUFcAAAAJ")[Google Scholar] \
  #link("https://github.com/EricBoittier")[GitHub] \
  #link("https://www.linkedin.com/in/ericboittier")[LinkedIn]
]

I build machine learning methods and scientific software for chemistry, molecular simulation, and graph-structured data.
My recent work centers on physically informed ML force fields and scalable molecular modeling workflows#footnote[
  Public profile and publication list:
  #link("https://scholar.google.ch/citations?hl=en&user=pAQXUFcAAAAJ")[Google Scholar].
].

== Profiles

#table(
  columns: (auto, 1fr),
  inset: 0.15em,
  stroke: none,
  [#image("../../assets/icon-github.svg", width: 0.95em, alt: "GitHub icon")],
  [#link("https://github.com/EricBoittier")[`@EricBoittier`] -- repositories, tooling, and collaborations#footnote[
    Public GitHub profile:
    #link("https://github.com/EricBoittier")[github.com/EricBoittier].
  ]],
  [#image("../../assets/icon-googlescholar.svg", width: 0.95em, alt: "Google Scholar icon")],
  [#link("https://scholar.google.ch/citations?hl=en&user=pAQXUFcAAAAJ")[Google Scholar] -- publications in computational chemistry and ML#footnote[
    Scholar profile indicates a verified University of Basel affiliation.
  ]],
  [#image("../../assets/icon-linkedin.svg", width: 0.95em, alt: "LinkedIn icon")],
  [#link("https://www.linkedin.com/in/ericboittier")[LinkedIn] -- professional network and updates],
)

== Research Snapshot

- *Physically informed machine learning for molecular simulation*:
  transfer learning, electrostatics, and force-field design for chemistry-aware models.
  #tufted.margin-note[
    #image("../../assets/blog-highlight-1.svg", width: 92%, alt: "Research highlight 1")
    #linebreak()
    Visual motif representing ML-assisted molecular modeling.
  ]

- *Open-source scientific software engineering*:
  production-oriented Python/C++ tooling and community contributions#footnote[
    Example merged contribution:
    #link("https://github.com/rdkit/rdkit/pull/7811")[rdkit/rdkit PR #7811].
  ].
  #tufted.margin-note[
    #image("../../assets/blog-highlight-2.svg", width: 92%, alt: "Research highlight 2")
    #linebreak()
    Open-source engineering and reproducible workflows.
  ]

- *Bridging model quality with practical simulation*:
  integrating ML potentials into broader computational pipelines and benchmarking strategies.
  #tufted.margin-note[
    #image("../../assets/blog-highlight-3.svg", width: 92%, alt: "Research highlight 3")
    #linebreak()
    End-to-end modeling from method design to simulation use.
  ]


== Experience
- *2020--2026*: Researcher, University of Basel (Basel, CH).
  Deep learning on graph chemistry data, including models for physical simulation and electrostatics;
  first place in the HyDRA spectroscopy challenge#footnote[
    Related challenge publication:
    #link("https://scholar.google.ch/citations?view_op=view_citation&hl=en&user=pAQXUFcAAAAJ&citation_for_view=pAQXUFcAAAAJ:roLk4NBRz8UC")[The first HyDRA challenge for computational vibrational spectroscopy].
  ];
  peer reviewer for *Journal of Chemical Physics*.
- *2019--2020*: Research Assistant, Translational Research Institute (Brisbane, AU).
  Small-molecule and biomolecular modeling with Schrödinger and OpenMM; collaborative end-to-end data science workflows in Python and R.

== Skills
- Programming: Python, Julia, R, Fortran, C++, OpenMPI, CUDA
- Machine Learning: JAX, TensorFlow, PyTorch, scikit-learn, NumPy
- DevOps: Git, CI/CD, Docker
- Data Science: SQL, Polars, Pandas

== Invited Technical Talks
- *2024*: *Bridging Machine Learning Force Fields with Anisotropic Electrostatic Models*, Swiss Chemical Society Machine Learning Group (Lausanne, CH)
- *2025*: *Mixing Machine Learned and Empirical Energy Functions*, apoCHARMM Meeting NIH (Boston, US; Online)

== Open Source
- Community-driven contributions at #link("https://github.com/EricBoittier")[`@EricBoittier`] across molecular modeling and ML tooling
- RDKit fixes across C++ and Python bindings#footnote[
  Merged example:
  #link("https://github.com/rdkit/rdkit/pull/7811")[issue #7572 fix in RDKit].
]
- OpenCV + SQL tooling for inventory tracking



== Education
- Ph.D. in Chemistry, University of Basel (2020--2025). Thesis: *Molecular Deep Learning for Quantitative Simulations and Electrostatic Models*.
- Bachelor of Advanced Science in Chemistry, University of Queensland (2015--2020). Thesis: *Development of Computational Tools for the Rational Design of Glycosaminoglycan Mimetics*.

== Papers
#{
  let bib = load-bibliography(read("papers.bib"))
  for item in bib.values().rev() [
    #let data = item.fields
    - #data.author, #emph(data.title), #data.journal, #data.year. DOI: #link(data.url)[#data.doi]#footnote[
      External DOI landing page:
      #link(data.url)[#data.url].
    ]
  ]
}
