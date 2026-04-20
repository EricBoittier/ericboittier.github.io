#import "../index.typ": template, tufted
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
  Scientific Software Architect -- Deep Learning Applications \
  Location: Arlesheim, BL 4144 \
  Phone: +41 76 234 35 81 \
  Email: #link("mailto:eric.boittier@icloud.com")[`eric.boittier@icloud.com`] \
  Languages: English (native), German (B2)
]

I build machine learning methods and scientific software for chemistry, molecular simulation, and graph-structured data.


== Experience
- *2020--2026*: Researcher, University of Basel (Basel, CH). Deep learning on graph chemistry data, including models for physical simulation and electrostatics; first place in HYDRA Spectroscopy Prediction Challenge; peer reviewer for Journal of Chemical Physics.
- *2019--2020*: Research Assistant, Translational Research Institute (Brisbane, AU). Small-molecule and biomolecular modeling with Schrödinger and OpenMM; collaborative end-to-end data science workflows in Python and R.

== Skills
- Programming: Python, Julia, R, Fortran, C++, OpenMPI, CUDA
- Machine Learning: JAX, TensorFlow, PyTorch, scikit-learn, NumPy
- DevOps: Git, CI/CD, Docker
- Data Science: SQL, Polars, Pandas

== Invited Technical Talks
- *2024*: *Bridging Machine Learning Force Fields with Anisotropic Electrostatic Models*, Swiss Chemical Society Machine Learning Group (Lausanne, CH)
- *2025*: *Mixing Machine Learned and Empirical Energy Functions*, apoCHARMM Meeting NIH (Boston, US; Online)

== Open Source
- Community-driven contributions at #link("https://github.com/EricBoittier")[`@EricBoittier`]
- RDKit bug fixes across C++ and Python bindings
- OpenCV + SQL tooling for inventory tracking



== Education
- Ph.D. in Chemistry, University of Basel (2020--2025). Thesis: *Molecular Deep Learning for Quantitative Simulations and Electrostatic Models*.
- Bachelor of Advanced Science in Chemistry, University of Queensland (2015--2020). Thesis: *Development of Computational Tools for the Rational Design of Glycosaminoglycan Mimetics*.

== Papers
#{
  let bib = load-bibliography(read("papers.bib"))
  for item in bib.values().rev() [
    #let data = item.fields
    - #data.author, #emph(data.title), #data.journal, #data.year. DOI: #link(data.url)[#data.doi]
  ]
}
