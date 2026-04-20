#import "../config.typ": template, tufted
#show: template

= Machine Learning x Science

#tufted.margin-note[
  #image("../assets/ericboittier-github.jpg", width: 50%, alt: "Eric Boittier GitHub profile photo")
]

#tufted.margin-note[
  #text(size: 0.9em, weight: "medium")[Eric Boittier]
  #linebreak()
  #text(size: 0.8em)[GitHub profile photo.]
]

I am Eric Boittier, a scientist and software builder working at the interface of physical chemistry and machine learning.


== Profiles
- #link("https://github.com/ericboittier")[GitHub]
- #link("https://scholar.google.co.uk/citations?user=pAQXUFcAAAAJ")[Google Scholar]
- #link("https://www.linkedin.com/in/ericboittier")[LinkedIn]



== Paper Highlights

#tufted.margin-note[
  #figure(
    image("../assets/images_medium_ct4c00759_0010.gif", width: 100%, alt: "KernelMDCM publication figure"),
    caption: [Figure P1. KernelMDCM publication visual.]
  )
]
- #link("/publications/kernelmdcm/")[Kernel-based Minimal Distributed Charges: A Conformationally Dependent ESP-Model for Molecular Simulations] \
  Placeholder summary: machine-learned charge assignment for molecular simulation fidelity.#footnote[Extra note: this paper explores conformationally adaptive electrostatics for simulation accuracy.]

#tufted.margin-note[
  #figure(
    image("../assets/images_medium_ct1c00249_0008.gif", width: 100%, alt: "MLCCSDT publication figure"),
    caption: [Figure P2. MLCCSD(T) publication visual.]
  )
]
- #link("/publications/mlccsdt/")[Transfer Learning to CCSD(T): Accurate Anharmonic Frequencies from Machine Learning Models] \
  Placeholder summary: transfer learning to reach coupled-cluster-level vibrational accuracy.#footnote[Extra note: this work demonstrates transfer learning from lower-cost data toward high-level quantum targets.]

#tufted.margin-note[
  #figure(
    image("../assets/images_medium_ct1c00363_0012.gif", width: 100%, alt: "MLDATABASE publication figure"),
    caption: [Figure P3. MLDATABASE publication visual.]
  )
]
- #link("/publications/mldatabase/")[Impact of the Characteristics of Quantum Chemical Databases on Machine Learning Prediction of Tautomerization Energies] \
  Placeholder summary: how database composition affects model performance and transferability.#footnote[Extra note: this paper studies the relationship between dataset curation choices and generalization.]

== Blog Highlights

#tufted.margin-note[
  #image("../assets/Fortran_logo.svg", width: 40%, alt: "Fortran logo for profiling post")
  #linebreak()
  #text(size: 0.75em)[Figure B1. Fortran logo used for the profiling post.]
]
- #link("/blog/profiling-fortan/")[Fortran is fast. Profile your code to make it faster!] \
  Placeholder summary: practical notes on benchmarking and optimization workflow.#footnote[Extra note: this highlight focuses on profiling workflows and performance bottleneck analysis.]

#tufted.margin-note[
  #image("../assets/Jupyter_logo.svg.png", width: 40%, alt: "Jupyter logo for notebook post")
  #linebreak()
  #text(size: 0.75em)[Figure B2. Jupyter logo for notebook-to-blog workflows.]
]
- #link("/blog/test-notebook/")[From Jupyter to Blog] \
  Placeholder summary: converting computational notebooks into maintainable posts.#footnote[Extra note: this highlight emphasizes reproducible publishing from notebook environments.]

#tufted.margin-note[
  #image("../assets/GP_bands.png", width: 40%, alt: "Gaussian process bands figure for Bayesian optimization post")
  #linebreak()
  #text(size: 0.75em)[Figure B3. Gaussian process bands visualization.]
]
- #link("/blog/notes/")[Notes on Bayesian Optimization] \
  Placeholder summary: key heuristics for efficient model search in scientific workflows.#footnote[Extra note: this highlight references acquisition strategies and uncertainty-aware exploration.]