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
  This study introduces a kernel-based framework for predicting conformationally dependent distributed charges, enabling more physically faithful electrostatic representations in molecular simulations.#footnote[The method is designed to improve transferability across molecular conformations while preserving simulation efficiency.]

#tufted.margin-note[
  #figure(
    image("../assets/images_medium_ct1c00249_0008.gif", width: 100%, alt: "MLCCSDT publication figure"),
    caption: [Figure P2. MLCCSD(T) publication visual.]
  )
]
- #link("/publications/mlccsdt/")[Transfer Learning to CCSD(T): Accurate Anharmonic Frequencies from Machine Learning Models] \
  This work applies transfer learning to bridge lower-cost quantum data and CCSD(T)-level targets, delivering highly accurate anharmonic vibrational frequencies at reduced computational cost.#footnote[It demonstrates a practical route to near high-level quantum accuracy without prohibitive scaling.]

#tufted.margin-note[
  #figure(
    image("../assets/images_medium_ct1c00363_0012.gif", width: 100%, alt: "MLDATABASE publication figure"),
    caption: [Figure P3. MLDATABASE publication visual.]
  )
]
- #link("/publications/mldatabase/")[Impact of the Characteristics of Quantum Chemical Databases on Machine Learning Prediction of Tautomerization Energies] \
  This paper evaluates how database composition, chemical diversity, and sampling strategy influence machine-learning predictions of tautomerization energies.#footnote[The analysis highlights how data curation decisions directly affect generalization and model reliability.]

== Blog Highlights

#tufted.margin-note[
  #image("../assets/Fortran_logo.svg", width: 40%, alt: "Fortran logo for profiling post")
  #linebreak()
  #text(size: 0.75em)[Figure B1. Fortran logo used for the profiling post.]
]
- #link("/blog/profiling-fortan/")[Fortran is fast. Profile your code to make it faster!] \
  A practical guide to performance engineering in scientific codebases, covering robust benchmarking, profiler-driven diagnosis, and targeted optimization strategies for Fortran workflows.#footnote[Emphasis is placed on reproducible timing methodology and actionable bottleneck identification.]

#tufted.margin-note[
  #image("../assets/Jupyter_logo.svg.png", width: 40%, alt: "Jupyter logo for notebook post")
  #linebreak()
  #text(size: 0.75em)[Figure B2. Jupyter logo for notebook-to-blog workflows.]
]
- #link("/blog/test-notebook/")[From Jupyter to Blog] \
  A workflow for converting exploratory Jupyter notebooks into maintainable, publication-ready technical articles while preserving clarity, reproducibility, and narrative structure.#footnote[The approach separates experimentation from presentation to improve long-term maintainability.]

#tufted.margin-note[
  #image("../assets/GP_bands.png", width: 40%, alt: "Gaussian process bands figure for Bayesian optimization post")
  #linebreak()
  #text(size: 0.75em)[Figure B3. Gaussian process bands visualization.]
]
- #link("/blog/notes/")[Notes on Bayesian Optimization] \
  Concise notes on Bayesian optimization for scientific machine learning, with a focus on acquisition design, uncertainty-aware exploration, and efficient hyperparameter search under limited budgets.#footnote[The post emphasizes practical heuristics for balancing exploration and exploitation in real experiments.]