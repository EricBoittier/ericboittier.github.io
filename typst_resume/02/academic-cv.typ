#import "@preview/gloat:0.1.0": *

// Bold own name in author lists (matches several name spellings in the CSV)
#show regex("Eric D\.? Boittier|Eric Boittier"): name => text(weight: "bold", name)

#show: cv.with(
  author: "Eric Boittier, Ph.D.",
  address: "Arlesheim, BL 4144, Switzerland",
  contacts: (
    [#link("mailto:eric.boittier\@icloud.com")[eric.boittier\@icloud.com]],
    [#link("tel:+41762343581")[+41 76 234 35 81]],
    [#link("https://github.com/EricBoittier")[github.com/EricBoittier]],
  ),
  updated: datetime.today(),
)

#align(center)[
  Postdoctoral candidate combining deep learning \& scientific software with interests in NLP, agentic pipelines, \& rigorous validation for biomedical and research workflows.
]

= Education

#edu(
  institution: "University of Basel",
  location: "Basel, CH",
  degrees: (
    [Doctor of Philosophy, Chemistry],
  ),
  date: datetime(year: 2025, month: 6, day: 1),
  details: [
    Thesis: #emph[Molecular Deep Learning for Quantitative Simulations and Electrostatic Models].
  ],
)

#edu(
  institution: "The University of Queensland",
  location: "Brisbane, AU",
  degrees: (
    [Bachelor of Advanced Science (Honours I), Chemistry],
  ),
  date: datetime(year: 2020, month: 1, day: 1),
  details: [
    Thesis: #emph[Development of Computational Tools for the Rational Design of Glycosaminoglycan Mimetics].
  ],
)

= Research Experience

#exp(
  role: "Researcher",
  org: "University of Basel --- Department of Chemistry",
  location: "Basel, CH",
  start: datetime(year: 2020, month: 1, day: 1),
  end: "Present",
  details: [
    - Deep learning for graph-structured scientific data and computational chemistry.
    - Design, training, and validation of machine-learned potentials and distributed charge models.
    - Strong performance in community challenges (HyDRA vibrational spectroscopy; Physical Chemistry Chemical Physics, 2023).
  ],
)

#exp(
  role: "Research Assistant",
  org: "Translational Research Institute, Cancer and Aging Research Program",
  location: "Brisbane, AU",
  start: datetime(year: 2019, month: 1, day: 1),
  end: datetime(year: 2020, month: 12, day: 31),
  details: [
    - Interdisciplinary collaboration on small-molecule and biomolecular modelling (Schrödinger suite, OpenMM).
    - Structure- and ligand-based drug design; data pipelines in Python and R.
  ],
)

/* Temporarily omitted
= Awards

#award(
  date: datetime(year: 2023, month: 1, day: 1),
  name: "HyDRA vibrational spectroscopy challenge",
  from: "First place --- Physical Chemistry Chemical Physics community challenge issue",
)
*/

= Invited presentations

#pres(
  authors: ([Eric Boittier]),
  title: [Bridging machine learning force fields with anisotropic electrostatic models],
  conference: [Swiss Chemical Society, Machine Learning Group],
  date: datetime(year: 2024, month: 1, day: 1),
  location: "Lausanne, CH",
  kind: "Talk",
)

#pres(
  authors: ([Eric Boittier]),
  title: [Mixing machine learned and empirical energy functions],
  conference: [apoCHARMM Meeting, NIH],
  date: datetime(year: 2025, month: 1, day: 1),
  location: "Boston, US (online)",
  kind: "Talk",
)

= Service

#ser(
  role: "Peer reviewer",
  org: "Journal of Chemical Physics (AIP Publishing)",
  start: datetime(year: 2020, month: 1, day: 1),
  end: "Present",
  summary: "Manuscript review",
)

= Skills

#skills((
  (
    "Programming\t\t\t",
    ([Python], [Julia], [R], [Fortran], [#raw("C++")], [OpenMPI], [CUDA]),
  ),
  (
    "Frontend Development\t\t\t",
    ([Typescript], [JavaScript], [React], [Tailwind CSS]),
  ),
  (
    "Machine Learning\t\t\t",
    ([JAX], [PyTorch], [TensorFlow], [scikit-learn], [Weights \& Biases]),
  ),
  (
    "Engineering\t\t\t",
    ([Git], [CI/CD], [Docker], [Slurm]),
  ),
  (
    "Data Science\t\t\t",
    ([SnakeMake], [SQL], [Polars], [Pandas]),
  ),
  (
    "Large Language Models\t\t\t",
    ([Unsloth], [Hugging Face], [LiteLLM], [LangChain], [OpenClaw]),
  ),
))

#pagebreak()
= Publications

#include "publications_gloat.typ"
