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

= Education

#edu(
  institution: "University of Basel",
  location: "Basel, CH",
  degrees: (
    [Ph.D., Chemistry],
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
    [Bachelor of Advanced Science (Honours), Chemistry],
  ),
  date: datetime(year: 2020, month: 1, day: 1),
  details: [
    Thesis: #emph[Development of Computational Tools for the Rational Design of Glycosaminoglycan Mimetics].
  ],
)

= Research Experience

#exp(
  role: "Researcher",
  org: "University of Basel — Department of Chemistry",
  location: "Basel, CH",
  start: datetime(year: 2020, month: 1, day: 1),
  end: "Present",
  details: [
    - Deep learning for graph-structured scientific data and computational chemistry.
    - Design, training, and validation of machine-learned potentials and distributed charge models.
    - Top performance in the HyDRA computational vibrational spectroscopy challenge (Physical Chemistry Chemical Physics, 2023).
  ],
)

#exp(
  role: "Research Assistant",
  org: "Translational Research Institute, The University of Queensland",
  location: "Brisbane, AU",
  start: datetime(year: 2019, month: 1, day: 1),
  end: datetime(year: 2020, month: 12, day: 31),
  details: [
    - Small-molecule and biomolecular modelling (Schrödinger suite, OpenMM).
    - Structure- and ligand-based drug design; data pipelines in Python and R.
  ],
)

= Awards

#award(
  date: datetime(year: 2023, month: 1, day: 1),
  name: "HyDRA vibrational spectroscopy challenge",
  from: "First place — Physical Chemistry Chemical Physics community challenge issue",
)

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
  title: [Mixing machine-learned and empirical energy functions],
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
    "Programming",
    ([Python], [Julia], [R], [Fortran], [#raw("C++")], [OpenMPI], [CUDA]),
  ),
  (
    "Machine learning",
    ([JAX], [PyTorch], [TensorFlow], [scikit-learn], [NumPy]),
  ),
  (
    "Engineering",
    ([Git], [CI/CD], [Docker], [SQL], [Polars], [Pandas]),
  ),
))

= Publications

#include "publications_gloat.typ"
