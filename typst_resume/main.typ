#import "@preview/minimal-cv:0.2.0": *
#import "@preview/cades:0.3.1": qr-code

// Optional document metadata.
#set document(
  title: "Curriculum Vitae",
  author: "Eric Boittier <eric.boittier@icloud.com>",
  keywords: ("cv", "resume"),
  date: datetime(year: 2025, month: 3, day: 16),
)


// Right margin: used for page and to push Contact box into corner (dx = margin = flush right)
#let page-margin-right = 35pt
#set page(
  margin: (
    left: 47pt,
    right: page-margin-right,
    bottom: 1.25cm,
  ),
)


// Learn about theming at https://github.com/lelimacon/typst-minimal-cv
#show: cv.with(
  theme: (
    font-size: 12pt,
    spacing: 17pt,
  )
)

// Alternate theming used in several areas.
#let accent-theme = (
  accent-color: maroon,
  body-color: maroon,
)
#place(
dx: 11cm,
dy: -1cm,
figure(
  placement: top,  // must be `auto`, `top`, or `bottom`
  image("headshot.jpeg", width: 14%),
))
#block[
  // #par[]
  #v(-11pt)
  = Eric Boittier, Ph.D.
  #v(5pt)
  == Scientific Software -- Deep Learning
  #v(5pt)
  
  #place(top + right, dx: 245pt, dy: -60pt)[
    #block(
      width: 230pt,
      fill: none,
      inset: 5pt,
      
      

      section(
        theme: (
          font-size: 10pt,
          accent-color: rgb("#88888800"),
          gutter-body-color: rgb("888"),
          body-color: rgb("808"),
          section-style: "underlined",
          spacing: 13pt,
        ),
        [Contact],
        {
          entry(
            [],
            [#emoji.house Arlesheim, BL 4144],
            none,
          )
          entry(
            [],
            [#emoji.phone +4176 234 35 81],
            none,
          )
          entry(
            [],
            link("mailto:eric.boittier\u{40}icloud.com", [#emoji.mail.arrow eric.boittier\u{40}icloud.com]),
            none,
          )
          entry(
            [],
            [#emoji.bubble.r EN (native) DE
            (B2)],
            none,
          )
        }
      )
    )
  ]
]


#let cv-two-col(left, right) = grid(
  columns: (6fr, 16pt, 3.4fr),
  left,
  {},
  right,
)

#let left-col-theme = theme.with(
  gutter-width: 26pt,
  section-style: "underlined",
)

#let right-col-theme = theme.with(
  section-style: "underlined",
)

#cv-two-col(
  {
    show: left-col-theme
    
    section(
      theme: (spacing: 16pt),
      [Professional Experience],
      {
        entry(
          // theme: accent-theme,
          right: [*\@Uni. Basel* -- Basel, CH],
          chronology(start: "2026", end: ""),
          [#text(fill: maroon)[Postdoctoral Scientist]],
          [
            Machine learning (ML) on graph-structured data
          ]
        )
        entry(
          // theme: accent-theme,
          right: [],
          chronology(start: "2020", end: "2025"),
          [#text(fill: maroon)[Teaching Assistant & PhD Researcher]],
          [
            #list(
              [Designed, trained, validated deep learning models],
              [1st place team, HYDRA Spectroscopy Challenge],
              [Peer reviewer, Journal of Chemical Physics],
            )
          ]
        )
        entry(
          right: [*\@TRI* -- Brisbane, AU],
          chronology(start: "2019", end: "2020"),
          [#text(fill: maroon)[Research Assistant]],
          [
            #par[Small-molecule \& (bio)molecular modelling, Schrödinger software suite, OpenMM, High Performance Computing  ]
            #list(
              [Collaborated with multidisciplinary experts, building end-to-end solutions in Python and R],
              [Co-supervised and authored research on structure- and ligand-based drug design],
            )
          ]
        )
      }
    )
  },
  {
    show: right-col-theme

    section(
      theme: (gutter-width: 78pt),
      [Skills \& Competencies],
      {
        entry(
          [Software],
                    none,
          [Python, C++, Julia, Fortran],
        )
        entry(
          [Web],
                    none,
          [Typescript, React, Tailwind CSS],
        )
                entry(
          [DevOps],
          none,
          [Git, CI/CD, Docker],
        )
        entry(
          [Machine Learning],
          none,
          [JAX, PyTorch, scikit-learn],
        )

        entry(
          [MLOps],
          none,
          [Unsloth, HuggingFace, Litellm, OpenClaw],
        )
        entry(
          [Data Science],
          none,
          [SQL, Polars, Pandas],
        )
      }
    )
  },
)

#cv-two-col(
  {
    show: left-col-theme

    section(
      [Education],
      {
        entry(
          // theme: accent-theme,
          right: [*\@Uni. Basel* -- Basel, CH],
          chronology(start: "2020", end: "2025"),
          [#text(fill: maroon)[Doctor of Philosophy, Chemistry]],
          [#par[Thesis: `Molecular Deep Learning for Quantitative Simulations and Electrostatic Models`]],
        )
        entry(
          right: [*\@Uni. Queensland* -- Brisbane, AU],
          chronology(start: "2015", end: "2020"),
          [#text(fill: maroon)[Bachelor of Advanced Science, Chemistry]],
          [#par[Thesis: `Development of Computational Tools for the Rational Design of Glycosaminoglycan Mimetics`. \ Honours (1st class), Dean's Commendation]],
        )
      }
    )
  },
  {
    show: right-col-theme

    section(
      theme: (gutter-width: 30pt),
      [Invited Talks],
      {
        entry(
          [2025],
          [#par(justify: true)[*`Mixing Machine Learned with Empirical Energy Functions`* \ #text(fill: maroon)[apoCHARMM Meeting]  \ 
          Boston, US (Online)]],
                    none,

        )
        entry(
          [2024],
          [#par(justify: true)[*`Bridging Machine Learning and Electrostatic Models`* \ #text(fill: maroon)[Swiss Chemical Society] \ #text(fill: maroon)[Machine~Learning Group] \ Lausanne, CH]],
          none,
        )

      }
    )
  },
)
 #v(-20pt) 
#{
  show: theme.with(section-style: "underlined");
  
  section(
    theme: (gutter-width: 1pt, column-gutter: 0.5em),
    [Open Source Software Development],
    [
  #v(-10pt)    
  #grid(
        columns: (0.1fr, 1fr),
        
        [#qr-code("https://github.com/EricBoittier", width: 1.1cm)],
        [#par[Community-driven/open-source contributions available on GitHub (\u{40}EricBoittier)]],
        
        )
        
      #grid(
        columns: (1fr, 1fr),
        gutter: 1em,
        [#list(
          [#text(fill: maroon)[Cheminformatics software] (RDKit) bug fixes for C++ and python bindings, attended User Group Meeting (Zurich, 2024) ],
          // [Computer vision tool for item tracking and inventory management for an online marketplace with Python/SQL],
          // [Frontend development for a literature carousel UI for browsing arXiv preprints, using React and Tailwind CSS],
        )],
        [#list(
          // [Cheminformatics software RDKit: bug fixes for C++ and python layers, hackathons and feature requests],
          [#text(fill: maroon)[Computer vision tool] (OpenCV) for item tracking and #text(fill: maroon)[inventory management] (SQL) for an online retailer (CardMarket)],
          // [Frontend development for a literature carousel UI for browsing arXiv preprints, using React and Tailwind CSS],
        )],
                [#list(
          // [Cheminformatics software RDKit: bug fixes for C++ and python layers, hackathons and feature requests],
          [AI for a popular online strategy game using #text(fill: maroon)[Reinforcement learning] (Stable Baselines3) with self-play and masked Proximal Policy Optimization],
          // [Frontend development for a literature carousel UI for browsing arXiv preprints, using React and Tailwind CSS],
        )],
                [#list(
          // [Cheminformatics software RDKit: bug fixes for C++ and python layers, hackathons and feature requests],
          [#text(fill: maroon)[Open Source Computational Pipeline] (mmml) for Density Functional Theory and Molecular Dynamics with machine learned interaction potentials written in JAX (python)],
          // [Frontend development for a literature carousel UI for browsing arXiv preprints, using React and Tailwind CSS],
        )],
      )
    ]
)
}


// #block(
//   width: 100%,
//   fill: white,
//   inset: 8pt,
//   [Languages: English (native), German (B2)]
// )



