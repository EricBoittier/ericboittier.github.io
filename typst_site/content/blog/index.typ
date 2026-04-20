#import "../index.typ": template, tufted
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#show: template.with(title: "Posts")

= Posts

Read posts directly from this timeline, or open each dedicated page from the title.

#tufted.margin-note[
  #text(size: 0.85em, weight: "semibold")[Contents]
  #linebreak()
  #link("/blog/test-notebook/")[2023-08-17 · From Jupyter to Blog]
  #linebreak()
  #link("/blog/fluxschnell/")[2023-08-17 · Flux.1]
  #linebreak()
  #link("/blog/profiling-fortan/")[2022-08-17 · Fortran is fast]
  #linebreak()
  #link("/blog/notes/")[2022-08-01 · Notes on Bayesian Optimization]
  #linebreak()
  #link("/blog/papertest/")[2022-08-17 · Paper (Draft Archive)]
]

#let render-post(path) = {
  let md-content = read(path)
  let md-content = md-content.replace(regex("(?s)^---.*?---\\s*"), "")
  cmarker.render(md-content, math: mitex)
}

#let timeline-entry(date, title, url, post-path) = [
  #table(
    columns: (18%, 82%),
    inset: 0.25em,
    stroke: none,
    align: (left, left),
    [#text(weight: "semibold")[#date]],
    [
      #link(url)[#text(weight: "semibold")[#title]]
      #linebreak()
      #text(size: 0.85em, fill: rgb("#555"))[Direct link: #link(url)[#url]]
      #linebreak()
      #linebreak()
      #render-post(post-path)
    ],
  )
]

#timeline-entry("2023-08-17", "From Jupyter to Blog", "/blog/test-notebook/", "test-notebook/post.md")
#linebreak()
#linebreak()
#linebreak()
#linebreak()

#timeline-entry("2023-08-17", "Flux.1", "/blog/fluxschnell/", "fluxschnell/post.md")
#linebreak()
#linebreak()
#linebreak()
#linebreak()

#timeline-entry("2022-08-17", "Fortran is fast. Profile your code to make it faster!", "/blog/profiling-fortan/", "profiling-fortan/post.md")
#linebreak()
#linebreak()
#linebreak()
#linebreak()

#timeline-entry("2022-08-01", "Notes on Bayesian Optimization", "/blog/notes/", "notes/post.md")
#linebreak()
#linebreak()
#linebreak()
#linebreak()

#timeline-entry("2022-08-17", "Paper (Draft Archive)", "/blog/papertest/", "papertest/post.md")
