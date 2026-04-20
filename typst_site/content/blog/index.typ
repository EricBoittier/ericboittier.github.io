#import "../../config.typ": template, tufted
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#show: template.with(title: "Posts")

= Posts

Read posts directly from this timeline, or open each dedicated page from the title.

#let render-post(path) = {
  let md-content = read(path)
  let md-content = md-content.replace(regex("(?s)^---.*?---\\s*"), "")
  cmarker.render(md-content, math: mitex)
}

#let posts = (
  (
    date: "2023-08-17",
    title: "From Jupyter to Blog",
    url: "/blog/test-notebook/",
    path: "test-notebook/post.md",
  ),
  (
    date: "2023-08-17",
    title: "Flux.1",
    url: "/blog/fluxschnell/",
    path: "fluxschnell/post.md",
  ),
  (
    date: "2022-08-17",
    title: "Fortran is fast. Profile your code to make it faster!",
    url: "/blog/profiling-fortan/",
    path: "profiling-fortan/post.md",
  ),
  (
    date: "2022-08-01",
    title: "Notes on Bayesian Optimization",
    url: "/blog/notes/",
    path: "notes/post.md",
  ),
)

#tufted.margin-note[
  #text(size: 20em, weight: "semibold")[Contents]
  #for post in posts [
    #linebreak()
    #link(post.url)[#text(size: 20em)[#post.date · #post.title]]
  ]
]

#let timeline-entry(date, title, url, post-path) = [
  #table(
    columns: (18%, 82%),
    inset: 0.25em,
    stroke: none,
    align: (left, left),
    [#text(weight: "semibold")[#date]],
    [
      #link(url)[#text(weight: "semibold", size: 3.1em)[#title]]
      #linebreak()
      #text(size: 0.85em, fill: rgb("#555"))[Direct link: #link(url)[#url]]
      #linebreak()
      #linebreak()
      #render-post(post-path)
    ],
  )
]

#for post in posts [
  #timeline-entry(post.date, post.title, post.url, post.path)
  #linebreak()
  #linebreak()
  #linebreak()
  #linebreak()
]
