#import "@preview/tufted:0.1.1"

#let base-template = tufted.tufted-web.with(
  header-links: (
    "/": "Home",
    "/blog/": "Posts",
    "/publications/": "Publications",
    "/cv/": "CV",
    "/contact/": "Contact",
    "/all-posts/": "Archive",
  ),
  title: "Eric Boittier",
)

#let template(title: "Eric Boittier", content) = base-template.with(title: title)(
  {
    content
    html.script(src: "/assets/theme-toggle.js")
  },
)
