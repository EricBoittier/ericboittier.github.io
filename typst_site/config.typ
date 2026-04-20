#import "@preview/tufted:0.1.1"

#let template = tufted.tufted-web.with(
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
