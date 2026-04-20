#import "@preview/tufted:0.1.1"

#let template = tufted.tufted-web.with(
  header-links: (
    "/": "Home",
    "/about/": "About",
    "/blog/": "Posts",
    "/publications/": "Publications",
    "/projects/": "Projects",
    "/cv/": "CV",
    "/contact/": "Contact",
    "/all-posts/": "Archive",
    "/arxiv-tinder/": "arXiv Tinder",
  ),
  title: "Eric Boittier",
)
