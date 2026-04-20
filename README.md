# ericboittier.github.io

Personal website and resume sources.

## Current structure

- `typst_site/` — active website source (Typst + Tufted template)
- `typst_resume/` — resume and cover-letter Typst sources
- `.github/workflows/publish.yml` — GitHub Pages deploy workflow (builds `typst_site`)

## Website workflow

From `typst_site/`:

```bash
make html
```

Output is generated in `typst_site/_site/`.

## Content map (Typst site)

- `typst_site/content/index.typ` — homepage
- `typst_site/content/about/` — about page
- `typst_site/content/blog/` — posts
- `typst_site/content/publications/` — publications
- `typst_site/content/projects/` — projects
- `typst_site/content/cv/` — CV page + bibliography
- `typst_site/content/contact/` — contact page
- `typst_site/content/all-posts/` — archive page
- `typst_site/content/arxiv-tinder/` — arXiv Tinder route documentation
