# Eric Boittier

Source repository for [ericboittier.github.io](https://ericboittier.github.io), built with Typst.

## Repository structure

- `typst_site/` - active website source (Typst + Tufted web template)
- `typst_resume/` - resume and cover-letter source files
- `.github/workflows/publish.yml` - GitHub Pages deploy pipeline

## Build locally

```bash
cd typst_site
make html
make preview
```

Generated files are written to `typst_site/_site/`.

## Site content map

- `typst_site/content/index.typ` - homepage
- `typst_site/content/about/` - about page
- `typst_site/content/blog/` - posts
- `typst_site/content/publications/` - publications
- `typst_site/content/projects/` - projects
- `typst_site/content/cv/` - CV page + bibliography
- `typst_site/content/contact/` - contact page
- `typst_site/content/all-posts/` - archive page

## Deployment

Pushes to `main` trigger `.github/workflows/publish.yml`, which:
- builds the site in `typst_site/`
- uploads `typst_site/_site/` as the GitHub Pages artifact
- deploys to GitHub Pages

## Open source

This site is open source. Improvements are welcome via issues or pull requests.
