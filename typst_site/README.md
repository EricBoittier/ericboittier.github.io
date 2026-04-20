# Typst Site

This is the active website source for `ericboittier.github.io`, built with Typst using the Tufted web template.

## Build

```bash
make html
```

Generated HTML is written to `_site/`.

## Preview locally (production-like)

Do not open `_site/index.html` directly with `file://...`; absolute URLs like `/assets/...` will not resolve the same way as GitHub Pages.

Use the local server target instead:

```bash
make preview
```

Then open [http://localhost:8000](http://localhost:8000). This gives the same URL/path behavior as production for CSS and links.

## Directory structure

- `config.typ` — site template config and header links
- `assets/` — static CSS assets copied into `_site/assets`
- `content/index.typ` — homepage
- `content/blog/` — blog index and post pages
- `content/publications/` — publication index and per-paper pages
- `content/cv/` — CV page and bibliography files
- `content/contact/` — contact page
- `content/all-posts/` — archive page
- `content/privacy/` — privacy placeholder page
- `content/terms/` — terms placeholder page

## Notes

- Markdown-backed migrated posts keep source text in local `post.md` files and render via `cmarker`.
- GitHub Pages deployment is handled from the repository root workflow and publishes this folder's `_site/` output.
