#import "../index.typ": template, tufted
#show: template.with(title: "arXiv Tinder")

= arXiv Tinder

The original site included an interactive swipe-style paper browser for arXiv queries.

This Typst migration preserves the route and documents the workflow:

- Enter keywords for your area (for example, "machine learning").
- Filter by minimum publication date.
- Review candidate papers and keep matches.
- Export selected results to CSV.

The next iteration can reintroduce the full JavaScript interface as a dedicated static app mounted under this route.
