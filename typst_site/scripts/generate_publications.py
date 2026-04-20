#!/usr/bin/env python3
"""Generate the Typst publications index from assets/citations.csv."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote_plus


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "assets" / "citations.csv"
OUT_PATH = ROOT / "content" / "publications" / "index.typ"


def clean(value: str) -> str:
    return " ".join((value or "").split()).strip().strip(",;")


def value(row: dict[str, str], key: str) -> str:
    target = key.lower()
    for k, v in row.items():
        if (k or "").strip().lstrip("\ufeff").lower() == target:
            return v or ""
    return ""


def fmt_authors(authors: str) -> str:
    names = [clean(x) for x in (authors or "").split(";") if clean(x)]
    if not names:
        return "Unknown authors"
    if len(names) <= 4:
        return ", ".join(names)
    return ", ".join(names[:3]) + ", et al."


def typst_escape(text: str) -> str:
    return (text or "").replace("\\", "\\\\").replace('"', '\\"')


def publication_url(title: str, publication: str) -> str:
    pub = publication or ""
    if "arxiv:" in pub.lower():
        marker = pub.lower().split("arxiv:", 1)[1].strip()
        arxiv_id = marker.split()[0].strip(".,;)")
        if arxiv_id:
            return f"https://arxiv.org/abs/{arxiv_id}"
    query = quote_plus(title)
    return f"https://scholar.google.com/scholar?q={query}"


def fmt_citation(row: dict[str, str]) -> str:
    title = clean(value(row, "Title")) or "Untitled"
    publication = clean(value(row, "Publication")) or "Unpublished"
    volume = clean(value(row, "Volume"))
    number = clean(value(row, "Number"))
    pages = clean(value(row, "Pages"))
    authors = fmt_authors(value(row, "Authors"))

    venue = publication
    if volume and number:
        venue += f", {volume}({number})"
    elif volume:
        venue += f", {volume}"
    if pages:
        venue += f", {pages}"

    url = publication_url(title, publication)
    linked_title = f'#link("{typst_escape(url)}")[{typst_escape(title)}]'
    return f"- {linked_title}  \\\n  {venue}. Authors: {authors}."


def load_rows() -> list[dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def generate() -> None:
    rows = load_rows()
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        year_raw = clean(value(row, "Year"))
        if not year_raw.isdigit():
            continue
        grouped[int(year_raw)].append(row)

    lines: list[str] = [
        '#import "../index.typ": template, tufted',
        '#show: template.with(title: "Publications")',
        "",
        "= Publications",
        "",
        "Peer-reviewed articles and preprints generated from `assets/citations.csv`.",
        "",
    ]

    for year in sorted(grouped.keys(), reverse=True):
        lines.append(f"== {year}")
        year_rows = sorted(grouped[year], key=lambda r: clean(value(r, "Title")).lower())
        for row in year_rows:
            lines.append(fmt_citation(row))
            lines.append("")

    OUT_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    generate()
