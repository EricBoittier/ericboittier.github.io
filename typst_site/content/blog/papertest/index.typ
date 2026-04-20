#import "../index.typ": template, tufted
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#show: template.with(title: "Paper")

= Paper (Draft Archive)

This post was previously marked as a draft in the Hugo site. It is migrated here for completeness.

Published: 2022-08-17

#{
  let md-content = read("post.md")
  let md-content = md-content.replace(regex("(?s)^---.*?---\\s*"), "")
  cmarker.render(md-content, math: mitex)
}
