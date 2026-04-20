#import "../../../config.typ": template, tufted
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#show: template.with(title: "Flux.1")

= Flux.1

Published: 2022-08-17 | Updated: 2023-08-17

#{
  let md-content = read("post.md")
  let md-content = md-content.replace(regex("(?s)^---.*?---\\s*"), "")
  cmarker.render(md-content, math: mitex)
}
