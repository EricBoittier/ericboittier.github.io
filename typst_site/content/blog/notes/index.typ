#import "../../../config.typ": template, tufted
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#show: template.with(title: "Notes on Bayesian Optimization")

= Notes on Bayesian Optimization

Published: 2022-08-01

#{
  let md-content = read("post.md")
  let md-content = md-content.replace(regex("(?s)^---.*?---\\s*"), "")
  cmarker.render(md-content, math: mitex)
}
