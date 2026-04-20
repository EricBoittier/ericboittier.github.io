#import "../../../config.typ": template, tufted
#import "@preview/cmarker:0.1.8"
#import "@preview/mitex:0.2.6": mitex
#show: template.with(title: "Fortran is fast. Profile your code to make it faster!")

= Fortran is fast. Profile your code to make it faster!

Published: 2022-08-17

#{
  let md-content = read("post.md")
  let md-content = md-content.replace(regex("(?s)^---.*?---\\s*"), "")
  cmarker.render(md-content, math: mitex)
}
