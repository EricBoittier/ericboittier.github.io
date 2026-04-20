---
title: Notes on Baysian Optmization
subtitle: Check your priors! 

# Summary for listings and search engines
summary: If your cost function is too expensive to evaluate, you should check this out! 

# Link this post with a project
projects: []

# Date published
date: "2022-08-01T00:00:00Z"

# Date updated
date: "2022-08-01T00:00:00Z"

# Is this an unpublished draft?
draft: false

# Show this page in the Featured widget?
featured: true

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: [**Martin Krasser**](https://krasserm.github.io/2018/03/21/bayesian-optimization/)'
  focal_point: ""
  placement: 2
  preview_only: false

authors:
- admin

params:
math: true

tags:
- code
- machinelearning

categories:
- optimization
- machine learning
---

## Surrogate Models to the Rescue

If you have a cost function that is too expensive to evaluate, you should check out Bayesian Optimization.

The idea is to use a surrogate model to approximate the cost function and then use this model to find the best point to evaluate next.

The most common surrogate model is a Gaussian Process (GP), which is a distribution over functions. The GP is defined by its mean function $m(x)$ and covariance function $k(x, x')$:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

The GP is updated with the new data point and then used to find the next point to evaluate. This is typically done by maximizing an acquisition function, such as the Expected Improvement (EI):

$$EI(x) = \mathbb{E}[\max(f(x) - f(x^+), 0)]$$

where $f(x^+)$ is the current best observed value.
