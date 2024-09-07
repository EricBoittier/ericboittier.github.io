---
title: From Juptyer to Blog
subtitle: Fun with Notebooks

# Summary for listings and search engines
summary: Automatically add notebooks to the blog?! 

# Link this post with a project
projects: []

# Date published
date: "2022-08-17T00:00:00Z"

# Date updated
date: "2023-08-17T00:00:00Z"

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
- how-to 
- juptyer
---

The following post was made
by creating a Jupyter notebook and converting it to a blog post,
using the nbconvert tool.

```bash
jupyter nbconvert --to markdown Test\ Notebook\ Blog\ Post.ipynb --NbConvertApp.output_files_dir=.
```

The command above will convert the notebook to Markdown and save it in the same directory as the notebook.
Adding the usual Hugo front matter to the markdown file will allow it to be rendered as a blog post.
Assuming you already have an index.md file with front matter, something like:

```bash
cat 'Test Notebook Blog Post.md' | tee -a index.md
```

...will do the trick!

# Test Notebook Blog Post

```python
import matplotlib.pyplot as plt
```

```python
import numpy as np
```

Let's do something a bit random!

```python
X = np.random.rand(100).reshape(10,10)
```

```python
plt.imshow(X)
```

    <matplotlib.image.AxesImage at 0x1248bd4b0>

![png](./Test%20Notebook%20Blog%20Post_5_1.png)
